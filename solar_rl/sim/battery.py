"""Battery models. Energies are per 1-hour step, so kW and kWh are interchangeable.

SimpleBattery  The original model: fixed power limit, constant charge/discharge efficiency,
               one SoC window for grid-on and outage operation, no ageing.

LFPBattery     48 V LiFePO4 bank behind a hybrid inverter:
  * power limits: max C-rate, tapering linearly to a small C-rate between charge_taper_soc
    and full (constant-voltage phase), and between discharge_taper_soc and the cutoff
  * conversion losses that depend on power, for both directions (AC side power P):
        loss(P) = loss_const * P_r + loss_lin * P + loss_quad * P^2 / P_r     (P > 0)
    charging stores P - loss(P); discharging takes P + loss(P) from the cells
  * separate minimum SoC with the grid on (inverter "back to grid" setting) and during an
    outage (backup cutoff)
  * degradation: cycle ageing counted per half-cycle with the incremental depth-of-discharge
    method (damage of a full cycle of depth d = (d / d_ref)^k / N_ref) and calendar ageing
    (fade per year at 25 C, x q10 per +10 C). State of health falls from initial_soh; the
    usable capacity is capacity x SoH. Damage is reported as a fraction of battery life
    (1.0 = reached eol_soh), so wear cost = damage x replacement cost.
"""
import math


class SimpleBattery:
    def __init__(self, bc, soc):
        self.bc = bc
        self.soc = float(soc)
        self.soh = 1.0
        self.capacity = bc.capacity_kwh
        self.floor_grid = bc.soc_min
        self.floor_backup = bc.soc_min
        self.soc_max = bc.soc_max
        self.begin_hour()

    @property
    def energy(self):
        return self.soc * self.capacity

    def begin_hour(self):
        self.soc_start = self.soc
        self.dc_in = self.dc_out = 0.0

    def charge_limit(self):
        return self.bc.max_kw

    def discharge_limit(self, floor=None):
        return self.bc.max_kw

    def max_charge_input(self):
        room = (self.soc_max - self.soc) * self.capacity / self.bc.eta_charge
        return max(0.0, min(self.bc.max_kw, room))

    def max_discharge_output(self, floor=None):
        floor = self.floor_grid if floor is None else floor
        avail = (self.soc - floor) * self.capacity * self.bc.eta_discharge
        return max(0.0, min(self.bc.max_kw, avail))

    def charge(self, kwh_in):
        kwh_in = min(kwh_in, self.max_charge_input())
        self.soc += kwh_in * self.bc.eta_charge / self.capacity
        self.dc_in += kwh_in * self.bc.eta_charge
        return kwh_in

    def discharge(self, kwh_out, floor=None):
        kwh_out = min(kwh_out, self.max_discharge_output(floor))
        self.soc -= kwh_out / self.bc.eta_discharge / self.capacity
        self.dc_out += kwh_out / self.bc.eta_discharge
        return kwh_out

    def ac_to_reach(self, soc_target):
        return max(0.0, (soc_target - self.soc) * self.capacity / self.bc.eta_charge)

    def end_hour(self, temp_c):
        return 0.0, 0.0


class LFPBattery:
    def __init__(self, bc, soc, soh=None):
        self.bc = bc
        self.soc = float(soc)
        self.soh = bc.initial_soh if soh is None else soh
        self.floor_grid = bc.soc_min_grid
        self.floor_backup = bc.soc_cutoff_backup
        self.soc_max = bc.soc_max_lfp
        self.p_ref = bc.capacity_kwh * max(bc.max_charge_c, bc.max_discharge_c)
        self.turn_soc = self.soc        # SoC at the last direction change
        self.direction = 0              # +1 charging, -1 discharging
        self.begin_hour()

    @property
    def capacity(self):
        return self.bc.capacity_kwh * self.soh

    @property
    def energy(self):
        return self.soc * self.capacity

    # ------------------------------------------------------------- power limits
    def begin_hour(self):
        self.soc_start = self.soc
        self.dc_in = self.dc_out = 0.0
        self._charge_lim = self._taper_charge(self.soc)
        self._discharge_lim = {}

    def _taper_charge(self, soc):
        b = self.bc
        p_max, p_end = b.max_charge_c * b.capacity_kwh, b.charge_taper_end_c * b.capacity_kwh
        if soc <= b.charge_taper_soc:
            return p_max
        x = (soc - b.charge_taper_soc) / max(self.soc_max - b.charge_taper_soc, 1e-9)
        return p_max - (p_max - p_end) * min(x, 1.0)

    def _taper_discharge(self, soc, floor):
        b = self.bc
        p_max, p_end = b.max_discharge_c * b.capacity_kwh, b.discharge_taper_end_c * b.capacity_kwh
        start = max(b.discharge_taper_soc, floor)
        if soc >= start:
            return p_max
        x = (start - soc) / max(start - floor, 1e-9)
        return p_max - (p_max - p_end) * min(x, 1.0)

    def charge_limit(self):
        return self._charge_lim

    def discharge_limit(self, floor=None):
        floor = self.floor_grid if floor is None else floor
        if floor not in self._discharge_lim:
            self._discharge_lim[floor] = self._taper_discharge(self.soc_start, floor)
        return self._discharge_lim[floor]

    # ------------------------------------------------------------- losses
    def loss(self, p):
        if p <= 0:
            return 0.0
        b = self.bc
        return b.loss_const * self.p_ref + b.loss_lin * p + b.loss_quad * p * p / self.p_ref

    def _ac_for_stored(self, dc):
        """AC input that stores `dc` kWh: solve P - loss(P) = dc (smaller root)."""
        if dc <= 0:
            return 0.0
        b = self.bc
        a, bb, c = b.loss_quad / self.p_ref, -(1.0 - b.loss_lin), b.loss_const * self.p_ref + dc
        disc = bb * bb - 4 * a * c
        if a == 0:
            return c / -bb
        if disc < 0:            # cannot store that much in one hour; return the max-stored point
            return -bb / (2 * a)
        return (-bb - math.sqrt(disc)) / (2 * a)

    def _ac_from_cells(self, dc):
        """AC output obtainable from `dc` kWh of cell energy: solve P + loss(P) = dc."""
        b = self.bc
        c0 = b.loss_const * self.p_ref
        if dc <= c0:
            return 0.0
        a, bb, c = b.loss_quad / self.p_ref, 1.0 + b.loss_lin, c0 - dc
        if a == 0:
            return -c / bb
        return (-bb + math.sqrt(bb * bb - 4 * a * c)) / (2 * a)

    def efficiency(self, p, direction):
        """One-way efficiency at AC power p (kW): 'charge' or 'discharge'."""
        if p <= 0:
            return 0.0
        return (p - self.loss(p)) / p if direction == "charge" else p / (p + self.loss(p))

    # ------------------------------------------------------------- energy moves
    def max_charge_input(self):
        room_dc = (self.soc_max - self.soc) * self.capacity
        return max(0.0, min(self._charge_lim, self._ac_for_stored(room_dc)))

    def max_discharge_output(self, floor=None):
        floor = self.floor_grid if floor is None else floor
        avail_dc = (self.soc - floor) * self.capacity
        if avail_dc <= 0:
            return 0.0
        return max(0.0, min(self.discharge_limit(floor), self._ac_from_cells(avail_dc)))

    def charge(self, kwh_in):
        kwh_in = max(0.0, min(kwh_in, self.max_charge_input()))
        dc = max(kwh_in - self.loss(kwh_in), 0.0)
        self.soc = min(self.soc + dc / self.capacity, self.soc_max)
        self.dc_in += dc
        return kwh_in

    def discharge(self, kwh_out, floor=None):
        kwh_out = max(0.0, min(kwh_out, self.max_discharge_output(floor)))
        dc = kwh_out + self.loss(kwh_out) if kwh_out > 0 else 0.0
        floor = self.floor_grid if floor is None else floor
        self.soc = max(self.soc - dc / self.capacity, min(floor, self.soc))
        self.dc_out += dc
        return kwh_out

    def ac_to_reach(self, soc_target):
        return self._ac_for_stored(max(0.0, (soc_target - self.soc) * self.capacity))

    # ------------------------------------------------------------- ageing
    def _phi(self, depth):
        b = self.bc
        return (depth / b.cycle_life_dod) ** b.dod_exponent / b.cycle_life

    def end_hour(self, temp_c):
        """Book this hour's ageing. Returns (cycle damage, calendar damage) as fractions of life."""
        b = self.bc
        d_soc = self.soc - self.soc_start
        cyc = 0.0
        if abs(d_soc) > 1e-12:
            direction = 1 if d_soc > 0 else -1
            if direction != self.direction:
                self.direction, self.turn_soc = direction, self.soc_start
            d1 = abs(self.soc_start - self.turn_soc)
            d2 = abs(self.soc - self.turn_soc)
            cyc = 0.5 * (self._phi(d2) - self._phi(d1))
        rate = b.calendar_fade_per_year * b.calendar_q10 ** ((temp_c - b.calendar_ref_temp_c) / 10.0)
        cal = rate / 8760.0 / (b.initial_soh - b.eol_soh)
        self.soh -= (cyc + cal) * (b.initial_soh - b.eol_soh)
        return cyc, cal


def make_battery(bc, soc):
    return SimpleBattery(bc, soc) if bc.model == "simple" else LFPBattery(bc, soc)

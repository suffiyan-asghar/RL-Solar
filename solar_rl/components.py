"""Physical / behavioural models used by the environment.

Each model is small and independent so group members can improve one without
touching the others (e.g. replace the load model with real inverter logs).
"""
import numpy as np


# ------------------------------------------------------------------ Solar PV
def pv_power_kw(ghi, temp_c, cfg):
    """AC power (kW) from irradiance (W/m2) and air temperature (C).

    Simplification: panels are treated as horizontal (POA ~= GHI). pvlib can
    compute plane-of-array irradiance for a tilted array if you want more realism.
    """
    ghi = np.asarray(ghi, dtype=float)
    t_cell = temp_c + ghi / 800.0 * (cfg.pv_noct - 20.0)
    temp_factor = 1.0 + cfg.pv_temp_coeff * (t_cell - 25.0)
    p = cfg.pv_kwp * ghi / 1000.0 * temp_factor * (1 - cfg.pv_soiling) * cfg.pv_system_derate
    return np.clip(p, 0.0, cfg.pv_kwp)


# ------------------------------------------------------------------ Demand
def essential_kw(hour, temp_c):
    """Lights, fans, fridge, internet for a typical household (kW)."""
    fridge = 0.10
    internet = 0.03
    lights = 0.30 if (hour >= 18 or hour <= 0) else (0.10 if 6 <= hour < 18 else 0.05)
    fan_need = np.clip((temp_c - 24.0) / 8.0, 0.0, 1.0)      # 0 when cool, 1 when hot
    fans = 0.40 * fan_need * (1.0 if (hour >= 7 or hour <= 2) else 0.7)
    return fridge + internet + lights + fans


def cooling_kw(hour, temp_c, cfg):
    """Air-conditioning demand (kW): rises with temperature, used afternoons and nights."""
    occupied = (13 <= hour <= 17) or (hour >= 21 or hour <= 3)
    if not occupied:
        return 0.0
    return cfg.ac_kw * float(np.clip((temp_c - 29.0) / 5.0, 0.0, 1.0))


# ------------------------------------------------------------------ Load-shedding
class OutageModel:
    """Grid availability = scheduled blocks + random (unscheduled) outages.

    Scheduled blocks are fixed for the episode (like a published schedule).
    Unscheduled outages follow a 2-state Markov chain whose start probability
    rises in the evening and on hot days. The agent is NOT told about
    unscheduled outages in advance.
    """

    def __init__(self, cfg, level, rng):
        self.cfg, self.level, self.rng = cfg, level, rng
        self.schedule = self._make_schedule()
        self.unsched_on = False

    def _make_schedule(self):
        n = self.cfg.scheduled_blocks[self.level]
        length = self.cfg.scheduled_block_len[self.level]
        hours = set()
        if n == 0:
            return hours
        # spread blocks over the day with a random offset
        offset = int(self.rng.integers(0, 24 // n))
        for b in range(n):
            start = (offset + b * (24 // n)) % 24
            for k in range(length):
                hours.add((start + k) % 24)
        return hours

    def scheduled(self, hour):
        return hour in self.schedule

    def hours_to_next_scheduled(self, hour, cap=12):
        for k in range(1, cap + 1):
            if (hour + k) % 24 in self.schedule:
                return k
        return cap

    def step(self, hour, temp_c):
        """Advance one hour and return True if the grid is ON this hour."""
        p_start = self.cfg.outage_start_prob[self.level]
        p_start *= 1.0 + 0.15 * max(temp_c - 32.0, 0.0)   # heat stress
        if 18 <= hour <= 23:
            p_start *= 1.5                                # evening peak
        if self.unsched_on:
            if self.rng.random() < self.cfg.outage_end_prob:
                self.unsched_on = False
        elif self.rng.random() < p_start:
            self.unsched_on = True
        return not (self.unsched_on or self.scheduled(hour))


# ------------------------------------------------------------------ Battery
class Battery:
    def __init__(self, cfg, soc):
        self.cfg = cfg
        self.soc = float(soc)

    @property
    def energy(self):
        return self.soc * self.cfg.batt_kwh

    def max_charge_input(self):
        """kWh that can be pushed IN this hour (before charging losses)."""
        room = (self.cfg.soc_max - self.soc) * self.cfg.batt_kwh / self.cfg.eta_charge
        return max(0.0, min(self.cfg.batt_max_kw, room))

    def max_discharge_output(self):
        """kWh that can be delivered OUT this hour (after discharge losses)."""
        avail = (self.soc - self.cfg.soc_min) * self.cfg.batt_kwh * self.cfg.eta_discharge
        return max(0.0, min(self.cfg.batt_max_kw, avail))

    def charge(self, kwh_in):
        kwh_in = min(kwh_in, self.max_charge_input())
        self.soc += kwh_in * self.cfg.eta_charge / self.cfg.batt_kwh
        return kwh_in

    def discharge(self, kwh_out):
        kwh_out = min(kwh_out, self.max_discharge_output())
        self.soc -= kwh_out / self.cfg.eta_discharge / self.cfg.batt_kwh
        return kwh_out

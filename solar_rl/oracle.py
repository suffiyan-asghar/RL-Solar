"""Perfect-foresight oracle: a linear program over the whole episode.

At the start of an episode the oracle copies the simulator and plays the copy forward to
read the future exactly: solar, base demand, outdoor temperature, grid status (outages are
exogenous: they do not depend on actions). It then solves one LP (scipy HiGHS) for the
cheapest schedule and replays it hour by hour through the real simulator with
Dispatch(battery="setpoint", ...).

LP per hour t (all kWh):
  balance   pv - curt + imp + dis = (E - uE) + (O - uO) + ac + f + exp + ch + standby
  battery   e[t+1] = e[t] + eta_c ch - dis / eta_d,  floor_t * cap <= e[t+1] <= soc_max * cap
            0 <= ch <= P_ch,  0 <= dis <= P_dis      (no taper; constant efficiencies)
  grid off  imp = exp = 0, served load <= backup limit; grid on: uE = uO = 0
  room      T[t+1] = a T[t] + (1 - a) (T_out + R (A G + Q_int - COP ac)),  comfort x[t] >= mean T - setpoint
  flexible  sum of f over each day up to the deadline + missed_day = energy due
  objective import cost (incl. energy taxes) - export credit + cycle wear + penalties
Simplifications make it an approximate bound: constant efficiencies at 30 % of rated power,
no power taper, cycle wear linear in throughput (exact for dod_exponent = 1), no calendar wear
(constant). The replayed result is what the real simulator gives for the LP's plan.
"""
import copy

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .sim import Dispatch


def peek(env):
    """Exogenous future of the current episode, read from a copy of the simulator."""
    sim = copy.deepcopy(env.sim)
    sim.record = False
    n = env.cfg.episode.hours - sim.t_in_episode
    rows = []
    flex_due = []
    day_start_remaining = sim.flex_remaining
    for k in range(n):
        t = sim.t
        cc = sim.demand.cooling_cfg(t)
        fc = sim.demand.flex_cfg(t)
        rows.append(dict(t=t, hour=int(sim.hours[t]), month=int(sim.months[t]), grid_on=bool(sim.grid_on),
                         pv=float(sim.pv[t]), E=float(sim.cur_essential), O=float(sim.cur_other),
                         rule_ac=float(sim.cur_rule_cooling) if cc is not None and cc.model == "rule" else 0.0,
                         T_out=float(sim.temp[t]), ghi=float(sim.ghi[t]), cc=cc, fc=fc,
                         flex_start=sim.flex_remaining if (k == 0 or sim.hours[t] == 0) else None))
        sim.step(Dispatch(battery="hold", ac="off"))
    return rows, day_start_remaining


def solve(env):
    c = env.cfg
    rows, _ = peek(env)
    T = len(rows)
    b = env.sim.batt
    bc = c.battery
    cap = b.capacity
    if bc.model == "simple":
        eta_c, eta_d = bc.eta_charge, bc.eta_discharge
        p_ch = p_dis = bc.max_kw
    else:
        p_ch, p_dis = bc.max_charge_c * bc.capacity_kwh, bc.max_discharge_c * bc.capacity_kwh
        p_typ = 0.3 * max(p_ch, p_dis)
        eta_c, eta_d = b.efficiency(p_typ, "charge"), b.efficiency(p_typ, "discharge")
    floor_on, floor_off, soc_max = b.floor_grid, b.floor_backup, b.soc_max
    rw = c.reward
    tariff = env.tariff
    replacement = env.replacement_cost
    if rw.wear == "flat":
        wear_dis, wear_dc = rw.flat_wear_rs_per_kwh, 0.0
    else:
        wear_dis, wear_dc = 0.0, replacement * 0.5 / (bc.cycle_life_dod * bc.cycle_life * cap)
    standby = c.inverter.standby_w / 1000.0
    backup = c.inverter.backup_limit_kw
    thermal = env.sim.room is not None and rows[0]["cc"] is not None and rows[0]["cc"].model == "thermal"

    # variable layout
    names = ["imp", "exp", "ch", "dis", "curt", "uE", "uO", "ac", "uC", "f", "x"]
    nv = len(names)
    idx = {nm: i for i, nm in enumerate(names)}

    def v(nm, t):
        return t * nv + idx[nm]
    e0 = T * nv                      # battery energy e[1..T]
    tr0 = e0 + T                     # room temperature T[1..T]
    days = []                        # flexible-load day segments
    for t, r in enumerate(rows):
        if r["flex_start"] is not None:
            days.append({"due": r["flex_start"], "hours": []})
        if days and r["hour"] <= r["fc"].deadline_hour:
            days[-1]["hours"].append(t)
    m0 = tr0 + (T if thermal else 0)
    N = m0 + len(days)
    cost = np.zeros(N)
    lb, ub = np.zeros(N), np.full(N, np.inf)
    A_eq, b_eq, A_ub, b_ub = [], [], [], []

    def row(entries):
        r = np.zeros(N)
        for j, val in entries:
            r[j] += val
        return r

    room = env.sim.room
    for t, r in enumerate(rows):
        price = tariff.import_rate(r["month"], r["hour"]) * (1.0 + tariff.energy_tax)
        cost[v("imp", t)] = price
        cost[v("exp", t)] = -tariff.export_rate
        cost[v("ch", t)] = wear_dc * eta_c
        cost[v("dis", t)] = wear_dis + wear_dc / eta_d
        cost[v("uE", t)] = rw.penalty_unserved_essential
        cost[v("uO", t)] = rw.penalty_unserved_other
        cost[v("uC", t)] = rw.penalty_unmet_cooling
        cost[v("x", t)] = rw.penalty_comfort_degree_hour
        ub[v("ch", t)], ub[v("dis", t)] = p_ch, p_dis
        ub[v("curt", t)] = r["pv"]
        cc, fc = r["cc"], r["fc"]
        on = r["grid_on"]
        if not on:
            ub[v("imp", t)] = ub[v("exp", t)] = 0.0
        ub[v("uE", t)] = 0.0 if on else r["E"]
        ub[v("uO", t)] = 0.0 if on else r["O"]
        ub[v("f", t)] = fc.max_kw if t in sum((d["hours"] for d in days), []) else 0.0
        if thermal:
            ub[v("ac", t)] = cc.ac_max_kw_elec
            ub[v("uC", t)] = 0.0
        else:
            lb[v("ac", t)] = ub[v("ac", t)] = r["rule_ac"]
            ub[v("uC", t)] = 0.0 if on else r["rule_ac"]
        if not thermal or cc is None or r["hour"] not in cc.occupied_hours:
            ub[v("x", t)] = 0.0
        # energy balance
        A_eq.append(row([(v("imp", t), 1), (v("dis", t), 1), (v("curt", t), -1), (v("uE", t), 1), (v("uO", t), 1),
                         (v("ac", t), -1), (v("uC", t), 1), (v("f", t), -1), (v("exp", t), -1), (v("ch", t), -1)]))
        b_eq.append(r["E"] + r["O"] + standby - r["pv"])
        # battery dynamics
        prev = [] if t == 0 else [(e0 + t - 1, -1)]
        A_eq.append(row([(e0 + t, 1), (v("ch", t), -eta_c), (v("dis", t), 1 / eta_d)] + prev))
        b_eq.append(b.energy if t == 0 else 0.0)
        # the simulator never discharges below the floor but does not force the battery up to it
        # (e.g. right after an outage SoC can sit below the grid-on floor), so neither does the LP
        lb[e0 + t] = min((floor_on if on else floor_off) * cap, b.energy - 1e-9)
        ub[e0 + t] = soc_max * cap
        if not on and np.isfinite(backup):
            A_ub.append(row([(v("uE", t), -1), (v("uO", t), -1), (v("ac", t), 1), (v("uC", t), -1), (v("f", t), 1)]))
            b_ub.append(backup - r["E"] - r["O"])
        if thermal:
            a = room.decay
            R = cc.r_c_per_kw
            gain = cc.solar_aperture_m2 * r["ghi"] / 1000.0 + cc.internal_gain_kw
            # T[t+1] = a T[t] + (1-a)(T_out + R gain) - (1-a) R COP ac
            prevT = [] if t == 0 else [(tr0 + t - 1, -a)]
            A_eq.append(row([(tr0 + t, 1), (v("ac", t), (1 - a) * R * cc.ac_cop)] + prevT))
            b_eq.append((1 - a) * (r["T_out"] + R * gain) + (a * room.temp if t == 0 else 0.0))
            lb[tr0 + t] = -np.inf
            if r["hour"] in cc.occupied_hours:
                # mean temp = T_eq + (T0 - T_eq) k,  k = tau (1 - a);  T_eq = T_out + R gain - R COP ac
                k = room.tau * (1 - a)
                t0 = [] if t == 0 else [(tr0 + t - 1, k)]
                const_t0 = k * room.temp if t == 0 else 0.0
                # x >= (1-k)(T_out + R gain) - (1-k) R COP ac + k T0 - setpoint
                A_ub.append(row([(v("x", t), -1), (v("ac", t), -(1 - k) * R * cc.ac_cop)] + t0))
                b_ub.append(cc.setpoint_c - (1 - k) * (r["T_out"] + R * gain) - const_t0)
    for i, d in enumerate(days):
        A_eq.append(row([(v("f", t), 1) for t in d["hours"]] + [(m0 + i, 1)]))
        b_eq.append(d["due"])
        cost[m0 + i] = rw.penalty_flex_missed

    res = linprog(cost, A_ub=sparse.csr_matrix(np.array(A_ub)) if A_ub else None, b_ub=b_ub or None,
                  A_eq=sparse.csr_matrix(np.array(A_eq)), b_eq=b_eq, bounds=list(zip(lb, ub)), method="highs")
    if res.status != 0:
        raise RuntimeError(f"oracle LP failed: {res.message}")
    x = res.x
    calendar = 0.0
    if rw.wear != "flat" and rw.include_calendar_wear:
        for r in rows:
            rate = bc.calendar_fade_per_year * bc.calendar_q10 ** ((r["T_out"] - bc.calendar_ref_temp_c) / 10.0)
            calendar += rate / 8760.0 / (bc.initial_soh - bc.eol_soh) * replacement
    energy = [x[e0 + t] for t in range(T)]
    plan = []
    for t in range(T):
        plan.append(Dispatch(battery="setpoint", charge_kw=x[v("ch", t)], discharge_kw=x[v("dis", t)],
                             flex_kw=x[v("f", t)], run_flex=x[v("f", t)] > 1e-6,
                             ac_kw=x[v("ac", t)] if thermal else None))
    return plan, float(res.fun) + calendar, energy


class OraclePolicy:
    """Replays the LP plan; re-solves from the current state (with the remaining foresight)
    whenever the real battery energy drifts more than `replan_tol` x capacity from the plan."""
    name = "oracle"

    def __init__(self, replan_tol=0.03):
        self.replan_tol = replan_tol
        self.plan = self.lp_objective = self.energy = None
        self.replans = 0

    def act(self, env, obs):
        k = env.sim.t_in_episode
        b = env.sim.batt
        if k == 0 or self.plan is None:
            self.plan, self.lp_objective, self.energy = solve(env)
            self.offset, self.replans = 0, 0
        elif k > self.offset and abs(b.energy - self.energy[k - self.offset - 1]) > self.replan_tol * b.capacity:
            try:
                self.plan, _, self.energy = solve(env)
                self.offset = k
                self.replans += 1
            except RuntimeError:        # keep following the previous plan
                pass
        return self.plan[k - self.offset]

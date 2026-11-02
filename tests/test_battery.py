"""LFP battery: power limits and taper, efficiency, SoC floors, degradation."""
import numpy as np
import pytest

from solar_rl import load_config
from solar_rl.sim.battery import LFPBattery, SimpleBattery


@pytest.fixture
def bc():
    return load_config("default").battery


def fresh(bc, soc):
    b = LFPBattery(bc, soc)
    b.begin_hour()
    return b


# ---------------------------------------------------------------- limits
def test_charge_limit_full_rate_below_taper(bc):
    b = fresh(bc, 0.5)
    assert b.charge_limit() == pytest.approx(bc.max_charge_c * bc.capacity_kwh)
    assert b.charge(100.0) == pytest.approx(5.0)              # 0.5 C x 10 kWh


def test_charge_limit_tapers_near_full(bc):
    lims = [fresh(bc, s).charge_limit() for s in (0.90, 0.93, 0.96, 0.99, 1.0)]
    assert lims[0] == pytest.approx(5.0)
    assert all(a > b for a, b in zip(lims, lims[1:]))
    assert lims[-1] == pytest.approx(bc.charge_taper_end_c * bc.capacity_kwh)


def test_discharge_limit_tapers_near_cutoff(bc):
    hi = fresh(bc, 0.5).discharge_limit(bc.soc_min_grid)
    lo = fresh(bc, 0.12).discharge_limit(bc.soc_min_grid)
    assert hi == pytest.approx(5.0)
    assert lo < hi
    assert fresh(bc, 0.10).discharge_limit(0.10) == pytest.approx(bc.discharge_taper_end_c * bc.capacity_kwh)


def test_never_exceeds_soc_window(bc):
    b = fresh(bc, 0.97)
    for _ in range(10):
        b.begin_hour()
        b.charge(10.0)
    assert b.soc <= bc.soc_max_lfp + 1e-12
    b = fresh(bc, 0.2)
    for _ in range(10):
        b.begin_hour()
        b.discharge(10.0, b.floor_grid)
    assert b.soc == pytest.approx(bc.soc_min_grid, abs=1e-9)
    for _ in range(10):
        b.begin_hour()
        b.discharge(10.0, b.floor_backup)
    assert b.soc == pytest.approx(bc.soc_cutoff_backup, abs=1e-9)


def test_discharge_below_floor_does_not_create_energy(bc):
    b = fresh(bc, 0.07)                       # below the grid-on floor after an outage
    assert b.discharge(1.0, b.floor_grid) == 0.0
    assert b.soc == pytest.approx(0.07)


# ---------------------------------------------------------------- efficiency
def test_efficiency_depends_on_power(bc):
    b = fresh(bc, 0.5)
    eff = [b.efficiency(p, "discharge") for p in (0.1, 0.5, 1.5, 3.0, 5.0)]
    assert eff[0] < eff[2]                    # fixed losses dominate at low power
    assert eff[4] < eff[2]                    # resistive losses dominate at high power
    rt = b.efficiency(1.5, "charge") * b.efficiency(1.5, "discharge")
    assert 0.88 < rt < 0.93                   # ~90 % round trip at typical power


def test_round_trip_energy(bc):
    b = fresh(bc, 0.3)
    e0 = b.energy
    ac_in = b.charge(2.0)
    stored = b.energy - e0
    b.begin_hour()
    ac_out = b.discharge(100.0, floor=0.3)
    assert ac_in == pytest.approx(2.0)
    assert stored == pytest.approx(2.0 - b.loss(2.0))
    assert b.energy == pytest.approx(e0, abs=1e-9)
    assert ac_out / ac_in == pytest.approx(b.efficiency(2.0, "charge") * b.efficiency(ac_out, "discharge"), rel=1e-6)


def test_ac_to_reach_hits_target(bc):
    b = fresh(bc, 0.2)
    b.charge(b.ac_to_reach(0.4))
    assert b.soc == pytest.approx(0.4, abs=1e-9)


# ---------------------------------------------------------------- degradation
def cycle(b, depth, n, temp=25.0):
    """n full cycles of the given depth (in SoC), 1 kW steps."""
    cyc = cal = 0.0
    top = 0.95
    for _ in range(n):
        for target, fn in ((top - depth, "discharge"), (top, "charge")):
            while abs(b.soc - target) > 1e-6:
                b.begin_hour()
                if fn == "discharge":
                    need = (b.soc - target) * b.capacity
                    b.discharge(min(1.0, b._ac_from_cells(need)), floor=0.0)
                else:
                    b.charge(min(1.0, b.ac_to_reach(target)))
                c, k = b.end_hour(temp)
                cyc += c
                cal += k
    return cyc, cal


def test_cycle_damage_matches_datasheet(bc):
    b = LFPBattery(bc, 0.95)
    cyc, _ = cycle(b, bc.cycle_life_dod, 10)
    assert cyc == pytest.approx(10 / bc.cycle_life, rel=1e-6)       # 6000 cycles at 95 % DoD
    cost = bc.replacement_cost_rs_per_kwh * bc.capacity_kwh
    per_kwh = cyc * cost / (10 * bc.cycle_life_dod * bc.capacity_kwh)
    assert per_kwh == pytest.approx(7.0, rel=0.01)                    # = old flat Rs 7/kWh


def test_dod_exponent(bc):
    deep_k = load_config("default", {"battery": {"dod_exponent": 1.5}}).battery
    shallow, _ = cycle(LFPBattery(deep_k, 0.95), 0.2, 5)
    deep, _ = cycle(LFPBattery(deep_k, 0.95), 0.8, 5)
    # per unit of energy throughput, deep cycles age faster when k > 1
    assert deep / 0.8 > shallow / 0.2 * 1.5
    lin_s, _ = cycle(LFPBattery(bc, 0.95), 0.2, 5)
    lin_d, _ = cycle(LFPBattery(bc, 0.95), 0.8, 5)
    assert lin_d / 0.8 == pytest.approx(lin_s / 0.2, rel=1e-6)       # k = 1: throughput only


def test_calendar_ageing(bc):
    b = LFPBattery(bc, 0.5)
    for _ in range(8760):
        b.begin_hour()
        b.end_hour(25.0)
    assert 1.0 - b.soh == pytest.approx(bc.calendar_fade_per_year, rel=1e-6)
    hot = LFPBattery(bc, 0.5)
    for _ in range(8760):
        hot.begin_hour()
        hot.end_hour(35.0)
    assert 1.0 - hot.soh == pytest.approx(bc.calendar_fade_per_year * bc.calendar_q10, rel=1e-6)


def test_capacity_follows_soh(bc):
    b = LFPBattery(bc, 0.5)
    b.soh = 0.8
    assert b.capacity == pytest.approx(0.8 * bc.capacity_kwh)


def test_simple_battery_unchanged():
    bc = load_config("legacy_v1").battery
    b = SimpleBattery(bc, 0.5)
    assert b.charge(10) == pytest.approx(3.0)
    assert b.soc == pytest.approx(0.5 + 3.0 * 0.95 / 10)
    assert b.end_hour(30.0) == (0.0, 0.0)
    assert np.isclose(b.floor_grid, b.floor_backup)

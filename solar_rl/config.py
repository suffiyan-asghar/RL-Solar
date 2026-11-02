"""All simulator settings in one place.

Every number here is an ASSUMPTION that the group should check and, where
possible, replace with real values (tariff notices, battery datasheets,
inverter logs). Change values here, not inside the code.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SimConfig:
    # ---------------- Location & data ----------------
    latitude: float = 24.86          # Karachi
    longitude: float = 67.01
    timezone: str = "Asia/Karachi"
    data_dir: str = "data"           # where karachi_<year>.csv files live
    years: List[int] = field(default_factory=lambda: [2020, 2021, 2022])

    # ---------------- Episode ----------------
    episode_hours: int = 168         # one week, 1-hour steps
    forecast_hours: int = 6          # look-ahead given to the agent
    forecast_noise: float = 0.15     # relative std of forecast error (grows with lead time)

    # ---------------- Solar PV ----------------
    pv_kwp: float = 5.0              # array size (kW peak)
    pv_temp_coeff: float = -0.004    # power change per deg C above 25 C (cell temp)
    pv_noct: float = 45.0            # nominal operating cell temperature
    pv_soiling: float = 0.05         # dust loss (fraction)
    pv_system_derate: float = 0.90   # wiring, mismatch, inverter losses

    # ---------------- Battery ----------------
    batt_kwh: float = 10.0
    soc_min: float = 0.10
    soc_max: float = 0.95
    batt_max_kw: float = 3.0         # max charge / discharge power
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    wear_cost_per_kwh: float = 7.0   # Rs per kWh discharged (battery price / lifetime throughput)

    # ---------------- Demand (household) ----------------
    ac_kw: float = 1.5               # one inverter AC
    flex_daily_kwh: float = 1.5      # water motor + washing machine + iron
    flex_max_kw: float = 1.0         # flexible load that can run per hour
    flex_deadline_hour: int = 22     # flexible energy must be done by this hour
    load_noise: float = 0.15         # relative std of hourly demand noise

    # ---------------- Grid / tariffs (Rs per kWh) ----------------
    import_offpeak: float = 42.0
    import_peak: float = 50.0
    peak_hours: tuple = (18, 19, 20, 21, 22)
    export_rate: float = 11.0        # net billing buyback (legacy net metering ~26)

    # ---------------- Load-shedding ----------------
    # "light" | "medium" | "heavy" | "random" (random = domain randomisation)
    outage_level: str = "random"
    # base probability per hour that an UNscheduled outage starts, per level
    outage_start_prob: dict = field(default_factory=lambda: {"light": 0.01, "medium": 0.03, "heavy": 0.06})
    outage_end_prob: float = 0.40    # chance an unscheduled outage ends each hour (mean ~2.5 h)
    # scheduled blocks per day and block length (hours), per level
    scheduled_blocks: dict = field(default_factory=lambda: {"light": 0, "medium": 2, "heavy": 3})
    scheduled_block_len: dict = field(default_factory=lambda: {"light": 0, "medium": 2, "heavy": 3})

    # ---------------- Reward weights (Rs per kWh) ----------------
    penalty_unserved_essential: float = 200.0   # lambda_1: value of lost essential load
    penalty_unmet_cooling: float = 30.0         # lambda_3: discomfort when AC is shed
    penalty_flex_missed: float = 60.0           # flexible energy not done by deadline
    reward_scale: float = 0.01                  # rewards are divided to ~unit scale

    # ---------------- Misc ----------------
    allow_synthetic_weather: bool = True
    seed: Optional[int] = None


LEVELS = ["light", "medium", "heavy"]

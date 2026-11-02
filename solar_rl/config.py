"""Simulator settings: loaded from YAML (configs/), validated by dataclasses.

    cfg = load_config()                       # configs/default.yaml
    cfg = load_config("shop")                 # configs/shop.yaml merged on top of default.yaml
    cfg = load_config("default", {"pv": {"kwp": 8.0}})   # plus overrides
    cfg = load_config("results/run/config.yaml")         # a saved, fully resolved config

Every number is an ASSUMPTION; see docs/ASSUMPTIONS.md. Change values in the YAML files,
not in code. Unknown keys and out-of-range values raise ConfigError.
"""
import copy
import dataclasses
import datetime as dt
import os
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "configs")
LEVELS = ["light", "medium", "heavy"]


class ConfigError(ValueError):
    pass


def _check(cond, msg):
    if not cond:
        raise ConfigError(msg)


def _frac(name, x):
    _check(0.0 <= x <= 1.0, f"{name} must be in [0, 1], got {x}")


def _pos(name, x, strict=False):
    _check(x > 0 if strict else x >= 0, f"{name} must be {'>' if strict else '>='} 0, got {x}")


def _hours(name, hours):
    _check(all(isinstance(h, int) and 0 <= h <= 23 for h in hours), f"{name}: hours must be ints 0-23")


# ----------------------------------------------------------------------------- sections
@dataclass
class SiteConfig:
    name: str = "Karachi"
    latitude: float = 24.86
    longitude: float = 67.01
    timezone: str = "Asia/Karachi"

    def __post_init__(self):
        _check(-90 <= self.latitude <= 90, "site.latitude out of range")
        _check(-180 <= self.longitude <= 180, "site.longitude out of range")


@dataclass
class DataConfig:
    dir: str = "data"
    years: List[int] = field(default_factory=lambda: [2020, 2021, 2022])
    allow_synthetic: bool = True

    def __post_init__(self):
        _check(len(self.years) > 0, "data.years must not be empty")


@dataclass
class EpisodeConfig:
    hours: int = 168
    initial_soc_range: List[float] = field(default_factory=lambda: [0.2, 0.9])
    forecast_hours: int = 6
    solar_forecast_noise: float = 0.15
    temp_forecast_noise_c: float = 1.5

    def __post_init__(self):
        _pos("episode.hours", self.hours, strict=True)
        _check(len(self.initial_soc_range) == 2 and self.initial_soc_range[0] <= self.initial_soc_range[1],
               "episode.initial_soc_range must be [low, high]")
        _pos("episode.forecast_hours", self.forecast_hours, strict=True)


@dataclass
class RngConfig:
    streams: str = "independent"

    def __post_init__(self):
        _check(self.streams in ("independent", "shared"), "rng.streams must be independent|shared")


@dataclass
class PVConfig:
    model: str = "pvlib"
    kwp: float = 5.0
    temp_coeff: float = -0.004
    noct: float = 45.0
    soiling: float = 0.05
    system_derate: float = 0.90
    tilt: float = 20.0
    azimuth: float = 180.0
    albedo: float = 0.20
    transposition: str = "haydavies"
    iam_b0: Optional[float] = 0.05
    faiman_u0: float = 25.0
    faiman_u1: float = 6.84
    dc_losses: float = 0.0681
    inverter_ac_kw: Optional[float] = None
    inverter_eff: float = 0.96
    inverter_eff_ref: float = 0.9637
    soiling_model: str = "kimber"
    soiling_rate_per_day: float = 0.008
    soiling_max: float = 0.30
    rain_clean_mm: float = 6.0
    rain_grace_days: int = 14
    cleaning_interval_days: Optional[int] = 14

    def __post_init__(self):
        _check(self.model in ("pvlib", "simple"), "pv.model must be pvlib|simple")
        _pos("pv.kwp", self.kwp)
        for n in ("soiling", "system_derate", "albedo", "dc_losses", "soiling_max"):
            _frac(f"pv.{n}", getattr(self, n))
        _check(0 <= self.tilt <= 90, "pv.tilt must be 0-90 degrees")
        _check(0 <= self.azimuth <= 360, "pv.azimuth must be 0-360 degrees")
        _check(self.transposition in ("isotropic", "haydavies", "perez"), "pv.transposition invalid")
        _check(0 < self.inverter_eff <= 1 and 0 < self.inverter_eff_ref <= 1, "pv inverter efficiencies in (0, 1]")
        _check(self.soiling_model in ("kimber", "constant", "none"), "pv.soiling_model must be kimber|constant|none")
        _pos("pv.soiling_rate_per_day", self.soiling_rate_per_day)
        _check(self.cleaning_interval_days is None or self.cleaning_interval_days > 0,
               "pv.cleaning_interval_days must be > 0 or null")


@dataclass
class InverterConfig:
    rated_kw: float = 6.0
    backup_max_kw: Optional[float] = 6.0
    transfer_time_ms: float = 10.0
    standby_w: float = 20.0

    def __post_init__(self):
        _pos("inverter.rated_kw", self.rated_kw, strict=True)
        _check(self.backup_max_kw is None or self.backup_max_kw > 0, "inverter.backup_max_kw must be > 0 or null")
        _pos("inverter.transfer_time_ms", self.transfer_time_ms)
        _pos("inverter.standby_w", self.standby_w)

    @property
    def backup_limit_kw(self):
        return float("inf") if self.backup_max_kw is None else self.backup_max_kw


@dataclass
class BatteryConfig:
    model: str = "lfp"
    capacity_kwh: float = 10.0
    initial_soh: float = 1.0
    soc_min: float = 0.10
    soc_max: float = 0.95
    max_kw: float = 3.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc_min_grid: float = 0.10
    soc_cutoff_backup: float = 0.05
    soc_max_lfp: float = 1.00
    max_charge_c: float = 0.5
    max_discharge_c: float = 0.5
    charge_taper_soc: float = 0.90
    charge_taper_end_c: float = 0.05
    discharge_taper_soc: float = 0.15
    discharge_taper_end_c: float = 0.10
    loss_const: float = 0.002
    loss_lin: float = 0.03
    loss_quad: float = 0.03
    cycle_life: float = 6000
    cycle_life_dod: float = 0.95
    dod_exponent: float = 1.0
    eol_soh: float = 0.70
    calendar_fade_per_year: float = 0.02
    calendar_ref_temp_c: float = 25.0
    calendar_q10: float = 1.35
    replacement_cost_rs_per_kwh: float = 39900.0

    def __post_init__(self):
        _check(self.model in ("lfp", "simple"), "battery.model must be lfp|simple")
        _pos("battery.capacity_kwh", self.capacity_kwh, strict=True)
        for n in ("initial_soh", "soc_min", "soc_max", "eta_charge", "eta_discharge", "soc_min_grid",
                  "soc_cutoff_backup", "soc_max_lfp", "charge_taper_soc", "discharge_taper_soc",
                  "cycle_life_dod", "eol_soh", "calendar_fade_per_year"):
            _frac(f"battery.{n}", getattr(self, n))
        _check(self.soc_min < self.soc_max, "battery.soc_min must be < soc_max")
        _check(self.soc_cutoff_backup <= self.soc_min_grid < self.soc_max_lfp,
               "battery: need soc_cutoff_backup <= soc_min_grid < soc_max_lfp")
        _check(self.soc_min_grid <= self.discharge_taper_soc <= self.charge_taper_soc <= self.soc_max_lfp,
               "battery: need soc_min_grid <= discharge_taper_soc <= charge_taper_soc <= soc_max_lfp")
        _check(0 < self.charge_taper_end_c <= self.max_charge_c, "battery.charge_taper_end_c must be in (0, max_charge_c]")
        _check(0 < self.discharge_taper_end_c <= self.max_discharge_c,
               "battery.discharge_taper_end_c must be in (0, max_discharge_c]")
        for n in ("max_kw", "max_charge_c", "max_discharge_c", "cycle_life", "dod_exponent", "calendar_q10"):
            _pos(f"battery.{n}", getattr(self, n), strict=True)
        for n in ("loss_const", "loss_lin", "loss_quad", "replacement_cost_rs_per_kwh"):
            _pos(f"battery.{n}", getattr(self, n))
        _check(self.loss_lin + self.loss_quad + self.loss_const < 0.5, "battery losses unrealistically high")
        _check(self.eol_soh < self.initial_soh, "battery.eol_soh must be < initial_soh")


@dataclass
class ApplianceConfig:
    name: str
    kw: float
    priority: str = "essential"
    hours: Any = "all"
    probability: float = 1.0
    temp_scale: Optional[List[float]] = None
    hour_factor: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        _pos(f"appliance {self.name}.kw", self.kw)
        _check(self.priority in ("essential", "other"), f"appliance {self.name}: priority must be essential|other")
        if self.hours == "all":
            self.hours = list(range(24))
        _hours(f"appliance {self.name}", self.hours)
        _frac(f"appliance {self.name}.probability", self.probability)
        if self.temp_scale is not None:
            _check(len(self.temp_scale) == 2 and self.temp_scale[0] < self.temp_scale[1],
                   f"appliance {self.name}: temp_scale must be [low, high]")
        self.hour_factor = {int(k): float(v) for k, v in self.hour_factor.items()}


@dataclass
class CoolingConfig:
    model: str = "thermal"          # thermal | rule | none
    # rule (original model)
    kw: float = 1.5
    hours: List[int] = field(default_factory=list)
    temp_low_c: float = 29.0
    temp_high_c: float = 34.0
    # thermal (first-order RC room)
    r_c_per_kw: float = 6.0
    c_kwh_per_c: float = 1.0
    solar_aperture_m2: float = 0.5
    internal_gain_kw: float = 0.2
    ac_cooling_kw: float = 5.3
    ac_cop: float = 3.3
    occupied_hours: List[int] = field(default_factory=list)
    setpoint_c: float = 26.0
    precool_setpoint_c: float = 23.0
    initial_temp_c: float = 26.0

    def __post_init__(self):
        _check(self.model in ("thermal", "rule", "none"), "cooling.model must be thermal|rule|none")
        _hours("cooling.hours", self.hours)
        _hours("cooling.occupied_hours", self.occupied_hours)
        for n in ("r_c_per_kw", "c_kwh_per_c", "ac_cop"):
            _pos(f"cooling.{n}", getattr(self, n), strict=True)
        for n in ("kw", "solar_aperture_m2", "internal_gain_kw", "ac_cooling_kw"):
            _pos(f"cooling.{n}", getattr(self, n))
        _check(self.precool_setpoint_c <= self.setpoint_c, "cooling.precool_setpoint_c must be <= setpoint_c")
        _check(self.temp_low_c < self.temp_high_c, "cooling.temp_low_c must be < temp_high_c")

    @property
    def ac_max_kw_elec(self):
        return self.ac_cooling_kw / self.ac_cop


@dataclass
class FlexConfig:
    daily_kwh: float = 1.5
    max_kw: float = 1.0
    deadline_hour: int = 22

    def __post_init__(self):
        _pos("flexible.daily_kwh", self.daily_kwh)
        _pos("flexible.max_kw", self.max_kw, strict=True)
        _check(0 <= self.deadline_hour <= 23, "flexible.deadline_hour must be 0-23")


@dataclass
class DemandProfile:
    name: str
    appliances: List[ApplianceConfig]
    cooling: CoolingConfig
    flexible: FlexConfig
    variants: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        names = [a.name for a in self.appliances]
        _check(len(names) == len(set(names)), f"profile {self.name}: duplicate appliance names")
        _check(set(self.variants) <= {"weekend", "ramadan"}, "profile variants must be weekend|ramadan")
        for v in self.variants.values():
            _check(set(v) <= {"appliances", "cooling", "flexible"}, "variant keys: appliances|cooling|flexible")

    def variant(self, weekend: bool, ramadan: bool) -> "DemandProfile":
        """Profile for a day type: base, then weekend overrides, then Ramadan overrides."""
        d = profile_to_dict(self)
        for key, on in (("weekend", weekend), ("ramadan", ramadan)):
            if on and key in self.variants:
                d = merge_profile(d, self.variants[key])
        d["variants"] = {}
        return build(DemandProfile, d, "profile")


@dataclass
class DemandConfig:
    model: str = "appliances"
    profile: Any = "household"
    profile_overrides: Dict[str, Any] = field(default_factory=dict)
    noise: float = 0.15
    measured_log: Optional[str] = None
    measured_priority: str = "essential"
    weekend_days: List[int] = field(default_factory=lambda: [6])
    ramadan_dates: Dict[int, List[Any]] = field(default_factory=dict)

    def __post_init__(self):
        _check(self.model in ("appliances", "measured"), "demand.model must be appliances|measured")
        _check(self.model != "measured" or self.measured_log, "demand.measured_log is required for model: measured")
        _check(self.measured_priority in ("essential", "other"), "demand.measured_priority must be essential|other")
        _pos("demand.noise", self.noise)
        _check(all(0 <= d <= 6 for d in self.weekend_days), "demand.weekend_days must be 0 (Mon) - 6 (Sun)")
        rd = {}
        for y, rng in (self.ramadan_dates or {}).items():
            _check(len(rng) == 2, f"demand.ramadan_dates[{y}] must be [first, last]")
            rd[int(y)] = [_date(x) for x in rng]
        self.ramadan_dates = rd
        if isinstance(self.profile, str):
            raw = load_yaml(os.path.join(CONFIG_DIR, "demand", f"{self.profile}.yaml"))
        elif isinstance(self.profile, DemandProfile):
            raw = profile_to_dict(self.profile)
        else:
            raw = copy.deepcopy(self.profile)
        if self.profile_overrides:
            raw = merge_profile(raw, self.profile_overrides)
            self.profile_overrides = {}
        self.profile = build(DemandProfile, raw, "demand.profile")


@dataclass
class ScheduleConfig:
    source: str = "synthetic"
    csv_path: Optional[str] = None
    category_by_level: Dict[str, Optional[str]] = field(default_factory=dict)
    blocks_per_day: Dict[str, int] = field(default_factory=lambda: {"light": 0, "medium": 2, "heavy": 3})
    block_hours: Dict[str, int] = field(default_factory=lambda: {"light": 0, "medium": 2, "heavy": 3})

    def __post_init__(self):
        _check(self.source in ("synthetic", "csv", "none"), "outages.schedule.source must be synthetic|csv|none")
        _check(self.source != "csv" or self.csv_path, "outages.schedule.csv_path required for source: csv")
        _check(set(self.blocks_per_day) >= set(LEVELS) and set(self.block_hours) >= set(LEVELS),
               "outages.schedule needs blocks_per_day and block_hours for light, medium, heavy")


@dataclass
class MarkovConfig:
    start_prob: Dict[str, float] = field(default_factory=lambda: {"light": 0.01, "medium": 0.03, "heavy": 0.06})
    end_prob: float = 0.40
    heat_threshold_c: float = 32.0
    heat_factor_per_c: float = 0.15
    evening_hours: List[int] = field(default_factory=lambda: [18, 19, 20, 21, 22, 23])
    evening_factor: float = 1.5

    def __post_init__(self):
        _check(set(self.start_prob) >= set(LEVELS), "outages.markov.start_prob needs light, medium, heavy")
        for v in self.start_prob.values():
            _frac("outages.markov.start_prob", v)
        _frac("outages.markov.end_prob", self.end_prob)
        _hours("outages.markov.evening_hours", self.evening_hours)


@dataclass
class DurationConfig:
    distribution: str = "lognormal"
    params: Dict[str, float] = field(default_factory=lambda: {"s": 0.6, "scale": 2.0})
    max_hours: int = 12

    PARAMS = {"lognormal": {"s", "scale", "loc"}, "weibull": {"c", "scale", "loc"},
              "gamma": {"a", "scale", "loc"}, "exponential": {"scale", "loc"}}

    def __post_init__(self):
        _check(self.distribution in self.PARAMS, "duration.distribution must be lognormal|weibull|gamma|exponential")
        bad = set(self.params) - self.PARAMS[self.distribution]
        _check(not bad, f"duration.params {sorted(bad)} not valid for {self.distribution} "
                        f"(allowed: {sorted(self.PARAMS[self.distribution])})")
        _pos("duration.max_hours", self.max_hours, strict=True)


@dataclass
class SemiMarkovConfig:
    start_prob: Dict[str, float] = field(default_factory=lambda: {"light": 0.01, "medium": 0.03, "heavy": 0.06})
    hour_factor: List[float] = field(default_factory=lambda: [1.0] * 24)
    month_factor: List[float] = field(default_factory=lambda: [1.0] * 12)
    heat_threshold_c: float = 32.0
    heat_factor_per_c: float = 0.15
    duration: DurationConfig = field(default_factory=DurationConfig)

    def __post_init__(self):
        _check(set(self.start_prob) >= set(LEVELS), "outages.semi_markov.start_prob needs light, medium, heavy")
        _check(len(self.hour_factor) == 24, "outages.semi_markov.hour_factor needs 24 values")
        _check(len(self.month_factor) == 12, "outages.semi_markov.month_factor needs 12 values")


@dataclass
class ReplayConfig:
    csv_path: Optional[str] = None
    off_threshold: float = 0.5

    def __post_init__(self):
        _frac("outages.replay.off_threshold", self.off_threshold)


@dataclass
class OutageConfig:
    level: str = "random"
    model: str = "semi_markov"
    intensity: float = 1.0
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    markov: MarkovConfig = field(default_factory=MarkovConfig)
    semi_markov: SemiMarkovConfig = field(default_factory=SemiMarkovConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)

    def __post_init__(self):
        _check(self.level in LEVELS + ["random"], "outages.level must be light|medium|heavy|random")
        _check(self.model in ("markov", "semi_markov", "replay"), "outages.model must be markov|semi_markov|replay")
        _check(self.model != "replay" or self.replay.csv_path, "outages.replay.csv_path required for model: replay")
        _pos("outages.intensity", self.intensity)


@dataclass
class TariffPeriod:
    name: str
    hours: List[int]
    rate: float
    months: Any = "all"

    def __post_init__(self):
        _hours(f"tariff period {self.name}", self.hours)
        if self.months == "all":
            self.months = list(range(1, 13))
        _check(all(1 <= m <= 12 for m in self.months), f"tariff period {self.name}: months must be 1-12")
        _pos(f"tariff period {self.name}.rate", self.rate)


@dataclass
class ImportTariff:
    default_rate: float
    periods: List[TariffPeriod] = field(default_factory=list)


@dataclass
class TaxConfig:
    name: str
    percent: float
    applies_to: List[str] = field(default_factory=lambda: ["energy"])

    def __post_init__(self):
        _pos(f"tax {self.name}.percent", self.percent)
        _check(set(self.applies_to) <= {"energy", "fixed"}, f"tax {self.name}: applies_to must be energy|fixed")


@dataclass
class ExportTariff:
    scheme: str = "net_billing"
    rate: float = 11.0

    def __post_init__(self):
        _check(self.scheme in ("net_billing", "net_metering"), "export.scheme must be net_billing|net_metering")
        _pos("export.rate", self.rate)


@dataclass
class TariffConfig:
    name: str
    verified: Any
    import_: ImportTariff
    fixed_monthly_rs: float = 0.0
    taxes: List[TaxConfig] = field(default_factory=list)
    export: ExportTariff = field(default_factory=ExportTariff)

    def __post_init__(self):
        _pos("tariff.fixed_monthly_rs", self.fixed_monthly_rs)


@dataclass
class RewardConfig:
    penalty_unserved_essential: float = 200.0
    penalty_unserved_other: float = 30.0
    penalty_unmet_cooling: float = 0.0
    penalty_comfort_degree_hour: float = 10.0
    penalty_flex_missed: float = 60.0
    wear: str = "degradation"
    flat_wear_rs_per_kwh: float = 7.0
    include_calendar_wear: bool = True
    scale: float = 0.01

    def __post_init__(self):
        _check(self.wear in ("degradation", "flat"), "reward.wear must be degradation|flat")
        _pos("reward.scale", self.scale, strict=True)


@dataclass
class RLConfig:
    action_space: str = "reserve"
    observation: str = "extended"
    reserve_levels: List[float] = field(default_factory=lambda: [0.10, 0.30, 0.50, 0.70])

    def __post_init__(self):
        _check(self.action_space in ("reserve", "legacy"), "rl.action_space must be reserve|legacy")
        _check(self.observation in ("extended", "legacy"), "rl.observation must be extended|legacy")
        for r in self.reserve_levels:
            _frac("rl.reserve_levels", r)


@dataclass
class SimConfig:
    site: SiteConfig = field(default_factory=SiteConfig)
    data: DataConfig = field(default_factory=DataConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    rng: RngConfig = field(default_factory=RngConfig)
    pv: PVConfig = field(default_factory=PVConfig)
    inverter: InverterConfig = field(default_factory=InverterConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    outages: OutageConfig = field(default_factory=OutageConfig)
    tariff: Any = "residential_placeholder"
    reward: RewardConfig = field(default_factory=RewardConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    def __post_init__(self):
        if isinstance(self.tariff, str):
            raw = load_yaml(os.path.join(CONFIG_DIR, "tariffs", f"{self.tariff}.yaml"))
            self.tariff = build(TariffConfig, _tariff_keys(raw), "tariff")
        elif isinstance(self.tariff, dict):
            self.tariff = build(TariffConfig, _tariff_keys(self.tariff), "tariff")

    def to_dict(self):
        """Fully resolved config as plain data (profile and tariff inlined). load_config() accepts it."""
        d = _plain(dataclasses.asdict(self))
        d["tariff"]["import"] = d["tariff"].pop("import_")
        return d

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"_resolved": True, **self.to_dict()}, f, sort_keys=False, allow_unicode=True)

    def copy(self, overrides=None):
        return build(SimConfig, merge(self.to_dict(), overrides or {}), "config")


# ----------------------------------------------------------------------------- loading
def load_yaml(path):
    if not os.path.exists(path):
        raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_config_path(name):
    if name is None:
        return os.path.join(CONFIG_DIR, "default.yaml")
    if os.path.exists(name):
        return name
    p = os.path.join(CONFIG_DIR, name if name.endswith(".yaml") else f"{name}.yaml")
    if os.path.exists(p):
        return p
    raise ConfigError(f"no config '{name}' (looked for {name} and {p})")


def available_scenarios():
    return sorted(f[:-5] for f in os.listdir(CONFIG_DIR) if f.endswith(".yaml"))


def load_config(name="default", overrides=None) -> SimConfig:
    """Load configs/default.yaml, merge a scenario preset (name or path) and overrides on top."""
    path = resolve_config_path(name)
    raw = load_yaml(path)
    if raw.pop("_resolved", False):
        base = raw                                  # a saved, fully resolved config
    else:
        base = load_yaml(os.path.join(CONFIG_DIR, "default.yaml"))
        if os.path.abspath(path) != os.path.abspath(os.path.join(CONFIG_DIR, "default.yaml")):
            base = merge(base, raw)
    if overrides:
        base = merge(base, overrides)
    return build(SimConfig, base, "config")


REPLACE_KEYS = {"params"}     # dicts that an override replaces instead of merging into


def merge(base, over):
    """Deep-merge dicts; lists and scalars in `over` replace those in `base`."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in REPLACE_KEYS:
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def merge_profile(base, over):
    """Merge a profile override: appliances are matched by name (new names are added)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k == "appliances":
            by_name = {a["name"]: a for a in out.get("appliances", [])}
            for a in v:
                if a["name"] in by_name:
                    by_name[a["name"]].update(copy.deepcopy(a))
                else:
                    out.setdefault("appliances", []).append(copy.deepcopy(a))
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def profile_to_dict(p):
    return _plain(dataclasses.asdict(p))


def _tariff_keys(raw):
    raw = copy.deepcopy(raw)
    if "import" in raw:
        raw["import_"] = raw.pop("import")
    return raw


def _date(x):
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x))


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, dt.date):
        return x.isoformat()
    return x


def build(cls, data, where):
    """Build dataclass `cls` from a dict, recursively; unknown keys raise ConfigError."""
    if dataclasses.is_dataclass(data):
        return data
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(names)}")
    kwargs = {}
    for k, v in data.items():
        kwargs[k] = _coerce(hints[k], v, f"{where}.{k}")
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(f"{where}: {e}") from None


def _coerce(tp, v, where):
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union and type(None) in args:
        if v is None:
            return None
        tp = [a for a in args if a is not type(None)][0]
        origin, args = typing.get_origin(tp), typing.get_args(tp)
    if dataclasses.is_dataclass(tp):
        return build(tp, v, where)
    if origin in (list, List) and args and dataclasses.is_dataclass(args[0]):
        if not isinstance(v, list):
            raise ConfigError(f"{where}: expected a list")
        return [build(args[0], x, f"{where}[{i}]") for i, x in enumerate(v)]
    if tp is float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {v!r}")
        return float(v)
    if tp is int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ConfigError(f"{where}: expected an integer, got {v!r}")
        return v
    if tp is bool:
        if not isinstance(v, bool):
            raise ConfigError(f"{where}: expected true/false, got {v!r}")
        return v
    if tp is str:
        if not isinstance(v, str):
            raise ConfigError(f"{where}: expected a string, got {v!r}")
        return v
    if origin in (list, List):
        if not isinstance(v, list):
            raise ConfigError(f"{where}: expected a list, got {v!r}")
        if args and args[0] is float:
            return [_coerce(float, x, where) for x in v]
        return list(v)
    if origin in (dict, Dict):
        if not isinstance(v, dict):
            raise ConfigError(f"{where}: expected a mapping, got {v!r}")
        return dict(v)
    return v

"""Tariffs (time-of-use, fixed charges, taxes, export schemes) and config validation."""
import pytest

from solar_rl import SolarHomeEnv, load_config
from solar_rl.config import ConfigError, build, TariffConfig, _tariff_keys
from solar_rl.tariff import Tariff
from tests.conftest import synth_overrides


def make(raw):
    return Tariff(build(TariffConfig, _tariff_keys(raw), "tariff"))


BASE = {"name": "t", "verified": True,
        "import": {"default_rate": 40.0, "periods": [{"name": "peak", "hours": [18, 19, 20], "rate": 60.0},
                                                     {"name": "summer_day", "hours": [12, 13], "months": [6, 7],
                                                      "rate": 50.0}]},
        "fixed_monthly_rs": 744.0,
        "taxes": [{"name": "gst", "percent": 18.0, "applies_to": ["energy", "fixed"]},
                  {"name": "duty", "percent": 2.0, "applies_to": ["energy"]}],
        "export": {"scheme": "net_billing", "rate": 10.0}}


def test_time_of_use_rates():
    t = make(BASE)
    assert t.import_rate(1, 10) == 40.0
    assert t.import_rate(1, 19) == 60.0
    assert t.import_rate(6, 12) == 50.0
    assert t.import_rate(1, 12) == 40.0


def test_taxes_and_fixed_charge():
    t = make(BASE)
    imp, exp, fixed, rate = t.hour_cost(2.0, 1.0, 1, 19, year=2023)
    assert rate == 60.0
    assert imp == pytest.approx(2.0 * 60.0 * 1.20)            # GST 18 % + duty 2 % on energy
    assert exp == pytest.approx(10.0)                         # exports are not taxed
    assert fixed == pytest.approx(744.0 * 1.18 / (31 * 24))  # GST only on the fixed charge, spread over January
    total = sum(t.hour_cost(0, 0, 2, h % 24, year=2023)[2] for h in range(28 * 24))
    assert total == pytest.approx(744.0 * 1.18)


def test_presets_and_export_scenario():
    nb = load_config("default").tariff
    nm = load_config("legacy_net_metering").tariff
    assert nb.export.scheme == "net_billing" and nm.export.scheme == "net_metering"
    assert nm.export.rate > nb.export.rate
    assert nb.verified is False                                # placeholders until real values are set


def test_placeholder_tariff_warns():
    with pytest.warns(UserWarning, match="UNVERIFIED"):
        Tariff(load_config("default").tariff)


def test_export_rate_changes_cost_not_physics(cfg_factory):
    stats = {}
    for rate in (11.0, 26.0):
        env = SolarHomeEnv(cfg_factory(extra={"tariff": "legacy_net_metering_placeholder"})
                           .copy({"tariff": {"export": {"rate": rate}}}))
        env.reset(seed=0, options={"level": "light"})
        while True:
            *_, trunc, info = env.step(0)
            if trunc:
                break
        stats[rate] = info["episode_stats"]
    assert stats[11.0]["export_kwh"] == pytest.approx(stats[26.0]["export_kwh"])
    assert stats[26.0]["export_credit"] == pytest.approx(stats[11.0]["export_credit"] * 26 / 11)


# ---------------------------------------------------------------- config validation
@pytest.mark.parametrize("over,msg", [
    ({"pv": {"kwp": -1}}, "pv.kwp"),
    ({"pv": {"tilt": 120}}, "tilt"),
    ({"battery": {"soc_min": 0.9, "soc_max": 0.5}}, "soc_min"),
    ({"battery": {"eta_charge": 1.5}}, "eta_charge"),
    ({"outages": {"level": "extreme"}}, "outages.level"),
    ({"pv": {"kwpp": 5}}, "unknown key"),
    ({"pv": {"kwp": "five"}}, "expected a number"),
    ({"demand": {"profile": "no_such_profile"}}, "not found"),
    ({"tariff": "no_such_tariff"}, "not found"),
    ({"outages": {"model": "replay"}}, "csv_path"),
    ({"rl": {"action_space": "continuous"}}, "rl.action_space"),
])
def test_invalid_configs_rejected(over, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config("default", over)


def test_saved_config_roundtrip(tmp_path):
    cfg = load_config("shop", {"pv": {"kwp": 7.5}})
    p = tmp_path / "cfg.yaml"
    cfg.save(p)
    again = load_config(str(p))
    assert again.to_dict() == cfg.to_dict()

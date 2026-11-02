"""Solar model: annual kWh/kWp against a pvlib ModelChain reference, soiling behaviour,
and the original simple model kept unchanged."""
import numpy as np
import pandas as pd
import pvlib
import pytest

from solar_rl import load_config
from solar_rl.sim.pv import pv_power, pv_power_simple, pvlib_components, soiling_series


def reference_annual_kwh_per_kwp(weather, cfg):
    """Independent reference: pvlib ModelChain with PVWatts components and the same settings."""
    p, site = cfg.pv, cfg.site
    times = (weather.index + pd.Timedelta(minutes=30)).tz_localize(site.timezone)
    loc = pvlib.location.Location(site.latitude, site.longitude, tz=site.timezone)
    system = pvlib.pvsystem.PVSystem(
        surface_tilt=p.tilt, surface_azimuth=p.azimuth, albedo=p.albedo,
        module_parameters={"pdc0": p.kwp * 1000, "gamma_pdc": p.temp_coeff, "b": p.iam_b0},
        inverter_parameters={"pdc0": cfg.inverter.rated_kw * 1000 / p.inverter_eff, "eta_inv_nom": p.inverter_eff,
                             "eta_inv_ref": p.inverter_eff_ref},
        temperature_model_parameters={"u0": p.faiman_u0, "u1": p.faiman_u1},
        losses_parameters=dict(soiling=0, shading=0, snow=0, mismatch=2, wiring=2, connections=0.5, lid=1.5,
                               nameplate_rating=1, age=0, availability=0))
    mc = pvlib.modelchain.ModelChain(system, loc, transposition_model=p.transposition, aoi_model="ashrae",
                                     spectral_model="no_loss", temperature_model="faiman", losses_model="pvwatts",
                                     dc_model="pvwatts", ac_model="pvwatts")
    w = pd.DataFrame({"ghi": weather["ghi"].to_numpy(), "dni": weather["dni"].to_numpy(),
                      "dhi": weather["dhi"].to_numpy(), "temp_air": weather["temp_c"].to_numpy(),
                      "wind_speed": weather["wind_ms"].to_numpy()}, index=times)
    mc.run_model(w)
    ac = mc.results.ac.clip(lower=0).fillna(0)
    return ac.sum() / 1000 / p.kwp


@pytest.mark.parametrize("tilt,azimuth", [(20, 180), (30, 180), (10, 135)])
def test_annual_yield_matches_pvlib_reference(weather_2021, tilt, azimuth):
    cfg = load_config("default", {"pv": {"soiling_model": "none", "tilt": tilt, "azimuth": azimuth}})
    ours = pv_power(weather_2021, cfg).sum() / cfg.pv.kwp
    ref = reference_annual_kwh_per_kwp(weather_2021, cfg)
    assert ours == pytest.approx(ref, rel=0.01), (ours, ref)
    assert 1300 < ours < 2000      # plausible for Karachi


def test_real_data_yield_plausible():
    cfg = load_config("default", {"data": {"years": [2023], "allow_synthetic": False}})
    try:
        from solar_rl.sim.weather import load_years
        w = load_years([2023], cfg)
    except FileNotFoundError:
        pytest.skip("real 2023 weather not downloaded")
    if "dni" not in w or w.attrs["source"] != "nasa_power":
        pytest.skip("old-format weather file")
    ours = pv_power(w, cfg.copy({"pv": {"soiling_model": "none"}})).sum() / cfg.pv.kwp
    assert ours == pytest.approx(reference_annual_kwh_per_kwp(w, cfg), rel=0.01)


def test_inverter_clipping(weather_2021):
    cfg = load_config("default", {"pv": {"soiling_model": "none", "inverter_ac_kw": 3.0}})
    comp = pvlib_components(weather_2021, cfg)
    assert comp["pac_kw"].max() <= 3.0 + 1e-9
    assert comp["clipped_kw"].sum() > 0
    big = pvlib_components(weather_2021, load_config("default", {"pv": {"soiling_model": "none"}}))
    assert big["pac_kw"].sum() > comp["pac_kw"].sum()


def test_inverter_efficiency_lowers_output(weather_2021):
    a = pv_power(weather_2021, load_config("default", {"pv": {"inverter_eff": 0.96}})).sum()
    b = pv_power(weather_2021, load_config("default", {"pv": {"inverter_eff": 0.90}})).sum()
    assert b < a


def test_soiling_builds_up_and_resets():
    idx = pd.date_range("2023-01-01", periods=24 * 40, freq="h")
    rain = pd.Series(0.0, index=idx)
    rain.loc["2023-01-21 14:00"] = 20.0           # one heavy shower on day 21
    w = pd.DataFrame({"precip_mm": rain})
    cfg = load_config("default", {"pv": {"cleaning_interval_days": None, "soiling_rate_per_day": 0.01,
                                         "rain_grace_days": 0}})
    s = soiling_series(w, cfg.pv)
    daily = s.resample("D").last()
    assert daily.iloc[10] == pytest.approx(0.10, abs=0.011)     # 1 %/day build-up
    assert daily.loc["2023-01-21"] < 0.01                        # washed by rain
    assert daily.iloc[-1] > daily.loc["2023-01-22"]              # builds up again
    cleaned = soiling_series(w, cfg.copy({"pv": {"cleaning_interval_days": 7}}).pv)
    assert cleaned.max() <= 0.075                                # manual washing every 7 days
    assert soiling_series(w, cfg.copy({"pv": {"soiling_model": "none"}}).pv).max() == 0


def test_simple_model_unchanged(weather_2021):
    cfg = load_config("legacy_v1")
    ghi, temp = weather_2021["ghi"].to_numpy(), weather_2021["temp_c"].to_numpy()
    p = pv_power_simple(ghi, temp, cfg.pv)
    t_cell = temp + ghi / 800.0 * (45.0 - 20.0)
    expected = np.clip(5.0 * ghi / 1000.0 * (1 - 0.004 * (t_cell - 25.0)) * 0.95 * 0.90, 0, 5.0)
    np.testing.assert_allclose(p, expected, rtol=1e-12)
    np.testing.assert_allclose(pv_power(weather_2021, cfg), expected, rtol=1e-12)


def test_wind_cools_panels(weather_2021):
    cfg = load_config("default", {"pv": {"soiling_model": "none"}})
    calm = weather_2021.assign(wind_ms=0.5)
    windy = weather_2021.assign(wind_ms=8.0)
    assert pvlib_components(windy, cfg)["t_cell"].max() < pvlib_components(calm, cfg)["t_cell"].max()
    assert pv_power(windy, cfg).sum() > pv_power(calm, cfg).sum()

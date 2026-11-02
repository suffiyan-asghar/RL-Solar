"""PV output (AC kWh per hour) from hourly weather.

Two models (pv.model in the config):

simple  The original model: horizontal panels (POA = GHI), NOCT cell temperature, constant
        soiling and system derate, clipped at kWp.
pvlib   Plane-of-array irradiance for a tilted array (Hay-Davies by default) from GHI/DNI/DHI,
        ASHRAE incidence-angle loss on the beam, daily soiling build-up (Kimber model,
        cleaned by rain or manual washing), Faiman cell temperature with wind, PVWatts DC
        model with DC losses, PVWatts inverter with efficiency curve and AC clipping.

The solar position is computed at the middle of each hour because the weather values are
hour averages labelled at the start of the hour.
"""
import numpy as np
import pandas as pd


def pv_power_simple(ghi, temp_c, pv):
    """Original model. AC power (kW) from GHI (W/m2) and air temperature (C)."""
    ghi = np.asarray(ghi, dtype=float)
    t_cell = temp_c + ghi / 800.0 * (pv.noct - 20.0)
    temp_factor = 1.0 + pv.temp_coeff * (t_cell - 25.0)
    p = pv.kwp * ghi / 1000.0 * temp_factor * (1 - pv.soiling) * pv.system_derate
    return np.clip(p, 0.0, pv.kwp)


def soiling_series(weather, pv):
    """Fraction of energy lost to dust, per hour."""
    if pv.soiling_model == "none":
        return pd.Series(0.0, index=weather.index)
    if pv.soiling_model == "constant":
        return pd.Series(pv.soiling, index=weather.index)
    from pvlib.soiling import kimber

    out = []
    # run each contiguous year separately so gaps between years do not count as dry days
    for _, w in weather.groupby(weather.index.year):
        washes = None
        if pv.cleaning_interval_days:
            days = pd.date_range(w.index[0].normalize(), w.index[-1].normalize(), freq=f"{pv.cleaning_interval_days}D")
            washes = [d.date() for d in days[1:]]
        s = kimber(w["precip_mm"], cleaning_threshold=pv.rain_clean_mm, soiling_loss_rate=pv.soiling_rate_per_day,
                   grace_period=pv.rain_grace_days, max_soiling=pv.soiling_max, manual_wash_dates=washes,
                   initial_soiling=0.0, rain_accum_period=24)
        out.append(s)
    return pd.concat(out).reindex(weather.index).fillna(0.0)


def pvlib_components(weather, cfg):
    """Intermediate pvlib results as a DataFrame (useful for validation plots)."""
    import pvlib

    pv, site = cfg.pv, cfg.site
    idx = weather.index
    mid = (idx + pd.Timedelta(minutes=30)).tz_localize(site.timezone, nonexistent="shift_forward",
                                                        ambiguous="NaT")
    loc = pvlib.location.Location(site.latitude, site.longitude, tz=site.timezone)
    sp = loc.get_solarposition(mid)
    ghi = weather["ghi"].to_numpy()
    if "dni" in weather and "dhi" in weather:
        dni, dhi = weather["dni"].to_numpy(), weather["dhi"].to_numpy()
    else:   # decomposition for files that only have GHI
        e = pvlib.irradiance.erbs(weather["ghi"].set_axis(mid), sp["zenith"], mid)
        dni, dhi = e["dni"].to_numpy(), e["dhi"].to_numpy()
    dni_extra = pvlib.irradiance.get_extra_radiation(mid).to_numpy()
    airmass = loc.get_airmass(mid, solar_position=sp)["airmass_relative"].to_numpy()
    zen, azi = sp["apparent_zenith"].to_numpy(), sp["azimuth"].to_numpy()
    poa = pvlib.irradiance.get_total_irradiance(
        pv.tilt, pv.azimuth, zen, azi, dni, ghi, dhi, dni_extra=dni_extra, airmass=airmass,
        albedo=pv.albedo, model=pv.transposition)
    poa = pd.DataFrame({k: np.nan_to_num(np.asarray(v, float)) for k, v in poa.items()}, index=idx)
    aoi = pvlib.irradiance.aoi(pv.tilt, pv.azimuth, zen, azi)
    iam = pvlib.iam.ashrae(aoi, b=pv.iam_b0) if pv.iam_b0 is not None else np.ones(len(idx))
    iam = np.nan_to_num(np.asarray(iam, float))
    soil = soiling_series(weather, pv).to_numpy()
    effective = (poa["poa_direct"].to_numpy() * iam + poa["poa_diffuse"].to_numpy()) * (1 - soil)
    t_cell = pvlib.temperature.faiman(poa["poa_global"].to_numpy(), weather["temp_c"].to_numpy(),
                                      weather["wind_ms"].to_numpy(), u0=pv.faiman_u0, u1=pv.faiman_u1)
    pdc = pvlib.pvsystem.pvwatts_dc(effective, t_cell, pv.kwp * 1000.0, pv.temp_coeff) * (1 - pv.dc_losses)
    pdc = np.clip(np.asarray(pdc, float), 0, None)
    ac_kw = pv.inverter_ac_kw if pv.inverter_ac_kw is not None else cfg.inverter.rated_kw
    pac = pvlib.inverter.pvwatts(pdc, ac_kw * 1000.0 / pv.inverter_eff, eta_inv_nom=pv.inverter_eff,
                                 eta_inv_ref=pv.inverter_eff_ref)
    pac = np.clip(np.nan_to_num(np.asarray(pac, float)), 0, None)
    return pd.DataFrame({"poa_global": poa["poa_global"].to_numpy(), "poa_effective": effective,
                         "soiling": soil, "iam": iam, "t_cell": np.asarray(t_cell, float),
                         "pdc_kw": pdc / 1000.0, "pac_kw": pac / 1000.0,
                         "clipped_kw": np.clip(pdc * pv.inverter_eff / 1000.0 - pac / 1000.0, 0, None)},
                        index=idx)


def pv_power(weather, cfg):
    """AC output per hour (kW = kWh in a 1-hour step) for the configured model."""
    if cfg.pv.model == "simple":
        return pv_power_simple(weather["ghi"].to_numpy(), weather["temp_c"].to_numpy(), cfg.pv)
    return pvlib_components(weather, cfg)["pac_kw"].to_numpy()

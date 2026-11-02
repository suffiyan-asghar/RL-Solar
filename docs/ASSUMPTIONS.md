# Simulator assumptions

Every numeric setting in [`configs/default.yaml`](../configs/default.yaml), the demand profiles in
[`configs/demand/`](../configs/demand/) and the tariffs in [`configs/tariffs/`](../configs/tariffs/), with its
current value, unit and source. **Source** is empty where the value is a project assumption with
no source yet. When you change a value in a YAML file, update this table too.

Source abbreviations:
- **US5000**: Pylontech US5000 48 V LFP datasheet ([spec sheet](https://portal.segen.co.uk/reseller/docs/US5000%20spec.pdf)).
- **PVWatts**: Dobos, *PVWatts Version 5 Manual*, NREL/TP-6A20-62641 (2014).
- **Kimber**: Kimber et al., *The effect of soiling on large grid-connected PV systems in California*, IEEE PVSC (2006), as implemented in `pvlib.soiling.kimber`.
- **Faiman**: Faiman, *Assessing the outdoor operating temperature of PV modules*, Prog. Photovolt. 16 (2008), pvlib default coefficients.
- **Lahore soiling**: soiling study in Lahore, Renewable Energy 150 (2020) 456–468 ([link](https://ideas.repec.org/a/eee/renene/v150y2020icp456-468.html)). It reports about 0.8 %/day for a 30° panel from October to January.
- **Wang 2011**: Wang et al., *Cycle-life model for graphite-LiFePO4 cells*, J. Power Sources 196 (2011) 3942–3948.
- **Sandia**: SNL LFP calendar-ageing slides ([OSTI 2003268](https://www.osti.gov/servlets/purl/2003268)), about 8 % capacity loss over 3 years at 35 °C.
- **Retail listings**: Pakistani retailer spec pages for 6 kW hybrid inverters ([Itel 6 kW Pro](https://axipower.com.pk/products/itel-6kw-pro-hybrid-solar-inverter), [CoreTech 6 kW](https://kamalsolar.pk/products/coretech-nextgen-6kw-single-phase-hybrid-solar-inverter)). These are not manufacturer datasheets.
- **User decision**: values agreed with the project team on 2026-11-02 when no datasheet value existed.

## Site, data and episode

| Parameter | Value | Unit | Notes | Source |
|---|---|---|---|---|
| `site.latitude` / `longitude` | 24.86 / 67.01 | degrees | Karachi; also the NASA POWER download point | |
| `site.timezone` | Asia/Karachi | | PKT = UTC+5. Weather rows are hour-beginning local time | |
| `data.years` | 2020–2022 (train), 2023 (validation), 2024 (test) | | Train and validation years are set in `configs/train/*.yaml`, the test year in `evaluate.py` | |
| Weather variables | GHI, DNI, DHI, T2M, WS2M, PRECTOTCORR | W/m², °C, m/s, mm/h | Downloaded in UTC and converted to PKT | NASA POWER hourly API |
| `episode.hours` | 168 | h | One week | |
| `episode.initial_soc_range` | 0.2–0.9 | fraction | Random start SoC in training (evaluation uses 0.5) | |
| `episode.forecast_hours` | 6 | h | Forecast horizon in the observation | |
| `episode.solar_forecast_noise` | 0.15 | relative std | Std at lead k = 0.15 × (0.5 + k/6) | |
| `episode.temp_forecast_noise_c` | 1.5 | °C | Std at the 6-hour lead, linear in lead time | |

## Solar PV

| Parameter | Value | Unit | Notes | Source |
|---|---|---|---|---|
| `pv.model` | pvlib | | `simple` = original model | |
| `pv.kwp` | 5.0 | kWp | | |
| `pv.temp_coeff` | −0.004 | 1/°C | PVWatts γ_pdc | |
| `pv.tilt` | 20 | degrees | | Project requirement |
| `pv.azimuth` | 180 | degrees | Facing south | Project requirement |
| `pv.albedo` | 0.20 | fraction | Ground reflectance | pvlib default |
| `pv.transposition` | haydavies | | Hay & Davies sky-diffuse model | pvlib |
| `pv.iam_b0` | 0.05 | | ASHRAE incidence-angle modifier, applied to beam only | pvlib / ASHRAE default |
| `pv.faiman_u0` | 25.0 | W/m²K | Cell-temperature model | Faiman |
| `pv.faiman_u1` | 6.84 | W/m²K per m/s | Wind coefficient. Uses 2 m wind | Faiman |
| `pv.dc_losses` | 0.0681 | fraction | Mismatch 2 %, wiring 2 %, connections 0.5 %, LID 1.5 %, nameplate 1 % | PVWatts |
| `pv.inverter_ac_kw` | null (= `inverter.rated_kw`, 6.0) | kW | AC clipping limit | |
| `pv.inverter_eff` | 0.96 | fraction | Nominal efficiency | PVWatts |
| `pv.inverter_eff_ref` | 0.9637 | fraction | Reference efficiency | PVWatts |
| `pv.soiling_model` | kimber | | Daily build-up, reset by rain or washing | Kimber |
| `pv.soiling_rate_per_day` | 0.008 | fraction/day | No Karachi measurement found; Lahore value used as a proxy | Lahore soiling |
| `pv.soiling_max` | 0.30 | fraction | Kimber default | Kimber |
| `pv.rain_clean_mm` | 6.0 | mm/day | Kimber default | Kimber |
| `pv.rain_grace_days` | 14 | days | Kimber default | Kimber |
| `pv.cleaning_interval_days` | 14 | days | Manual panel washing | |
| `pv.noct` | 45 | °C | Simple model only | |
| `pv.soiling` | 0.05 | fraction | Simple model (and `soiling_model: constant`) | |
| `pv.system_derate` | 0.90 | fraction | Simple model only | |

With these settings the 2020–2024 NASA POWER data gives 1,560–1,730 kWh/kWp per year, or 1,630–1,790 with clean panels (`scripts/plot_monthly_energy.py`). This is within 1 % of a pvlib `ModelChain` reference (`tests/test_pv.py`).

## Hybrid inverter

| Parameter | Value | Unit | Notes | Source |
|---|---|---|---|---|
| `inverter.rated_kw` | 6.0 | kW | Common Pakistani hybrid size | Retail listings; User decision |
| `inverter.backup_max_kw` | 6.0 | kW | Max output to loads during an outage | User decision |
| `inverter.transfer_time_ms` | 10 | ms | "UPS mode". Loses essential load for 10 ms per outage start | Retail listings; User decision |
| `inverter.standby_w` | 20 | W | Self-consumption every hour | Retail listings (CoreTech) |

## Battery (LFP model)

| Parameter | Value | Unit | Notes | Source |
|---|---|---|---|---|
| `battery.capacity_kwh` | 10.0 | kWh | Nameplate | |
| `battery.soc_min_grid` | 0.10 | fraction | Lowest SoC with the grid on ("back to grid") | User decision |
| `battery.soc_cutoff_backup` | 0.05 | fraction | Lowest SoC in an outage (95 % usable) | US5000 (95 % DoD); User decision |
| `battery.soc_max_lfp` | 1.00 | fraction | | US5000 |
| `battery.max_charge_c` / `max_discharge_c` | 0.5 / 0.5 | C | Recommended continuous current (50 A per 100 Ah module) | US5000 |
| `battery.charge_taper_soc` | 0.90 | fraction | Start of the constant-voltage phase | |
| `battery.charge_taper_end_c` | 0.05 | C | Charge rate at 100 % | |
| `battery.discharge_taper_soc` | 0.15 | fraction | Discharge taper start | |
| `battery.discharge_taper_end_c` | 0.10 | C | Discharge rate at the cutoff | |
| `battery.loss_const` / `loss_lin` / `loss_quad` | 0.002 / 0.03 / 0.03 | | One-way loss = const·P_r + lin·P + quad·P²/P_r. Gives about 90 % round trip at 1.5 kW | Calibrated to Tesla Powerwall 2 datasheet RTE (90 %); curve shape assumed |
| `battery.cycle_life` | 6000 | cycles | To end of life | US5000 |
| `battery.cycle_life_dod` | 0.95 | fraction | DoD of the cycle-life test | US5000 |
| `battery.dod_exponent` | 1.0 | | Damage ∝ DoD^k. With k = 1, fade follows throughput | Wang 2011; User decision |
| `battery.eol_soh` | 0.70 | fraction | End-of-life state of health | |
| `battery.calendar_fade_per_year` | 0.02 | 1/year | At 25 °C. Derived from the 15-year design life to 70 % | US5000 (design life); User decision |
| `battery.calendar_ref_temp_c` | 25 | °C | | |
| `battery.calendar_q10` | 1.35 | ×/10 °C | 2.7 %/yr at 35 °C | Sandia; User decision |
| Battery temperature | outdoor air | °C | Used for calendar ageing | |
| `battery.replacement_cost_rs_per_kwh` | 39,900 | Rs/kWh | Rs 399,000 for 10 kWh, so a 95 % DoD cycle costs Rs 7.0 per kWh discharged (the original flat value) | User decision |
| `battery.initial_soh` | 1.0 | fraction | Reset every episode | |
| Simple model (`battery.model: simple`): `soc_min` 0.10, `soc_max` 0.95, `max_kw` 3.0, `eta_charge` 0.95, `eta_discharge` 0.95 | | | Original model | |

Calendar ageing adds about Rs 510 a week to every policy. It barely depends on actions (no SoC dependence yet), so the episode stats report it separately as `wear_calendar_cost`.

## Demand

All demand values are assumptions until calibrated with `scripts/compare_demand.py` against inverter-app logs.

**Household** ([`configs/demand/household.yaml`](../configs/demand/household.yaml)):

| Appliance | kW | Hours | Probability | Priority |
|---|---|---|---|---|
| fridge | 0.10 | all | 1.0 | essential |
| internet | 0.03 | all | 1.0 | essential |
| lights_evening | 0.30 | 18–00 | 1.0 | essential |
| lights_day | 0.10 | 06–17 | 0.8 | essential |
| lights_night | 0.05 | 01–05 | 1.0 | essential |
| fans | 0.40 × clip((T−24)/8) | all (×0.7 at 03–06) | 1.0 | essential |
| tv | 0.12 | 20–23 | 0.7 | other |
| microwave_kettle | 0.15 | 8, 13, 20 | 0.5 | other |
| charging_laptops | 0.08 | 09–22 | 0.6 | other |
| water_dispenser | 0.08 | all | 0.5 | other |
| flexible load | 1.5 kWh/day, ≤ 1.0 kW, due by 22:00 | | | flexible |

Weekend variant: TV also 14:00–17:00 and midnight, day lights always on, AC occupied from 10:00. Ramadan variant: suhoor lighting 0.30 kW at 01–05, kitchen at 03–04 and 18–19, TV until 01:00, AC occupancy shifted.

**Shop** ([`configs/demand/shop.yaml`](../configs/demand/shop.yaml)): chiller fridges 0.45 kW all day, lights 0.60 kW at 10–22, signboard 0.15 kW at 18–22, fans 0.30 kW, POS 0.06 kW, microwave 0.20 kW (p = 0.4). Shorter Sunday hours; open 13:00–01:00 in Ramadan. Flexible load 1.0 kWh/day.

| Parameter | Value | Unit | Notes | Source |
|---|---|---|---|---|
| `demand.noise` | 0.15 | lognormal σ | Multiplies hourly essential + other demand | |
| `demand.weekend_days` | [6] | day of week | Sunday | |
| `demand.ramadan_dates` | 2020–2025 | dates | Pakistan dates. Verify against Ruet-e-Hilal announcements | |
| `demand.measured_priority` | essential | | Logged load is treated as essential | |

## Room thermal model (AC)

| Parameter | Household | Shop | Unit | Notes | Source |
|---|---|---|---|---|---|
| `r_c_per_kw` | 6.0 | 3.5 | °C/kW | Thermal resistance to outside | |
| `c_kwh_per_c` | 1.0 | 1.5 | kWh/°C | Capacitance. Time constant R·C = 6 h / 5.3 h | |
| `solar_aperture_m2` | 0.5 | 1.0 | m² | Effective GHI gain area | |
| `internal_gain_kw` | 0.20 | 0.60 | kW | | |
| `ac_cooling_kw` | 5.3 | 7.0 | kW thermal | 1.5-ton and 2-ton AC (1 ton = 3.517 kW) | Definition of the refrigeration ton |
| `ac_cop` | 3.3 | 3.2 | | Constant (no outdoor-temperature dependence) | |
| `setpoint_c` | 26 | 26 | °C | Comfort threshold for degree-hours | |
| `precool_setpoint_c` | 23 | 23 | °C | | |
| `occupied_hours` | 13–17, 21–06 | 10–22 | h | AC runs on its normal setpoint only when occupied | |
| `initial_temp_c` | 26 | 28 | °C | | |
| Rule model (`household_v1`): 1.5 kW × clip((T−29)/5) at 13–17 and 21–03 | | | | Original model | |

## Load-shedding

| Parameter | Value | Unit | Notes | Source |
|---|---|---|---|---|
| `outages.model` | semi_markov | | `markov` = original model, `replay` = recorded log | |
| `outages.intensity` | 1.0 | × | Multiplier on unscheduled start probabilities | |
| `schedule.blocks_per_day` (light/medium/heavy) | 0 / 2 / 3 | blocks | Synthetic schedule, random daily offset | |
| `schedule.block_hours` | 0 / 2 / 3 | h | | |
| `semi_markov.start_prob` | 0.01 / 0.03 / 0.06 | 1/h | Base start probability | |
| `semi_markov.hour_factor` | 1.5 at 18–23, else 1 | × | | |
| `semi_markov.month_factor` | 1 | × | | |
| `semi_markov.heat_threshold_c` / `heat_factor_per_c` | 32 / 0.15 | °C, 1/°C | p × (1 + 0.15 × (T − 32)⁺) | |
| `semi_markov.duration` | lognormal s = 0.6, scale = 2.0, max 12 h | h | Mean ≈ 2.4 h, close to the old geometric mean of 2.5 h. **Replace with `scripts/fit_outages.py` output** | |
| `markov.start_prob` / `end_prob` | 0.01, 0.03, 0.06 / 0.40 | 1/h | Original model | |
| `replay.off_threshold` | 0.5 | fraction of hour | | |
| `configs/examples/schedule_example.csv` | | | **Example format only**, invented times | |

## Tariffs (placeholders: UNVERIFIED)

The real current values have not been provided yet. Until they are, every tariff file has `verified: false` and the code prints a warning.

| Parameter | Value | Unit | Notes | Source |
|---|---|---|---|---|
| Import off-peak (`residential_placeholder`) | 42 | Rs/kWh | Original project assumption | |
| Import peak 18:00–22:59 | 50 | Rs/kWh | Original project assumption | |
| Fixed monthly charge | 0 | Rs/month | **TODO** | |
| Taxes (GST, electricity duty) | 0 | % | **TODO** | |
| Net-billing export rate | 11 | Rs/kWh | | |
| Legacy net-metering export rate | 26 | Rs/kWh | Scenario `legacy_net_metering`. Exports credited at a flat rate (no unit netting) | |
| Commercial tariff (`commercial_placeholder`) | same as residential | | **TODO**: real commercial rates | |

## Reward

| Parameter | Value | Unit | Notes | Source |
|---|---|---|---|---|
| `penalty_unserved_essential` | 200 | Rs/kWh | Value of lost essential load | |
| `penalty_unserved_other` | 30 | Rs/kWh | | |
| `penalty_unmet_cooling` | 0 (legacy: 30) | Rs/kWh | Only for the rule-based AC model | |
| `penalty_comfort_degree_hour` | 10 | Rs/(°C·h) | Hour-mean room temperature above the setpoint, occupied hours | |
| `penalty_flex_missed` | 60 | Rs/kWh | | |
| `wear` | degradation | | Cycle + calendar damage × replacement cost (`flat` = Rs 7/kWh discharged) | |
| `scale` | 0.01 | reward/Rs | | |

## Scenario presets

| Preset | Changes from default |
|---|---|
| `small_home` | 3 kWp, 5 kWh, 4 kW inverter |
| `large_home` | 10 kWp, 20 kWh, 10 kW inverter, two AC rooms lumped (R/2, 2C, 2× aperture, gains and AC), bigger fridge, lights and fans, 3 kWh/day flexible |
| `shop` | Shop profile, commercial placeholder tariff |
| `heavy_loadshedding` | Outage level always heavy |
| `legacy_net_metering` | Export credited at Rs 26/kWh (placeholder) |
| `legacy_v1` | Reproduces the original simulator exactly (golden test) |

## Values still hard-coded in code

| Where | Value | Meaning |
|---|---|---|
| `sim/weather.py` | `KARACHI_MONTHLY_TEMP`, `KARACHI_MONTHLY_CLOUD`, `KARACHI_MONTHLY_RAIN_DAY`, AR(1) coefficients, sea-breeze wind | Synthetic weather only |
| `sim/pv.py` | Solar position at mid-hour | Weather values are hour averages |
| `sim/thermal.py` | Hour-mean temperature for comfort; AC reaches the setpoint by the end of the hour | |
| `sim/simulator.py` | Outage service order: standby → essential → AC → other → flexible | |
| `baselines.py` `naive_flex` | Flexible load from 08:00 | Baseline habit |
| `baselines.py` `HandTunedRule` | Reserves 0.6 / 0.3, risky hours 16:00–23:59, flexible load when solar ≥ 30 % of kWp | |
| `oracle.py` | Constant efficiencies at 30 % of rated power; re-plan when the SoC plan is off by > 3 % of capacity | LP approximations |
| `evaluation.py` | Evaluation start SoC 0.5; episode seeds 10000 × (seed + 1) + i | Fixed test episodes |
| `env.py` | Observation scaling constants (e.g. price / 60, temperature (T − 25) / 10) | |

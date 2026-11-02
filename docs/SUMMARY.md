# Project work summary

What has been done on the RL Solar + Battery Manager so far, in order, with the decisions made
along the way and what is still open. Details live in [SIMULATOR.md](SIMULATOR.md) (models and
equations), [ASSUMPTIONS.md](ASSUMPTIONS.md) (every value and its source) and the [README](../README.md)
(how to run things).

## 1. First run of the original project

- Set up a virtual environment (`.venv`, Python 3.14) and installed `requirements.txt`. The original sanity test passed.
- Downloaded real NASA POWER weather for Karachi, 2020–2024.
- Evaluated the five original baselines on 2024. `battery_first` was cheapest (Rs 169/week) but left 1.05 kWh/week of essential load unserved.
- Trained PPO (seed 0, 300k steps). It cost Rs 388/week with 0.72 kWh unserved: cheaper than `hand_tuned` but about twice the cost of `battery_first`. Its validation score had not settled.

## 2. Code review and first documentation

- Read the whole repository and wrote a summary of how the simulator, environment, baselines and scripts fit together.
- Added `*.log` to `.gitignore`. Git was already initialised (commit `a1cc63d initial work`).
- Wrote the first `docs/ASSUMPTIONS.md`, listing every value in the old `config.py`. It also flagged values that were hard-coded outside the config.

## 3. Rebuilt simulator

The simulator was rebuilt in one pass so that all later features fit the same design.

| Area | What was done |
|---|---|
| Structure | The physical simulator is in `solar_rl/sim/`, with no RL code. `env.py` is now a thin Gymnasium wrapper (observation, action mapping, reward). The simulator logs every hour to a DataFrame/CSV |
| Configuration | All settings are in YAML under `configs/`: a default, presets (`small_home`, `large_home`, `shop`, `heavy_loadshedding`, `legacy_net_metering`, `legacy_v1`), demand profiles, tariffs and training settings. Dataclasses validate every value and reject unknown keys |
| Randomness | One seed is split into independent random streams (episode set-up, outages, demand, forecasts). Actions never change outages or demand, so all policies face identical conditions |
| Weather | The downloader now also fetches direct and diffuse irradiance, wind and rain. It requests UTC and converts to Pakistan time |
| Solar | A pvlib model: 20° tilt facing south, panel temperature with wind, ASHRAE angle losses, daily dust build-up (Kimber model) washed by rain or every 14 days, inverter efficiency and clipping. The old simple model is kept as an option |
| Battery | A 48 V LFP model: power limits that taper near full and empty, power-dependent losses (about 90 % round trip), separate minimum charge with the grid on (10 %) and in outages (5 %), and cycle and calendar ageing. Ageing sets the wear cost. The old model is kept as an option |
| Inverter | 6 kW hybrid inverter, 6 kW backup limit, 10 ms transfer time, 20 W standby |
| Demand | Household and shop appliance profiles in YAML (power, hours, probability of use), with weekend and Ramadan variants. Real inverter-app logs can replace the profile |
| Air conditioning | A one-room thermal model (first-order RC). The AC cools toward 26 °C, or 23 °C when pre-cooling. Comfort is measured in degree-hours above the setpoint, and the room heats up during outages |
| Load-shedding | Scheduled blocks (synthetic or from a published schedule CSV), a semi-Markov model for unscheduled outages (lengths from a fitted distribution), and replay of a recorded outage log. `fit_outages.py` estimates the parameters from a log |
| Tariffs | Time-of-use import rates, fixed monthly charges, taxes, the net-billing export rate and legacy net metering as a scenario |
| RL | A new 16-action space: target battery reserve {10, 30, 50, 70 %} × flexible load × AC pre-cool. The old 8 actions remain an option. The observation grew to 36 values |
| Baselines | The old baselines, `battery_first` swept over reserves of 10–70 %, and a perfect-foresight LP oracle (scipy HiGHS) as an upper bound |
| Evaluation | Mean ± std across seeds; results by outage level, season and export rate; a cost vs. unserved-load scatter plot |
| Training | Settings in `configs/train/` (2M steps, n_steps 2048, gamma 0.995). `run_experiments.py` trains PPO and DQN on 5 seeds and then evaluates everything |
| Validation | `validate_sim.py` writes a report: monthly solar yield, demand profiles, outage durations, battery SoC behaviour and annual cost per baseline. When real logs are given, it compares them and reports the error |
| Dashboard | A Streamlit app (`app/dashboard.py`) that plays two policies side by side, with running totals and sliders for tariffs, export rate and outage intensity |

## 4. Decisions taken with the team

| Topic | Decision |
|---|---|
| Tariffs | Keep the old values (Rs 42 off-peak, Rs 50 peak, Rs 11 export, Rs 26 legacy) as **placeholders** marked unverified. Fixed charges and taxes stay at 0 until the real values are provided |
| Battery replacement cost | Rs 399,000 for 10 kWh. This keeps the old wear cost of Rs 7 per kWh discharged at datasheet cycling |
| Cycle ageing | Damage follows energy throughput (DoD exponent k = 1), based on Wang et al. (2011) |
| Inverter and ageing | 6 kW, 10 ms transfer, 20 W standby, 10 % / 5 % minimum charge, calendar ageing of 2 %/yr at 25 °C and 2.7 %/yr at 35 °C |

## 5. Problems found and fixed

- **Weather times were one hour off.** NASA POWER "LST" times are UTC+4 at Karachi, but they were used as Pakistan time (UTC+5). The old files are kept in `data/lst_v1/` so the original results can still be reproduced.
- **The battery could gain energy from nothing.** A zero-energy discharge below the minimum charge raised it back up to the minimum.
- **Preset outage settings could be mixed up.** Changing the outage-length distribution in a preset kept the default distribution's parameters.
- **The oracle could crash mid-episode.** Its linear program had no solution right after an outage, when the battery sat below its minimum charge.
- **Text encoding on Windows.** Config files are now read and written explicitly as UTF-8.

## 6. Verification

- 127 automated tests pass. They cover:
  - energy balance every hour, for every scenario and battery mode
  - battery charge and power limits
  - no grid import or export during outages
  - reproducibility from one seed
  - identical results however the same setup is configured
  - the solar model against pvlib's `ModelChain` (within 1 %)
  - battery, demand, thermal, outage, tariff, RL and dashboard behaviour
- **Golden test:** `configs/legacy_v1.yaml` reproduces 32 recorded episodes of the original code exactly (every reward, observation and episode statistic).
- **Short training check:** PPO and DQN, 30k steps, 2 seeds, followed by the full evaluation. At export Rs 11:

| Policy | Cost incl. wear (Rs/week) | Unserved essential (kWh/week) |
|---|---|---|
| oracle | 1,488 ± 42 | 0.28 |
| battery_first_r30 | 1,645 ± 27 | 0.99 |
| DQN | 1,461 ± 45 | 3.34 |
| PPO | 2,034 ± 40 | 0.49 |

These agents are deliberately undertrained; the run only checks that the pipeline works. The oracle has the best reward in every setting, as an upper bound should.

## 7. Things to keep in mind when reading results

- Results with the new default models are not comparable with the first run in section 1. Use `legacy_v1` to reproduce the old numbers.
- Calendar ageing adds about Rs 510 a week to every policy. It barely depends on actions and is reported separately.
- Policies that never pre-cool show about 16 °C·h of discomfort a week, because the AC only switches on when the room becomes occupied.

## 8. Still open

- [ ] Replace the placeholder tariffs with the current NEPRA / K-Electric values (import rates, fixed charges, GST and duty, buyback rate, commercial tariff)
- [ ] Calibrate demand profiles with real inverter-app logs (`compare_demand.py`)
- [ ] Record a real outage log, fit the outage model (`fit_outages.py`) and add a real published schedule
- [ ] Check the unsourced values listed in ASSUMPTIONS.md: thermal parameters, Karachi soiling rate, Ramadan dates, battery end-of-life point
- [ ] Tune PPO/DQN on the 2023 validation year and run the full experiments (E1–E6) with 2M-step training
- [ ] Write the report and record the demo with the dashboard
- [ ] Stop tracking `results_train_ppo.log` in git (`git rm --cached results_train_ppo.log`) and commit the work

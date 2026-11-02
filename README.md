# RL Solar + Battery Manager (Karachi)

CSE 482 Reinforcement Learning project. Every hour, an RL agent decides how a home (or shop) with rooftop solar, a battery and air conditioning should use solar, the battery and the grid. It has to cope with unpredictable load-shedding, net billing (Rs 11 export vs Rs 40+ import), battery wear and the comfort of an air-conditioned room.

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_weather.py --years 2020 2021 2022 2023 2024   # NASA POWER: GHI, DNI, DHI, temp, wind, rain
python -m pytest -q                                                    # ~125 tests, about 1 minute
```

If a weather file is missing, the simulator warns and uses synthetic Karachi weather instead. That is fine for development and tests, but **final results must use the real data**.

> Weather files are now downloaded in UTC and converted to Pakistan time. The first version of the downloader stored NASA POWER "LST" times, which are UTC+4 at Karachi's longitude and therefore one hour behind PKT. If you have old files, move them to `data/lst_v1/` and download again.

## 2. Project structure

```
configs/
  default.yaml            all simulator settings (every number is an assumption, see docs/ASSUMPTIONS.md)
  small_home.yaml ...     scenario presets merged on top of default.yaml:
                          small_home, large_home, shop, heavy_loadshedding, legacy_net_metering,
                          legacy_v1 (reproduces the original simulator exactly)
  demand/*.yaml           appliance profiles: household, shop (weekday/weekend/Ramadan variants), household_v1
  tariffs/*.yaml          time-of-use import rates, fixed charges, taxes, export rate (PLACEHOLDERS, see below)
  train/*.yaml            PPO / DQN hyperparameters
  examples/               example published-schedule CSV (format only)
solar_rl/
  config.py               YAML loader + dataclasses that validate every setting
  sim/                    the physical simulator, no RL code:
    simulator.py          Simulator: one Dispatch per hour -> energy flows; hourly log as a DataFrame/CSV
    weather.py pv.py      NASA POWER weather; pvlib PV model (POA, Faiman, soiling, inverter) or the simple model
    demand.py thermal.py  appliance profiles, measured logs; first-order RC room model for the AC
    battery.py            LFP battery (taper, power-dependent losses, cycle + calendar ageing) or the simple model
    outages.py            schedules (synthetic or CSV), semi-Markov / Markov outages, log replay, fitting
  tariff.py               electricity bill per hour
  env.py                  Gymnasium wrapper: observation, action mapping, reward
  baselines.py oracle.py  rule-based policies, battery_first reserve sweep, perfect-foresight LP oracle
  evaluation.py           fixed test episodes, mean ± std across seeds, scatter plot
  validation.py           simulated vs. real comparisons
scripts/                  see section 3
app/dashboard.py          Streamlit demo app (section 6)
docs/ASSUMPTIONS.md       every parameter with value, unit and source
docs/SIMULATOR.md         every model, its equations and limitations (for the report)
tests/                    energy balance, limits, reproducibility, golden test vs. the original, models
```

Loading a configuration from Python:

```python
from solar_rl import load_config, Simulator, Dispatch, SolarHomeEnv
cfg = load_config("shop", {"pv": {"kwp": 8.0}})      # preset + overrides; invalid values raise ConfigError
sim = Simulator(cfg); sim.reset(seed=0, level="heavy")
while not sim.done:
    sim.step(Dispatch(battery="reserve", reserve=0.3, run_flex=True, ac="normal"))
sim.save_log("week.csv")                              # every hour: flows, SoC, SoH, room temperature, ...
env = SolarHomeEnv(cfg)                               # the same simulator as a Gymnasium env
```

## 3. Quick start

```bash
python scripts/evaluate.py --seeds 0 1 --episodes 4            # all baselines + reserve sweep + oracle on 2024
python scripts/train.py --train-config configs/train/ppo.yaml --seeds 0 --timesteps 200000
python scripts/evaluate.py --models "ppo=models/ppo_seed{seed}/best_model.zip" --seeds 0
python scripts/plot_episode.py --policies ppo=models/ppo_seed0/best_model.zip oracle --level heavy \
       --date 2024-06-10 --out results/demo.png
python scripts/run_experiments.py                              # 5 seeds x PPO + DQN (2M steps), then full evaluation
python scripts/run_experiments.py --timesteps 30000 --seeds 0 1 --episodes 3 --models models_short   # quick check
```

| Script | What it does |
|---|---|
| `download_weather.py` | Fetches NASA POWER hourly data in UTC and converts it to PKT |
| `train.py` | Trains PPO or DQN from `configs/train/*.yaml`. Saves `best_model.zip`, `env_config.yaml` and `train_config.json` per seed |
| `run_experiments.py` | Trains 5 seeds of PPO and DQN, then runs `evaluate.py` with export rates 11 and 26 |
| `evaluate.py` | Evaluates baselines, the battery_first reserve sweep (10–70 %), the oracle and trained models. Reports mean ± std across seeds by outage level, season and export rate. Draws a cost vs. unserved-load scatter plot |
| `plot_episode.py` | Demo figure: one week, policies side by side (power, SoC, room temperature, actions) |
| `plot_monthly_energy.py` | Monthly kWh/kWp to compare with published yields (Global Solar Atlas, PVGIS) |
| `compare_demand.py` | Simulated vs. real inverter-app log (`timestamp,load_kw`) |
| `fit_outages.py` | Fits semi-Markov outage parameters from a log (`timestamp,grid_on`) |
| `validate_sim.py` | Validation report in `results/validation/`, with real-data error if logs are given |

## 4. How the simulator works

Full equations and limitations are in [docs/SIMULATOR.md](docs/SIMULATOR.md), and every number is in [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md).

| Part | Model |
|---|---|
| Time step | 1 hour. An episode is 1 week (168 steps) starting at midnight. One seed drives independent random streams (outages, demand, forecasts) |
| Weather | NASA POWER hourly GHI/DNI/DHI, temperature, wind, rain. Train 2020-22, validate 2023, test 2024 |
| Solar | pvlib: 20° tilt facing south, Hay-Davies POA, ASHRAE IAM, Faiman cell temperature with wind, PVWatts DC/inverter with clipping. Kimber soiling (0.8 %/day, washed by rain or every 14 days). About 1,600 kWh/kWp per year |
| Demand | YAML appliance profiles (power, hours, probability of use), weekday/weekend/Ramadan variants, lognormal noise. Flexible load of 1.5 kWh/day due by 22:00. Optional measured logs |
| AC | First-order RC room. The AC cools toward a 26 °C setpoint (23 °C when pre-cooling). Comfort is degree-hours above the setpoint. The room heats up during outages |
| Battery | 10 kWh 48 V LFP, 0.5C with taper near full and empty, power-dependent losses (about 90 % round trip), 10 % floor with the grid on and 5 % in outages. Cycle ageing by depth of discharge plus calendar ageing; wear cost comes from this model |
| Inverter | 6 kW hybrid, 6 kW backup limit, 10 ms transfer, 20 W standby |
| Load-shedding | Scheduled blocks (synthetic, or a published CSV) plus semi-Markov unscheduled outages (start probability by hour, month and heat; fitted duration distribution). Or replay of a recorded log |
| Tariff | Time-of-use import, fixed charges, taxes, net-billing export or legacy net metering (**placeholder values**) |

**Actions (default, 16):** target battery reserve {10, 30, 50, 70 %} × flexible load {delay, run} × AC {normal, pre-cool}. A dispatch controller then:
- discharges the battery above the reserve,
- charges it from solar first,
- charges it from the grid only while SoC is below the reserve.

The original 8 actions (battery mode × flexible load) are available as `rl.action_space: legacy`. During an outage the inverter automatically serves essential load, then AC, then other loads, then flexible load.

**Observation (36 values):** time features, SoC, solar, demand, expected AC demand, flexible energy left, grid status and outage history, outage level, price, next scheduled outage, temperatures, noisy 6-hour forecasts of solar and temperature, room temperature, occupancy, weekend and Ramadan flags, and battery state of health. The agent is **never told when unscheduled outages will happen**.

**Reward (Rs × 0.01):**

−(import cost − export credit) − battery wear − 200 × unserved essential kWh − 30 × unserved other kWh − 10 × degree-hours above the setpoint − 60 × flexible kWh missed

**Reproducing the original results:** `configs/legacy_v1.yaml` selects the original models (simple PV and battery, rule-based AC, Markov outages, one shared RNG, 8 actions, 30-value observation). `tests/test_golden.py` checks that it reproduces 32 recorded episodes of the original code (commit `a1cc63d`) exactly. The original weather files (LST labels) are in `data/lst_v1/`.

## 5. Experiments (from the proposal)

| ID | What | How |
|---|---|---|
| E1 | Main comparison | `run_experiments.py` (PPO + DQN × 5 seeds, all baselines, oracle) |
| E2 | Scenarios | `evaluate.py` splits by outage level and season. Use `--export-rates 11 26` for legacy net metering, or `--scenario shop` / `heavy_loadshedding` |
| E3 | System sizes | `--scenario small_home` / `large_home`, or override `pv.kwp`, `battery.capacity_kwh` |
| E4 | Robustness | `train.py --outage-level light`, then `evaluate.py --levels heavy`. Raise `episode.solar_forecast_noise` at test time. Replay real outage logs (`outages.model: replay`) |
| E5 | Reward ablation | Vary `reward.penalty_unserved_essential`, `penalty_comfort_degree_hour` and `battery.dod_exponent`; compare evening SoC and cycles |
| E6 | Algorithms | PPO vs DQN learning curves (`tensorboard --logdir models/tb`) |

Report every RL result as mean ± std over at least 5 seeds (`evaluate.py` does this automatically).

## 6. Dashboard (for the demo video)

```bash
streamlit run app/dashboard.py
```

It opens in the browser at http://localhost:8501.

1. **Sidebar:** choose a scenario, weather year, start date, load-shedding level, random seed and two policies (any baseline, the oracle, or trained models found in `models*/*/best_model.zip`).
2. **What if ...:** sliders for the off-peak and peak import rates, the export rate and the unscheduled-outage intensity. Press **Run** to re-simulate both policies on identical weather, demand and outages.
3. **Playback:** ▶ Play animates the week hour by hour (choose the speed), +24 h skips a day, and ⏭ End jumps to the end of the week.
4. **Each policy shows:**
   - running totals: energy cost, unserved load, battery wear, discomfort, grid import, SoC
   - charts of solar, demand and import; battery SoC; room and outdoor temperature; and the actions taken
   - grey bands mark hours when the grid is off
5. **Totals table:** the table at the bottom compares both policies.

For a recording, use a wide browser window, pick `heavy` load-shedding and a June date, and compare a trained agent with `battery_first_r30` or the `oracle`.

## 7. Still to do

- [ ] **Replace placeholder tariffs** in `configs/tariffs/` with the current NEPRA / K-Electric notice (import rates, fixed charges, GST and duty, buyback rate, commercial tariff), then set `verified: true`
- [ ] Calibrate demand profiles with real inverter-app logs (`compare_demand.py`)
- [ ] Record a real outage log and fit the outage model (`fit_outages.py`); add a real published schedule CSV
- [ ] Check the battery replacement cost, the calendar ageing rate and the end-of-life SoH against a quote or datasheet
- [ ] Check Ramadan dates and the Karachi soiling rate
- [ ] Tune PPO/DQN hyperparameters on the 2023 validation year
- [ ] Run E1–E6 with the full 2M-step training, analyse failure cases, write the report and record the demo

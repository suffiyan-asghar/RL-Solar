# RL Solar + Battery Manager (Karachi)

CSE 482 Reinforcement Learning project. An RL agent decides, every hour, how a home with
rooftop solar and a battery should use solar, the battery and the grid, under unpredictable
load-shedding, net billing (Rs 11 export vs Rs 40+ import) and battery wear.

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_weather.py --years 2020 2021 2022 2023 2024   # real NASA POWER data
python tests/test_env.py                                               # sanity checks
```

If the weather CSVs are missing, the simulator prints a warning and uses **synthetic**
Karachi weather. That is fine for development, but **final results must use the real data**.

## 2. Project structure

```
solar_rl/
  config.py       all assumptions (sizes, tariffs, outage rates, penalties) - edit here
  weather.py      loads data/karachi_<year>.csv, or generates synthetic weather
  components.py   PV model, demand model, load-shedding model, battery model
  env.py          the Gymnasium environment (state, actions, energy balance, reward)
  baselines.py    random, solar_first, battery_first, grid_first, hand_tuned
  evaluation.py   fixed test episodes so every policy faces identical conditions
scripts/
  download_weather.py   get real hourly irradiance + temperature (run on your laptop)
  train.py              train PPO / DQN with several seeds
  evaluate.py           compare policies on the test year, by outage level and season
  plot_episode.py       demo figure: one week, two policies side by side
tests/test_env.py       energy-conservation and battery-limit tests
```

## 3. Quick start

```bash
python scripts/evaluate.py                                    # baselines on 2024
python scripts/train.py --algo ppo --seeds 0 --timesteps 300000   # ~3 min on a laptop
python scripts/evaluate.py --models ppo=models/ppo_seed0/best_model.zip
python scripts/plot_episode.py --policies ppo=models/ppo_seed0/best_model.zip battery_first \
       --level heavy --date 2024-06-10 --out results/demo.png
```

## 4. How the simulator works

| Part | Model |
|---|---|
| Time step | 1 hour; episode = 1 week (168 steps), starts at midnight on a random date |
| Weather | Hourly GHI and temperature (NASA POWER), split by year: train 2020-22, validate 2023, test 2024 |
| Solar | 5 kWp, temperature derating (-0.4%/C), 5% dust loss, 90% system losses |
| Demand | Essential (lights, fans, fridge, internet), AC (rises with temperature), flexible 1.5 kWh/day due by 22:00; random noise |
| Load-shedding | Scheduled blocks (light 0, medium 2x2h, heavy 3x3h per day) + random outages (Markov chain, more likely in evenings and heat). Level is randomised each episode during training |
| Battery | 10 kWh, 10-95% SoC, 3 kW, 95% charge/discharge efficiency, wear Rs 7 per kWh discharged |
| Tariff | Import Rs 42 off-peak / Rs 50 peak (18-22h); export Rs 11 |

**Actions (8):** battery mode {grid_charge, solar_charge, hold, discharge} x flexible load {delay, run}.
During an outage the inverter automatically uses solar + battery for essential, then AC, then flexible load.

**Observation (30 values):** hour and day-of-year (sin/cos), SoC, solar now, demand now, flexible energy left,
hours to deadline, grid on/off, hours since last outage, outage level, price, hours to next scheduled
outage, temperature, episode progress, and a noisy 6-hour forecast of solar and temperature.
The agent is **never told when random outages will happen**.

**Reward (Rs x 0.01):** -(import cost - export credit) - battery wear - 200 x unserved essential kWh
- 30 x unmet AC kWh - 60 x flexible kWh missed.

All numbers live in `solar_rl/config.py`. They are assumptions: check them against real tariff
notices, battery datasheets and inverter logs, and record what you changed in the report.

## 5. Experiments (from the proposal)

| ID | What | How |
|---|---|---|
| E1 | Main comparison | `train.py --algo ppo --seeds 0 1 2 3 4` and `--algo dqn`, then `evaluate.py --models ...` |
| E2 | Scenarios | `evaluate.py` already splits by outage level and season; add `--export-rate 26` for legacy net metering |
| E3 | System sizes | change `pv_kwp`, `batt_kwh` in config, retrain and evaluate |
| E4 | Robustness | train with `--outage-level light`, test with `--levels heavy`; raise `forecast_noise` at test time |
| E5 | Reward ablation | vary `penalty_unserved_essential` and `wear_cost_per_kwh`, compare evening SoC and cycles |
| E6 | Algorithms | PPO vs DQN learning curves (`tensorboard --logdir models/tb`) |

Report every RL result as mean +/- std over at least 5 seeds.

## 6. Still to do

- [ ] Download real weather and re-run everything
- [ ] Calibrate the demand profile with real inverter-app logs (friends or family)
- [ ] Check tariffs, buyback rate and battery cost; update `config.py`
- [ ] Add the oracle baseline (linear program with perfect foresight, e.g. with `scipy.optimize.linprog` or PuLP)
- [ ] Tune PPO/DQN hyperparameters on the 2023 validation year
- [ ] Run E1-E6, analyse failure cases, write the report and record the demo

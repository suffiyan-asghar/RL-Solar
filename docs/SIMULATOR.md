# Simulator description

This section describes every model in the simulator, its equations and its limitations, and is
written to be pasted into the project report. Parameter values are listed in
[ASSUMPTIONS.md](ASSUMPTIONS.md); the code is in `solar_rl/sim/` (physics), `solar_rl/tariff.py` and
`solar_rl/env.py` (reward).

## Overview

The simulator represents one home (or small shop) in Karachi. It has a rooftop PV array, a
48 V LFP battery behind a hybrid inverter, a grid connection subject to load-shedding, and
household demand that includes one air-conditioned room.

- **Time step:** one hour (Δt = 1 h), so power in kW and energy in kWh are numerically equal.
- **Episode:** one week (168 steps) starting at midnight.
- **Hourly inputs:** each hour the controller (RL agent, rule-based baseline or LP oracle) chooses a dispatch decision with three parts:
  - a battery mode,
  - whether to run the flexible load,
  - an AC mode (normal setpoint, pre-cool, or off).
- **Hourly outputs:** the simulator returns the hour's energy flows (solar, load served and unserved, battery charge and discharge, grid import and export, curtailment, inverter standby), plus room temperature and battery ageing. Every hour is logged and can be saved as CSV.

The physical simulator contains no reward or price logic. The Gymnasium environment wraps it and adds the observation, the mapping from actions to dispatch decisions, and the reward.

All randomness comes from one seed. The seed is split with `numpy.random.SeedSequence` into independent streams for episode set-up, outages, demand and forecasts. Outages and demand therefore do not depend on the actions taken, so every policy faces identical conditions. This also lets the oracle read the future from a copy of the simulator.

## Weather

Hourly data from the NASA POWER API at 24.86° N, 67.01° E, 2020–2024:

- global, direct-normal and diffuse irradiance (GHI, DNI, DHI)
- air temperature at 2 m
- wind speed at 2 m
- precipitation

The data are requested in UTC and converted to Pakistan Standard Time (UTC+5). Values are hour averages, labelled at the start of the hour, so solar geometry is computed at the middle of each hour. Years are split into training (2020–22), validation (2023) and test (2024) sets.

*Limitations:*
- NASA POWER is a satellite and reanalysis product with about 0.5° resolution. It smooths local clouds, haze and the sea breeze.
- Forecasts given to the agent are the true future values with Gaussian noise. They are not a real forecast model.

## Solar PV

**Plane-of-array irradiance** uses the Hay–Davies transposition model with ground albedo ρ:

$$G_{poa} = G_{b}\cos\theta + G_{d,sky}(\text{Hay–Davies}) + \rho\,\text{GHI}\,\frac{1-\cos\beta}{2}$$

Here θ is the angle of incidence on a panel with tilt β = 20° facing south.

**Incidence-angle losses** on the beam component use the ASHRAE model, IAM = 1 − b₀(1/cos θ − 1).

**Soiling** follows the Kimber model:
- The soiled fraction s grows by a fixed rate each dry day, up to a maximum.
- It resets to zero when daily rain exceeds a threshold. No build-up happens for a grace period after rain.
- It also resets on manual washing every N days.

The effective irradiance is

$$G_{eff} = (G_{b,poa}\,\text{IAM} + G_{d,poa})(1-s)$$

**Cell temperature** uses the Faiman model with wind speed v:

$$T_c = T_a + \frac{G_{poa}}{u_0 + u_1 v}$$

**DC and AC power** use the PVWatts models:

$$P_{dc} = P_{dc0}\frac{G_{eff}}{1000}\left[1 + \gamma (T_c - 25)\right](1 - L_{dc})$$

The PVWatts inverter model applies a part-load efficiency curve (η_nom = 0.96) and clips the output at the inverter AC rating.

Over 2020–2024 this gives 1,560–1,730 kWh/kWp per year (1,630–1,790 with clean panels). The implementation matches pvlib's `ModelChain` within 1 % (unit test).

The original model is kept as `pv.model: simple`. It treats the panels as horizontal (POA = GHI) and uses a NOCT cell temperature, a constant 5 % soiling loss and a 0.90 system derate.

*Limitations:*
- No shading, snow or spectral effects, and no module degradation.
- Diffuse IAM is ignored.
- Hourly averaging smooths clipping, so clipping losses are underestimated.
- The soiling rate is taken from Lahore, not measured in Karachi.

## Demand

**Base load.** Base load is the sum of appliances defined in a YAML profile (household or shop). In each hour, appliance i is on with probability p_i during its allowed hours and draws k_i kW. Fans scale with outdoor temperature:

$$k_{fan} = k \cdot \text{clip}\left(\frac{T_a - T_{lo}}{T_{hi} - T_{lo}}, 0, 1\right)$$

Weekend and Ramadan variants override appliance hours, probabilities and AC occupancy. The hourly total is multiplied by lognormal noise, exp(N(0, σ²)) with σ = 0.15. Appliances are either *essential* (lights, fridge, fans, internet) or *other* (TV, kitchen, chargers).

**Flexible load.** The flexible load (water pump, washing machine, iron) needs E_flex kWh per day, at most P_flex kW per hour. Any energy not done by the deadline hour counts as missed.

**Measured demand.** An inverter-app log (timestamp, load kW, any interval) can replace the synthetic profile. It is averaged to hourly values, and hours with less than half their samples are treated as missing. Because the log already contains the AC and the flexible loads, the AC model and the flexible load are switched off in this mode. `scripts/compare_demand.py` compares simulated and logged demand (hourly, daily energy, mean profile).

*Limitations:*
- Appliance hours are independent Bernoulli draws, so there is no correlation between hours or appliances beyond the shared noise.
- Profiles are not yet calibrated with real logs.
- Fan demand uses outdoor, not indoor, temperature.

## Room thermal model and air conditioning

One air-conditioned room is modelled as a first-order RC network:

$$C\frac{dT}{dt} = \frac{T_a - T}{R} + A\frac{\text{GHI}}{1000} + Q_{int} - Q_{ac}$$

With constant inputs over the hour, the exact solution is

$$T(t) = T_{eq} + (T_0 - T_{eq})e^{-t/RC}, \qquad T_{eq} = T_a + R\left(A\frac{\text{GHI}}{1000} + Q_{int} - Q_{ac}\right)$$

**AC thermostat.** The AC delivers the cooling needed to reach the setpoint by the end of the hour, capped at its capacity. Electric power is P_ac = Q_ac / COP.

**AC modes.**
- *Normal:* the AC runs only in occupied hours, at the normal setpoint (26 °C).
- *Pre-cool:* the AC runs at 23 °C in any hour.

**Outages.** During an outage the AC is served only if the battery, solar and inverter backup limit allow. If it is only partly served, the delivered cooling is reduced by the same fraction, and the room warms toward T_eq. Pre-cooling before an expected outage is therefore a real option.

**Comfort.** Comfort loss is measured in degree-hours: the hour-mean room temperature above the setpoint, summed over occupied hours.

The original rule-based AC model is kept as `cooling.model: rule` (1.5 kW × clip((T_a − 29)/5) in fixed hours).

*Limitations:*
- One lumped zone, with no humidity or latent load.
- Constant COP (in reality COP falls at high outdoor temperature).
- Solar gains use GHI, not façade irradiance.
- R and C are not fitted to a real building.

## Battery (48 V LFP behind a hybrid inverter)

**State of charge** follows the cell energy:

$$SoC_{t+1} = SoC_t + \frac{E_{in,dc} - E_{out,dc}}{C_{nom}\,SoH}$$

SoC is kept within [SoC_min,grid, SoC_max] while the grid is on, and within [SoC_cutoff, SoC_max] during an outage.

**Power limits** default to 0.5C and taper linearly:
- Charge power falls from 0.5C at 90 % SoC to 0.05C at 100 % (the constant-voltage phase).
- Discharge power falls from 0.5C at 15 % SoC to 0.1C at the floor.

Limits are evaluated at the SoC at the start of the hour.

**Conversion losses** depend on the AC-side power P:

$$L(P) = a P_r + b P + c \frac{P^2}{P_r}$$

- Charging stores P − L(P).
- Discharging draws P + L(P) from the cells.
- One-way efficiency peaks at about 95–96 % at moderate power. Round trip is about 90 % at 1.5 kW and lower at very low or high power.
- The inverter also draws a constant standby power every hour.

**Degradation.**

*Cycle ageing* uses incremental depth-of-discharge counting. A half-cycle starts at each change of charge/discharge direction. Its depth grows from d₁ to d₂ within the hour, adding damage

$$D = \tfrac12\left[\Phi(d_2) - \Phi(d_1)\right], \qquad \Phi(d) = \frac{(d/d_{ref})^k}{N_{ref}}$$

with N_ref = 6000 cycles at d_ref = 95 %, and k = 1, so ageing follows throughput.

*Calendar ageing* removes

$$\frac{f_{25}\, Q_{10}^{(T-25)/10}}{8760}$$

of state of health per hour.

State of health falls by (D_cycle + D_calendar)(1 − SoH_EOL), and usable capacity scales with it. The wear cost in the reward is the damage (as a fraction of battery life) times the replacement cost. At datasheet conditions this equals the original flat Rs 7 per kWh discharged.

The original battery model is kept as `battery.model: simple` (3 kW, 95 % one-way efficiency, flat Rs 7/kWh wear).

*Limitations:*
- No temperature effect on capacity or efficiency, and the battery temperature is set equal to outdoor air.
- No SoC dependence of calendar ageing.
- Hour-level half-cycle counting misses cycles shorter than an hour.
- Taper and loss parameters are generic, not fitted to one product.

## Hybrid inverter and dispatch

**Grid on.** The battery mode decides the flows. The net load is N = E + O + C + F + P_standby − PV (essential, other, AC, flexible, standby, solar):

| Mode | Surplus (N < 0) | Deficit (N > 0) |
|---|---|---|
| grid_charge | charges battery, rest exported | grid; grid also charges the battery up to its limit |
| solar_charge | charges battery, rest exported | grid |
| hold | exported | grid |
| discharge | charges battery, rest exported | battery down to SoC_min,grid, then grid |
| reserve(r) | charges battery, rest exported | battery down to r, then grid; grid charges only while SoC < r, up to r |
| setpoint | explicit charge/discharge power (LP oracle) | |

**Grid off.** The inverter islands whatever mode was chosen:
- Solar plus battery (down to the backup cutoff) serve, in order: standby, essential, AC, other, flexible.
- Total output is limited by the backup power rating.
- Surplus solar charges the battery, and anything left over is curtailed.
- At each outage start, essential load goes unserved for the transfer time (10 ms, which is negligible).

*Limitations:*
- One AC bus. PV-to-battery DC coupling, inverter surge limits and phase balancing are not modelled.

## Load-shedding

Grid is off when an hour is in a *scheduled* block **or** an *unscheduled* outage is in progress.

**Scheduled blocks** come from one of two sources:
- *Synthetic:* n blocks of k hours a day, evenly spaced, with a random daily offset per episode.
- *Published schedule:* a CSV with feeder category, start and end time, mapped to the outage levels.

**Unscheduled outages** use a semi-Markov model. In each grid-on hour an outage starts with probability

$$p = p_{level}\; f_{hour}(h)\; f_{month}(m)\; \left[1 + \kappa (T_a - T_0)^+\right]\; \lambda$$

where λ is an intensity multiplier. Its duration is drawn from a fitted distribution (lognormal, Weibull, gamma or exponential), rounded up to whole hours and capped.

The original 2-state Markov chain, which has a constant end probability and therefore geometric durations, is kept as `outages.model: markov`.

**Replay.** A recorded outage log (timestamp, grid on/off) can replace both. An hour is "off" if the grid was off for at least half of it.

**Fitting.** `scripts/fit_outages.py` estimates the model from such a log in four steps:
1. Hours of the day that are off on most days are treated as the schedule.
2. Durations are fitted by maximum likelihood, and the distribution is chosen by AIC.
3. Start probabilities and hour and month factors are computed as conditional start rates.
4. The heat factor is estimated by least squares.

During an outage level of `random`, the level is drawn per episode (domain randomisation). The agent never sees unscheduled outages in advance. It sees only the published schedule.

*Limitations:*
- Parameters are placeholders until fitted to a real log.
- Hourly resolution hides short trips.
- Voltage sags and brownouts are not modelled.

## Tariff

$$\text{cost}_t = I_t\, r(m,h)\,(1 + \tau_{energy}) - X_t\, r_{exp} + \frac{F_{month}(1+\tau_{fixed})}{\text{hours in month}}$$

Here I_t is imported kWh, r(m, h) the time-of-use rate, τ the tax percentages, X_t exported kWh, r_exp the export rate (net billing, or legacy net metering as a scenario) and F_month the fixed monthly charge. Fixed charges are reported but left out of the reward, because no action can change them.

*Limitations:*
- No slab (block) rates.
- Net metering is modelled as a flat export credit, not unit netting over the billing period.
- **Current rates are placeholders until replaced with the official notice.**

## Reward

$$r_t = -0.01\Big[\text{energy cost}_t + \text{wear}_t + 200\,U^{E}_t + 30\,U^{O}_t + 10\,\text{DH}_t + 60\,M^{flex}_t\Big]$$

| Symbol | Meaning |
|---|---|
| U^E, U^O | Unserved essential and other load (kWh) |
| DH | Comfort degree-hours |
| M^flex | Flexible energy missed at the deadline (kWh) |

All weights are in Rs and listed in ASSUMPTIONS.md.

## Reinforcement-learning interface

**Actions.** Either:
- *Reserve space* (default, 16 actions): target battery reserve {10, 30, 50, 70 %} × flexible load {delay, run} × AC {normal, pre-cool}. The reserve dispatch controller executes them.
- *Legacy space* (8 actions): battery mode {grid_charge, solar_charge, hold, discharge} × flexible load {delay, run}.

**Observation.** The extended observation (36 values) contains:
- time features
- SoC
- solar now
- demand now and expected AC demand
- flexible energy left
- grid status and outage history
- outage level
- price
- next scheduled outage
- temperatures
- noisy 6-hour forecasts
- room temperature, occupancy, weekend and Ramadan flags
- state of health

## Oracle (upper bound)

The oracle solves one linear program over the episode with perfect knowledge of solar, demand, temperature and grid status. These are read from a copy of the simulator, which is possible because they do not depend on actions.

- **Decision variables, per hour:** import, export, charge, discharge, curtailment, unserved load, AC power, flexible energy, room temperature and comfort slack.
- **Constraints:** energy balance, battery dynamics with constant efficiencies, SoC and power bounds, no grid during outages, backup limit, the linear RC room dynamics, and daily flexible-energy requirements.
- **Objective:** the reward (with cycle wear linear in throughput) plus the constant calendar wear.

The LP is solved with HiGHS (scipy). Its plan is then replayed in the real simulator, and the LP is re-solved when the battery state drifts more than 3 % from the plan. Because taper and power-dependent losses are linearised, the result is a near-upper bound rather than an exact one.

## Validation

**Automatic tests (`tests/`)** check:
- energy balance in every hour, for every scenario and battery mode
- the SoC window and power limits
- no import or export during outages
- reproducibility from one seed
- identical results however a configuration is specified (preset file, overrides, or saved config)
- exact reproduction of the original simulator on 32 recorded episodes (`configs/legacy_v1.yaml`)
- PV yield against pvlib `ModelChain`
- battery limits, efficiency and ageing
- tariff arithmetic
- outage schedules, durations, replay and parameter fitting

**Validation report.** `scripts/validate_sim.py` writes a report covering monthly yield, demand profiles, outage durations, SoC behaviour and annual cost per baseline. Where real logs are supplied, it compares simulated and real data and reports the error.

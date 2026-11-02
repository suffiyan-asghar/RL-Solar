"""RL side: action spaces, the reserve dispatch controller, baselines, the LP oracle,
evaluation statistics and a tiny end-to-end training run."""
import numpy as np
import pandas as pd
import pytest

from solar_rl import SolarHomeEnv, load_config
from solar_rl.baselines import default_baselines, make_baseline
from solar_rl.evaluation import episode_plan, evaluate, load_model, run_episode, summarise
from solar_rl.sim import Dispatch, Simulator
from tests.conftest import synth_overrides


def test_reserve_action_space_mapping(cfg_factory):
    env = SolarHomeEnv(cfg_factory())
    assert env.action_space.n == 16
    seen = set()
    for a in range(16):
        d = env.action_to_dispatch(a)
        assert d.battery == "reserve"
        seen.add((d.reserve, d.run_flex, d.ac))
        assert env.dispatch_to_action(d) == a
    assert len(seen) == 16
    assert {r for r, _, _ in seen} == {0.1, 0.3, 0.5, 0.7}


def test_legacy_action_space_mapping(cfg_factory):
    env = SolarHomeEnv(cfg_factory(extra={"rl": {"action_space": "legacy"}}))
    assert env.action_space.n == 8
    assert env.action_to_dispatch(7) == Dispatch(battery="discharge", run_flex=True)


def test_reserve_controller(cfg_factory):
    """Discharges only above the reserve; charges from the grid only below it."""
    sim = Simulator(cfg_factory(extra={"outages": {"level": "light", "intensity": 0.0}}))
    for reserve, soc0 in ((0.5, 0.8), (0.5, 0.2), (0.3, 0.3)):
        sim.reset(seed=0, soc=soc0)
        while not sim.done:
            before = sim.batt.soc
            f = sim.step(Dispatch(battery="reserve", reserve=reserve))
            if before >= reserve - 1e-9:
                assert f["soc"] >= reserve - 1e-6                     # never discharges below the reserve
                grid_charge = max(0.0, f["charged"] - max(f["solar"] - (f["essential"] + f["other"] + f["cooling"]
                                                                       + f["flex"] + f["standby"]), 0.0))
                assert grid_charge < 1e-6                            # no grid charging at/above reserve
            else:
                assert f["soc"] <= max(reserve, before) + 1e-6 or f["solar"] > 0   # grid only tops up to reserve
        if soc0 < reserve:
            assert sim.log[0]["import_kwh"] > 0 and sim.log[0]["charged"] > 0


def test_all_baselines_run(cfg_factory):
    env = SolarHomeEnv(cfg_factory())
    plan = episode_plan(env, 1, ["heavy"])
    df = evaluate(env, [make_baseline(n) for n in default_baselines()], plan)
    assert set(df["policy"]) == set(default_baselines())
    sweep = df[df.policy.str.startswith("battery_first_r")].set_index("policy")
    assert sweep.loc["battery_first_r70", "unserved_essential_kwh"] <= sweep.loc["battery_first_r10", "unserved_essential_kwh"] + 1e-9


@pytest.mark.parametrize("level", ["light", "heavy"])
def test_oracle_beats_baselines(cfg_factory, level):
    env = SolarHomeEnv(cfg_factory())
    plan = episode_plan(env, 2, [level], base_seed=5)
    names = ["solar_first", "battery_first", "grid_first", "hand_tuned", "battery_first_r30", "oracle"]
    df = evaluate(env, [make_baseline(n) for n in names], plan)
    mean_reward = df.groupby("policy")["reward"].mean()
    best_other = mean_reward.drop("oracle").max()
    assert mean_reward["oracle"] >= best_other - 0.05 * abs(best_other)


def test_oracle_on_legacy_config():
    env = SolarHomeEnv(load_config("legacy_v1", synth_overrides()))
    plan = episode_plan(env, 1, ["medium"])
    df = evaluate(env, [make_baseline(n) for n in ["battery_first", "oracle"]], plan)
    r = df.set_index("policy")["reward"]
    assert r["oracle"] >= r["battery_first"] - 0.05 * abs(r["battery_first"])


def test_summarise_mean_std_across_seeds():
    df = pd.DataFrame({"policy": ["a"] * 4, "eval_seed": [0, 0, 1, 1], "reward": [1.0, 3.0, 5.0, 7.0]})
    s = summarise(df, metrics=["reward"])
    assert s.loc["a", ("reward", "mean")] == pytest.approx(4.0)      # seed means 2 and 6
    assert s.loc["a", ("reward", "std")] == pytest.approx(np.std([2.0, 6.0], ddof=1))


def test_tiny_training_run_and_evaluation(tmp_path):
    from scripts.train import train
    tc = {"algo": "ppo", "scenario": "default", "env_overrides": synth_overrides(),
          "train_years": [2021], "val_years": [2021], "total_timesteps": 256, "n_envs": 2, "eval_freq": 128,
          "n_eval_episodes": 1,
          "hyperparameters": {"learning_rate": 3e-4, "n_steps": 64, "batch_size": 64, "n_epochs": 1,
                              "gamma": 0.995, "net_arch": [16, 16]}}
    runs = train(tc, [0], str(tmp_path), verbose=0)
    pol = load_model(f"{runs[0]}/best_model.zip", "ppo")
    assert pol.rl["action_space"] == "reserve"
    env = SolarHomeEnv(load_config("default", synth_overrides()))
    stats, tr = run_episode(env, pol, episode_plan(env, 1, ["medium"])[0], record=True)
    assert len(tr) == 168 and set(tr["mode"]) == {"reserve"}

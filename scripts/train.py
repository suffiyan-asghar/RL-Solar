"""Train PPO or DQN with several seeds. Hyperparameters live in configs/train/*.yaml.
The validation years pick the best checkpoint (EvalCallback).

    python scripts/train.py --train-config configs/train/ppo.yaml --seeds 0 1 2 3 4
    python scripts/train.py --train-config configs/train/dqn.yaml --seeds 0
    python scripts/train.py --train-config configs/train/ppo.yaml --timesteps 50000   # quick check

Each run is saved to <out>/<algo>_seed<seed>/: best_model.zip, final_model.zip,
env_config.yaml (the fully resolved simulator config, so evaluation rebuilds the same
action/observation spaces) and train_config.json.
"""
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=UserWarning)

import yaml  # noqa: E402

from solar_rl import SolarHomeEnv, Simulator, load_config  # noqa: E402
from solar_rl.config import merge  # noqa: E402


def load_train_config(path, overrides=None):
    with open(path) as f:
        tc = yaml.safe_load(f)
    return merge(tc, overrides or {})


def train(tc, seeds, out="models", timesteps=None, verbose=1):
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor

    algo = tc["algo"]
    timesteps = int(timesteps or tc["total_timesteps"])
    env_over = merge(tc.get("env_overrides") or {}, {})
    train_cfg = load_config(tc.get("scenario", "default"), merge(env_over, {"data": {"years": tc["train_years"]}}))
    val_cfg = load_config(tc.get("scenario", "default"),
                          merge(env_over, {"data": {"years": tc["val_years"]}, "outages": {"level": "random"}}))
    train_sim = Simulator(train_cfg)                # weather + PV computed once, shared by all envs
    val_sim = Simulator(val_cfg)
    hp = dict(tc["hyperparameters"])
    net_arch = hp.pop("net_arch", [128, 128])
    runs = []
    for seed in seeds:
        run = os.path.join(out, f"{algo}_seed{seed}")
        os.makedirs(run, exist_ok=True)
        train_cfg.save(os.path.join(run, "env_config.yaml"))
        with open(os.path.join(run, "train_config.json"), "w") as f:
            json.dump({**tc, "seed": seed, "timesteps": timesteps}, f, indent=2, default=str)
        n_envs = tc.get("n_envs", 8) if algo == "ppo" else 1
        venv = make_vec_env(lambda: SolarHomeEnv(train_cfg, sim=train_sim.fresh_copy()), n_envs=n_envs, seed=seed)
        eval_env = Monitor(SolarHomeEnv(val_cfg, sim=val_sim.fresh_copy()))
        eval_cb = EvalCallback(eval_env, best_model_save_path=run, log_path=run,
                               eval_freq=max(tc.get("eval_freq", 50_000) // n_envs, 1),
                               n_eval_episodes=tc.get("n_eval_episodes", 20), deterministic=True, verbose=verbose)
        tb = os.path.join(out, "tb")
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            tb = None
        cls = PPO if algo == "ppo" else DQN
        model = cls("MlpPolicy", venv, policy_kwargs=dict(net_arch=net_arch), seed=seed, verbose=0,
                    tensorboard_log=tb, device="cpu", **hp)
        print(f"Training {algo.upper()} seed={seed} for {timesteps:,} steps "
              f"(scenario {tc.get('scenario', 'default')}, weather {train_sim.weather.attrs['source']}, "
              f"actions {train_cfg.rl.action_space}, obs {train_cfg.rl.observation})")
        model.learn(total_timesteps=timesteps, callback=eval_cb, tb_log_name=f"{algo}_seed{seed}",
                    progress_bar=False)
        model.save(os.path.join(run, "final_model"))
        if not os.path.exists(os.path.join(run, "best_model.zip")):
            model.save(os.path.join(run, "best_model"))
        print(f"Saved {run}/best_model.zip and final_model.zip")
        runs.append(run)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-config", default="configs/train/ppo.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--timesteps", type=int, default=None, help="override total_timesteps")
    ap.add_argument("--scenario", default=None, help="override the scenario in the train config")
    ap.add_argument("--outage-level", default=None, help="train on one level: light|medium|heavy|random")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    over = {}
    if args.scenario:
        over["scenario"] = args.scenario
    if args.outage_level:
        over["env_overrides"] = {"outages": {"level": args.outage_level}}
    train(load_train_config(args.train_config, over), args.seeds, args.out, args.timesteps)


if __name__ == "__main__":
    main()

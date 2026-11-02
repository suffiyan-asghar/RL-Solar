"""Train PPO or DQN with several seeds. Validation year is used to pick the best checkpoint.

Examples
    python scripts/train.py --algo ppo --seeds 0 1 2 3 4 --timesteps 1000000
    python scripts/train.py --algo dqn --seeds 0 --timesteps 500000
    python scripts/train.py --algo ppo --seeds 0 --timesteps 50000      # quick smoke test
"""
import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=UserWarning)

from stable_baselines3 import DQN, PPO  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from solar_rl import SimConfig, SolarHomeEnv  # noqa: E402
from solar_rl.weather import load_years  # noqa: E402


def tensorboard_dir(path):
    try:
        import tensorboard  # noqa: F401
        return path
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--train-years", type=int, nargs="+", default=[2020, 2021, 2022])
    ap.add_argument("--val-years", type=int, nargs="+", default=[2023])
    ap.add_argument("--outage-level", default="random", help="light|medium|heavy|random")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    train_cfg = SimConfig(years=args.train_years, outage_level=args.outage_level)
    val_cfg = SimConfig(years=args.val_years, outage_level="random")
    train_weather = load_years(train_cfg.years, train_cfg)   # load once, share across envs
    val_weather = load_years(val_cfg.years, val_cfg)

    for seed in args.seeds:
        run = os.path.join(args.out, f"{args.algo}_seed{seed}")
        os.makedirs(run, exist_ok=True)
        with open(os.path.join(run, "config.json"), "w") as f:
            json.dump({"args": vars(args), "sim": asdict(train_cfg)}, f, indent=2, default=str)

        n_envs = args.n_envs if args.algo == "ppo" else 1
        venv = make_vec_env(lambda: SolarHomeEnv(train_cfg, train_weather), n_envs=n_envs, seed=seed)
        eval_env = Monitor(SolarHomeEnv(val_cfg, val_weather))
        eval_cb = EvalCallback(eval_env, best_model_save_path=run, log_path=run,
                               eval_freq=max(20_000 // n_envs, 1), n_eval_episodes=20,
                               deterministic=True, verbose=1)
        tb = tensorboard_dir(os.path.join(args.out, "tb"))

        if args.algo == "ppo":
            model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=1024, batch_size=256,
                        n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                        policy_kwargs=dict(net_arch=[128, 128]), seed=seed, verbose=0,
                        tensorboard_log=tb, device="cpu")
        else:
            model = DQN("MlpPolicy", venv, learning_rate=1e-4, buffer_size=200_000,
                        learning_starts=10_000, batch_size=128, gamma=0.99, train_freq=4,
                        target_update_interval=5_000, exploration_fraction=0.3,
                        exploration_final_eps=0.02, policy_kwargs=dict(net_arch=[128, 128]),
                        seed=seed, verbose=0, tensorboard_log=tb, device="cpu")

        print(f"Training {args.algo.upper()} seed={seed} for {args.timesteps:,} steps "
              f"(weather: {train_weather.attrs['source']})")
        model.learn(total_timesteps=args.timesteps, callback=eval_cb,
                    tb_log_name=f"{args.algo}_seed{seed}", progress_bar=False)
        model.save(os.path.join(run, "final_model"))
        print(f"Saved {run}/best_model.zip and final_model.zip")


if __name__ == "__main__":
    main()

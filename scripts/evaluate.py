"""Compare baselines (and optionally trained agents) on held-out test data.

Examples
    python scripts/evaluate.py                                   # baselines on 2024
    python scripts/evaluate.py --models ppo=models/ppo_seed0/best_model.zip dqn=models/dqn_seed0/best_model.zip
    python scripts/evaluate.py --levels heavy --export-rate 26   # scenario: legacy net metering, heavy outages
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd  # noqa: E402

from solar_rl import SimConfig, SolarHomeEnv  # noqa: E402
from solar_rl.baselines import ALL_BASELINES  # noqa: E402
from solar_rl.evaluation import SB3Policy, episode_plan, evaluate, season  # noqa: E402


def load_model(spec):
    name, path = spec.split("=", 1)
    from stable_baselines3 import DQN, PPO
    cls = DQN if "dqn" in name.lower() or "dqn" in path.lower() else PPO
    return SB3Policy(cls.load(path, device="cpu"), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2024])
    ap.add_argument("--episodes", type=int, default=26, help="start dates per level")
    ap.add_argument("--levels", nargs="+", default=["light", "medium", "heavy"])
    ap.add_argument("--baselines", nargs="+", default=list(ALL_BASELINES))
    ap.add_argument("--models", nargs="*", default=[], help="name=path/to/model.zip")
    ap.add_argument("--export-rate", type=float, default=None)
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()

    cfg = SimConfig(years=args.years)
    if args.export_rate is not None:
        cfg.export_rate = args.export_rate
    env = SolarHomeEnv(cfg)
    print(f"Weather source: {env.weather.attrs['source']}")

    policies = [ALL_BASELINES[b]() for b in args.baselines] + [load_model(m) for m in args.models]
    plan = episode_plan(env, args.episodes, args.levels)
    df = evaluate(env, policies, plan)
    df["season"] = df["month"].map(season)

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "episodes.csv"), index=False)

    cols = ["reward", "total_cost_incl_wear", "unserved_essential_kwh", "unmet_cooling_kwh",
            "flex_missed_kwh", "battery_cycles", "self_consumption"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("\n=== Weekly averages over all test episodes (mean) ===")
    print(df.groupby("policy")[cols].mean().sort_values("total_cost_incl_wear"))
    print("\n=== Weekly cost incl. battery wear (Rs) by outage level ===")
    print(df.pivot_table(index="policy", columns="level", values="total_cost_incl_wear", aggfunc="mean"))
    print("\n=== Unserved essential load (kWh/week) by outage level ===")
    print(df.pivot_table(index="policy", columns="level", values="unserved_essential_kwh", aggfunc="mean"))
    print("\n=== Weekly cost by season ===")
    print(df.pivot_table(index="policy", columns="season", values="total_cost_incl_wear", aggfunc="mean"))
    summary = df.groupby("policy")[cols].agg(["mean", "std"])
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    print(f"\nSaved {args.out}/episodes.csv and summary.csv")


if __name__ == "__main__":
    main()

"""Compare baselines, the oracle and trained agents on held-out test data.

    python scripts/evaluate.py                                      # all baselines + oracle, 5 seeds, 2024
    python scripts/evaluate.py --models "ppo=models/ppo_seed{seed}/best_model.zip" \\
                               "dqn=models/dqn_seed{seed}/best_model.zip"
    python scripts/evaluate.py --export-rates 11 26 --levels heavy  # net billing vs legacy net metering
    python scripts/evaluate.py --scenario shop --episodes 6 --seeds 0 1

{seed} in a model path is replaced by the evaluation seed, so seed s is evaluated with the
agent trained with seed s. Outputs in --out:
    episodes.csv   one row per (policy, episode, seed, export rate)
    summary.csv    mean and std across seeds of every metric, per policy and export rate
    summary.md     tables: overall, by outage level, by season, by export rate (mean +/- std)
    scatter_<rate>.png   cost vs. unserved essential load, reserve sweep as a line
"""
import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd  # noqa: E402

from solar_rl import SolarHomeEnv, load_config  # noqa: E402
from solar_rl.baselines import default_baselines, make_baseline  # noqa: E402
from solar_rl.evaluation import (episode_plan, evaluate, load_model, mean_pm_std, scatter_plot, season,  # noqa: E402
                                 summarise)

KEY_METRICS = ["total_cost_incl_wear", "unserved_essential_kwh", "comfort_degree_hours", "wear_cost",
               "battery_cycles", "reward"]


def run(args):
    over = {"data": {"years": args.years}}
    base_cfg = load_config(args.scenario, over)
    base_env = SolarHomeEnv(base_cfg)
    print(f"Scenario {args.scenario} | weather {base_env.weather.attrs['source']} | years {args.years}")
    rates = args.export_rates or [base_cfg.tariff.export.rate]
    names = args.baselines or default_baselines(include_oracle=not args.no_oracle)
    rows = []
    t0 = time.time()
    for rate in rates:
        cfg = base_cfg.copy({"tariff": {"export": {"rate": float(rate)}}})
        env = SolarHomeEnv(cfg, sim=base_env.sim)
        for seed in args.seeds:
            policies = [make_baseline(n) for n in names]
            for spec in args.models:
                name, path = spec.split("=", 1)
                p = path.format(seed=seed)
                if not os.path.exists(p):
                    print(f"  (skipping {name}: {p} not found)")
                    continue
                policies.append(load_model(p, name))
            plan = episode_plan(env, args.episodes, args.levels, base_seed=10000 * (seed + 1))
            df = evaluate(env, policies, plan)
            df["eval_seed"] = seed
            df["export_rate"] = rate
            rows.append(df)
            print(f"  export {rate:g} seed {seed}: {len(df)} episodes ({time.time() - t0:.0f}s)")
    df = pd.concat(rows, ignore_index=True)
    df["season"] = df["month"].map(season)
    return df


def report(df, out):
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "episodes.csv"), index=False)
    summ = summarise(df, by=("export_rate", "policy"))
    summ.to_csv(os.path.join(out, "summary.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    lines = [f"# Evaluation\n\n{df['eval_seed'].nunique()} seeds x {df.groupby(['eval_seed', 'export_rate', 'policy']).size().iloc[0]} "
             f"episodes per policy. Values are weekly means, mean ± std across seeds.\n"]
    for rate in sorted(df["export_rate"].unique()):
        d = df[df["export_rate"] == rate]
        s = summarise(d)
        s = s.sort_values(("reward", "mean"), ascending=False)
        tab = pd.DataFrame({m: mean_pm_std(s, m) for m in KEY_METRICS if (m, "mean") in s})
        lines += [f"\n## Export rate Rs {rate:g}/kWh: overall (sorted by reward)\n", tab.to_markdown(), ""]
        print(f"\n=== Export rate {rate:g}: weekly means, mean +/- std across seeds ===")
        print(tab.to_string().replace("±", "+/-"))
        for by in ("level", "season"):
            sb = summarise(d, by=("policy", by), metrics=["total_cost_incl_wear", "unserved_essential_kwh"])
            cost = mean_pm_std(sb, "total_cost_incl_wear").unstack(by)
            uns = mean_pm_std(sb, "unserved_essential_kwh").unstack(by)
            lines += [f"\n### Cost incl. wear (Rs/week) by {by}\n", cost.to_markdown(),
                      f"\n### Unserved essential load (kWh/week) by {by}\n", uns.to_markdown(), ""]
            if by == "level":
                print(f"\n--- cost by outage level ---\n{cost.to_string()}")
        scatter_plot(s, os.path.join(out, f"scatter_{rate:g}.png"),
                     title=f"Cost vs. unserved essential load (export Rs {rate:g}/kWh)")
    if df["export_rate"].nunique() > 1:
        se = summarise(df, by=("policy", "export_rate"), metrics=["total_cost_incl_wear"])
        tab = mean_pm_std(se, "total_cost_incl_wear").unstack("export_rate")
        lines += ["\n## Cost incl. wear (Rs/week) by export rate\n", tab.to_markdown()]
        print(f"\n--- cost by export rate ---\n{tab.to_string()}")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved {out}/episodes.csv, summary.csv, summary.md and scatter plots")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="default")
    ap.add_argument("--years", type=int, nargs="+", default=[2024])
    ap.add_argument("--episodes", type=int, default=13, help="start dates per level and seed")
    ap.add_argument("--levels", nargs="+", default=["light", "medium", "heavy"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--baselines", nargs="+", default=None, help="default: all + reserve sweep + oracle")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--models", nargs="*", default=[], help='name=path, "{seed}" is replaced by the seed')
    ap.add_argument("--export-rates", type=float, nargs="+", default=None)
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()
    report(run(args), args.out)


if __name__ == "__main__":
    main()

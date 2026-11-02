"""Train PPO and DQN with 5 seeds each, then run the full evaluation (E1, E2 and E6).

    python scripts/run_experiments.py                       # full: 2 x 5 runs of 2M steps (hours)
    python scripts/run_experiments.py --timesteps 20000 --seeds 0 1 --episodes 2   # quick check
    python scripts/run_experiments.py --skip-train          # evaluate existing models only

Models go to <models>/<algo>_seed<seed>/, results to <out>/ (see scripts/evaluate.py).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import evaluate as ev  # noqa: E402
from scripts.train import load_train_config, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["ppo", "dqn"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--timesteps", type=int, default=None, help="override configs/train/<algo>.yaml")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results/experiments")
    ap.add_argument("--episodes", type=int, default=13)
    ap.add_argument("--export-rates", type=float, nargs="+", default=[11.0, 26.0])
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_train:
        for algo in args.algos:
            tc = load_train_config(os.path.join("configs", "train", f"{algo}.yaml"),
                                   {"scenario": args.scenario} if args.scenario else None)
            train(tc, args.seeds, args.models, args.timesteps, verbose=0)
            print(f"[{time.time() - t0:.0f}s] trained {algo} seeds {args.seeds}")

    eval_args = argparse.Namespace(
        scenario=args.scenario or "default", years=[2024], episodes=args.episodes,
        levels=["light", "medium", "heavy"], seeds=args.seeds, baselines=None, no_oracle=False,
        models=[f"{a}=" + os.path.join(args.models, f"{a}_seed{{seed}}", "best_model.zip") for a in args.algos],
        export_rates=args.export_rates, out=args.out)
    ev.report(ev.run(eval_args), args.out)
    print(f"[{time.time() - t0:.0f}s] done")


if __name__ == "__main__":
    main()

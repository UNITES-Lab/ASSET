from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asset", description="Acquisition-sensitive subspace estimation for test-time adaptation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Download a dataset and build its tasks")
    prepare.add_argument("dataset", choices=["slake", "radimgnet", "gmai", "mimic", "vrb"])
    prepare.add_argument("--data-dir", type=Path, default=Path("data"))
    prepare.add_argument("--workers", type=int, default=8)
    prepare.add_argument("--device", default="cuda")

    adapt = subparsers.add_parser("adapt", help="Adapt a source LoRA model on a task")
    adapt.add_argument("--adapter", type=Path, required=True)
    adapt.add_argument("--task", type=Path, required=True)
    adapt.add_argument("--output", type=Path, required=True)
    adapt.add_argument(
        "--variant", choices=["asset", "random_subspace", "no_subspace", "fixed_subspace"], default="asset"
    )
    adapt.add_argument("--rounds", type=int, default=5)
    adapt.add_argument("--steps", type=int, default=8)
    adapt.add_argument("--support-size", type=int, default=32)
    adapt.add_argument("--reference-size", type=int, default=32)
    adapt.add_argument("--support-batch-size", type=int, default=2)
    adapt.add_argument("--reference-batch-size", type=int, default=4)
    adapt.add_argument("--sensitivity-views", type=int, default=2)
    adapt.add_argument("--consistency-views", type=int, default=2)
    adapt.add_argument("--proxy-bank", choices=["full", "general", "modality"], default="full")
    adapt.add_argument("--lr", type=float, default=1e-3)
    adapt.add_argument("--lr-lora", type=float, default=1e-4)
    adapt.add_argument("--lambda-cons", type=float, default=1.0)
    adapt.add_argument("--lambda-src", type=float, default=1.0)
    adapt.add_argument("--lambda-pres", type=float, default=1.0)
    adapt.add_argument("--lambda-alpha", type=float, default=1e-3)
    adapt.add_argument("--energy", type=float, default=0.95)
    adapt.add_argument("--rank-cap", type=int, default=16)
    adapt.add_argument("--energy-floor", type=float, default=1e-8)
    adapt.add_argument("--beta", type=float, default=0.7)
    adapt.add_argument("--gamma", type=float, default=1.0)
    adapt.add_argument("--uniform-weighting", action="store_true")
    adapt.add_argument("--seed", type=int, default=2026)
    adapt.add_argument("--device", default="cuda")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate an adapter on a task")
    evaluate.add_argument("--adapter", type=Path, required=True)
    evaluate.add_argument("--task", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        if args.dataset == "mimic":
            from asset.data.mimic import prepare

            prepare(args.data_dir)
        elif args.dataset == "vrb":
            from asset.data.vrb import prepare

            prepare(args.data_dir, args.workers)
        else:
            from asset.data.medical import prepare

            prepare(args.dataset, args.data_dir, args.workers, args.device)
    elif args.command == "adapt":
        from asset.adapt import adapt

        adapt(args)
    else:
        from asset.evaluate import evaluate

        evaluate(args)
    return 0

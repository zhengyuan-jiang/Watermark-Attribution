#!/usr/bin/env python3
"""Run a small, reproducible watermark-selection comparison."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watermark_attribution.metrics import maximum_pairwise_accuracy  # noqa: E402
from watermark_attribution.selection import select_watermarks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Random, NRG, and A-BSTA on a small watermark pool."
    )
    parser.add_argument(
        "--method",
        choices=("all", "random", "nrg", "absta"),
        default="all",
        help="selection method to run",
    )
    parser.add_argument("--users", type=int, default=25, help="number of watermarks")
    parser.add_argument("--length", type=int, default=64, help="watermark length")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="A-BSTA recursion depth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = ("random", "nrg", "absta") if args.method == "all" else (args.method,)

    print(
        f"users={args.users}, length={args.length}, seed={args.seed}, "
        f"A-BSTA depth={args.depth}"
    )
    print(f"{'method':<10} {'seconds':>10} {'max pairwise BA':>18}")
    for method in methods:
        start = time.perf_counter()
        pool = select_watermarks(
            method=method,
            num_users=args.users,
            n=args.length,
            seed=args.seed,
            depth=args.depth,
        )
        elapsed = time.perf_counter() - start
        if args.users < 2:
            maximum = "n/a"
        else:
            maximum = f"{maximum_pairwise_accuracy(pool):.6f}"
        print(f"{method:<10} {elapsed:>10.4f} {maximum:>18}")


if __name__ == "__main__":
    main()

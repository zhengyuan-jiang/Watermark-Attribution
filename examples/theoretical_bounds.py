#!/usr/bin/env python3
"""Print the theoretical bounds from Theorems 1, 3, and 4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watermark_attribution.theory import (  # noqa: E402
    fdr_upper_bound_independent,
    tar_lower_bound,
    tdr_lower_bound,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute theoretical TDR, FDR, and TAR bounds. The defaults "
            "reproduce the setting reported in Table 5 of the paper."
        )
    )
    parser.add_argument("--users", type=int, default=100_000_000, help="number of users s")
    parser.add_argument("--length", type=int, default=64, help="watermark length n")
    parser.add_argument("--beta", type=float, default=0.99, help="decoder accuracy beta")
    parser.add_argument("--gamma", type=float, default=0.05, help="decoder randomness gamma")
    parser.add_argument("--tau", type=float, default=0.9, help="detection threshold tau")
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.2,
        help="minimum pairwise accuracy used by the TDR bound",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=0.8,
        help="maximum pairwise accuracy used by the TAR bound",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bounds = {
        "TDR lower bound": tdr_lower_bound(
            n=args.length,
            beta=args.beta,
            tau=args.tau,
            alpha_min=args.alpha_min,
        ),
        "FDR upper bound": fdr_upper_bound_independent(
            s=args.users,
            n=args.length,
            tau=args.tau,
            gamma=args.gamma,
        ),
        "TAR lower bound": tar_lower_bound(
            n=args.length,
            beta=args.beta,
            tau=args.tau,
            alpha_max=args.alpha_max,
        ),
    }

    print(
        f"s={args.users:,}, n={args.length}, beta={args.beta}, "
        f"gamma={args.gamma}, tau={args.tau}"
    )
    for name, value in bounds.items():
        print(f"{name:17}: {value:.8f} ({value:.4%})")


if __name__ == "__main__":
    main()

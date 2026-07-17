"""Run a configurable graph-conditioned trust-region search on one instance."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .baselines import gctr_policy
from .graphs import FAMILIES, make_instance
from . import __version__


def _vector(values, name):
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or out.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Optimize one exact-statevector MaxCut instance with a user-supplied "
            "Gaussian search policy. Mean and variance contain [gamma_1, beta_1, ...]."
        )
    )
    parser.add_argument("--version", action="version",
                        version=f"specops-gctr {__version__}")
    parser.add_argument("--family", choices=FAMILIES, default="er")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mean", nargs="+", type=float,
                        default=[0.4, 0.6, 0.8, 0.3])
    parser.add_argument("--variance", nargs="+", type=float,
                        default=[0.05, 0.05, 0.05, 0.05])
    parser.add_argument("--heuristic", nargs="+", type=float, default=None)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--budget-cap", type=int, default=None)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target", type=float, default=None,
        help="absolute objective value that triggers early stopping")
    target_group.add_argument(
        "--target-fraction", type=float, default=None,
        help="fraction of exact MaxCut that triggers early stopping; this is "
             "not the manuscript's fraction of an offline reference value")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--step-size", type=float, default=0.3)
    parser.add_argument("--gaussian-seeds", type=int, default=0)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--no-trust-region", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    mean = _vector(args.mean, "mean")
    variance = _vector(args.variance, "variance")
    heuristic = None if args.heuristic is None else _vector(args.heuristic, "heuristic")
    if mean.size % 2 or variance.size != mean.size:
        parser.error("mean must have even length and variance must have the same length")
    if heuristic is not None and heuristic.size != mean.size:
        parser.error("heuristic must have the same length as mean")
    if np.any(variance <= 0):
        parser.error("variance entries must be positive")
    if args.budget < 1 or args.n < 2:
        parser.error("n must be at least 2 and budget must be positive")
    if (args.target_fraction is not None
            and not 0 < args.target_fraction <= 1):
        parser.error("target-fraction must lie in (0, 1]")
    if args.target is not None and not np.isfinite(args.target):
        parser.error("target must be finite when supplied")
    if args.radius <= 0 or args.step_size <= 0:
        parser.error("radius and step-size must be positive")
    if args.gaussian_seeds < 0:
        parser.error("gaussian-seeds must be nonnegative")
    if args.budget_cap is not None and args.budget_cap < 1:
        parser.error("budget-cap must be positive when supplied")
    if args.shots is not None and args.shots < 1:
        parser.error("shots must be positive when supplied")

    instance = make_instance(args.family, args.n, args.seed)
    target = args.target
    if args.target_fraction is not None:
        target = args.target_fraction * instance.maxcut
    result = gctr_policy(
        instance,
        budget=args.budget,
        mu=mean,
        sigma_diag=variance,
        radius_scale=args.radius,
        rng=np.random.default_rng(args.seed),
        sample_seed=args.seed,
        target=target,
        heuristic_angles=heuristic,
        use_heuristic_seed=heuristic is not None,
        n_gaussian_seeds=args.gaussian_seeds,
        budget_cap=args.budget_cap,
        use_trust_region=not args.no_trust_region,
        shots=args.shots,
        step0=args.step_size,
    )
    payload = {
        "family": args.family,
        "n": args.n,
        "depth": mean.size // 2,
        "seed": args.seed,
        "maxcut": instance.maxcut,
        "expectation": result["value"],
        "expectation_ratio": result["value"] / instance.maxcut,
        "sampled_best_ratio": result["ratio"],
        "evaluations": result["evaluations"],
        "evaluations_used": result["evaluations_used"],
        "reached_target": result["reached_target"],
        "parameters": np.asarray(result["params"]).tolist(),
    }
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"GCTR family={args.family} n={args.n} p={payload['depth']} "
            f"evaluations={payload['evaluations']} expectation_ratio="
            f"{payload['expectation_ratio']:.4f}"
        )
        print("parameters=" + np.array2string(np.asarray(payload["parameters"]), precision=6))
    return 0


if __name__ == "__main__":
    sys.exit(main())

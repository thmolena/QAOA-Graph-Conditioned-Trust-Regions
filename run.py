"""Repository dispatcher for the two supported command-line workflows."""

from __future__ import annotations

import sys


USAGE = """usage: python run.py {optimize,reproduce,portfolio} [arguments]

commands:
  optimize   run one configurable GCTR optimization
  reproduce  run, replot, or validate the manuscript experiment
  portfolio  run the target-free, fixed-budget optimizer portfolio
"""


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE, end="")
        return 0

    command, forwarded = args[0], args[1:]
    if command == "optimize":
        from specops_gctr.cli import main as optimize_main
        return optimize_main(forwarded)
    if command == "reproduce":
        from specops_gctr.reproduce import main as reproduce_main
        return reproduce_main(forwarded)
    if command == "portfolio":
        from specops_gctr.portfolio_experiment import main as portfolio_main
        return portfolio_main(forwarded)

    print(f"unknown command: {command!r}\n\n{USAGE}", file=sys.stderr,
          end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

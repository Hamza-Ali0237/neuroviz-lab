"""Command line: ``nvzlab list`` and ``nvzlab demo <model> --json out.json``."""

from __future__ import annotations

import argparse
import sys

import nvzlab as nvz


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``nvzlab`` console script."""
    parser = argparse.ArgumentParser(prog="nvzlab", description="Run small ML models step by step.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list built-in models, datasets and ops")
    demo = sub.add_parser("demo", help="trace a built-in model")
    demo.add_argument("model")
    demo.add_argument("--kind", default="predict", choices=["predict", "train"])
    demo.add_argument("--samples", default=None, choices=["single", "batch"])
    demo.add_argument("--features", type=int, default=None)
    demo.add_argument("--iterations", type=int, default=1)
    demo.add_argument("--json", metavar="OUT", help="write the trace as JSON")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            print("models:   " + ", ".join(nvz.list_models()))
            print("datasets: " + ", ".join(nvz.list_datasets()))
            print("ops:      " + ", ".join(nvz.list_ops()))
        else:
            tr = nvz.trace(
                args.model,
                kind=args.kind,
                samples=args.samples,
                features=args.features,
                iterations=args.iterations,
            )
            print(tr.summary())
            if args.json:
                print(f"wrote {tr.save(args.json)}")
            else:
                print(
                    "(HTML viewer arrives in the next release step; use --json to save the trace)"
                )
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line: ``nvzlab list``, ``nvzlab demo <model>`` and ``nvzlab open trace.json``."""

from __future__ import annotations

import argparse
import sys

import nvzlab as nvz


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``nvzlab`` console script."""
    parser = argparse.ArgumentParser(prog="nvzlab", description="Run small ML models step by step.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list built-in models, datasets and ops")
    demo = sub.add_parser("demo", help="trace a built-in model and open the viewer")
    demo.add_argument("model")
    demo.add_argument("--kind", default="predict", choices=["predict", "train"])
    demo.add_argument("--samples", default=None, choices=["single", "batch"])
    demo.add_argument("--features", type=int, default=None)
    demo.add_argument("--iterations", type=int, default=1)
    demo.add_argument("--json", metavar="OUT", help="write the trace as JSON instead")
    demo.add_argument("--html", metavar="OUT", help="write the viewer page instead of opening it")
    opn = sub.add_parser("open", help="open a saved trace.json in the viewer")
    opn.add_argument("path")
    opn.add_argument("--html", metavar="OUT", help="write the viewer page instead of opening it")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            print("models:   " + ", ".join(nvz.list_models()))
            print("datasets: " + ", ".join(nvz.list_datasets()))
            print("ops:      " + ", ".join(nvz.list_ops()))
            return 0
        if args.command == "demo":
            tr = nvz.trace(
                args.model,
                kind=args.kind,
                samples=args.samples,
                features=args.features,
                iterations=args.iterations,
            )
            print(tr.summary())
        else:
            tr = nvz.Trace.load(args.path)
        if getattr(args, "json", None):
            print(f"wrote {tr.save(args.json)}")
        elif args.html:
            print(f"wrote {tr.save(args.html)}")
        else:
            tr.show()
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

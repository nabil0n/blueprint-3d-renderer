import argparse
import sys
from pathlib import Path

from blueprint3d import evaluation
from blueprint3d.evaluation.report import evaluate_directory, format_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m blueprint3d.evaluation", description=evaluation.__doc__)
    parser.add_argument("data_dir", nargs="?", type=Path, default=Path("data"))
    parser.add_argument("--truth", type=Path, help="ground truth JSON (default: <data_dir>/truth.json)")
    parser.add_argument("--out", type=Path, default=Path("eval-out"))
    args = parser.parse_args(argv)

    if not args.data_dir.is_dir():
        parser.error(f"{args.data_dir} is not a directory")
    scores = evaluate_directory(args.data_dir, args.truth or args.data_dir / "truth.json", args.out)
    sys.stdout.reconfigure(encoding="utf-8")
    print(format_table(scores))
    print(f"\nOverlays and overview.png written to {args.out}")
    return 0 if all(s.passed is not False for s in scores) else 1


if __name__ == "__main__":
    sys.exit(main())

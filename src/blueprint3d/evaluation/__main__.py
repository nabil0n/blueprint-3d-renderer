import argparse
import sys
from pathlib import Path

from blueprint3d import evaluation
from blueprint3d.evaluation.report import evaluate_directory, format_comparison, format_table
from blueprint3d.parsing.learned.segmentation import DEFAULT_MODEL_DIR
from blueprint3d.parsing.registry import DEFAULT_PARSER, available_parsers, parser_catalogue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m blueprint3d.evaluation", description=evaluation.__doc__)
    parser.add_argument("data_dir", nargs="?", type=Path, default=Path("data"))
    parser.add_argument("--truth", type=Path, help="ground truth JSON (default: <data_dir>/truth.json)")
    parser.add_argument("--out", type=Path, default=Path("eval-out"), help="results go to <out>/<parser>/")
    names = [info.name for info in parser_catalogue()]
    parser.add_argument("--parser", choices=[*names, "all"], default=DEFAULT_PARSER)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="exported learned models")
    args = parser.parse_args(argv)

    if not args.data_dir.is_dir():
        parser.error(f"{args.data_dir} is not a directory")
    available = available_parsers(args.model_dir)
    if args.parser == "all":
        chosen = list(available)
    elif args.parser in available:
        chosen = [args.parser]
    else:
        info = next(i for i in parser_catalogue(args.model_dir) if i.name == args.parser)
        parser.error(f"parser {args.parser!r} is not available. {info.description}")

    truth = args.truth or args.data_dir / "truth.json"
    results = {
        name: evaluate_directory(args.data_dir, truth, args.out / name, available[name]) for name in chosen
    }
    sys.stdout.reconfigure(encoding="utf-8")
    for name, scores in results.items():
        print(f"\n== {name} ==\n{format_table(scores)}")
    if len(results) > 1:
        print(f"\n== comparison ==\n{format_comparison(results)}")
    print(f"\nOverlays and overview.png written to {args.out}/<parser>/")
    return 0 if all(s.passed is not False for scores in results.values() for s in scores) else 1


if __name__ == "__main__":
    sys.exit(main())

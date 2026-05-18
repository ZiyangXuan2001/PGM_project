"""List local controlled Diving48 experiment runs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "experiments" / "experiment_registry.csv"

COLUMNS = [
    "run_name",
    "ablation_id",
    "model_variant",
    "lambda_smooth",
    "use_alpha",
    "classifier",
    "best_val_top1",
    "test_top1",
    "run_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List local experiment registry rows.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--sort", choices=["best_val_top1"], default=None)
    parser.add_argument("--last", type=int, default=None)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"registry not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def print_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No experiments found.")
        return
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in COLUMNS
    }
    header = "  ".join(column.ljust(widths[column]) for column in COLUMNS)
    print(header)
    print("  ".join("-" * widths[column] for column in COLUMNS))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in COLUMNS))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.registry)
    if args.sort == "best_val_top1":
        rows = sorted(rows, key=lambda row: numeric(row.get("best_val_top1")), reverse=True)
    if args.last is not None:
        if args.last <= 0:
            raise SystemExit("--last must be positive")
        rows = rows[: args.last] if args.sort else rows[-args.last :]
    print_table(rows)


if __name__ == "__main__":
    main()

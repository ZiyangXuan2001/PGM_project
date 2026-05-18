"""Summarize controlled DiffTraj-PGM experiment-grid results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "runs" / "diving48_v2_grid" / "results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize controlled grid-search results.")
    parser.add_argument("--results_csv", type=Path, default=DEFAULT_RESULTS_CSV)
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"results CSV not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["lambda_smooth"] = float(row["lambda_smooth"])
            row["use_alpha"] = parse_bool(row["use_alpha"])
            row["best_val_acc"] = float(row["best_val_acc"])
            row["best_val_loss"] = float(row["best_val_loss"])
            row["final_train_acc"] = float(row["final_train_acc"])
            row["final_train_loss"] = float(row["final_train_loss"])
            row["seed"] = int(row["seed"])
            rows.append(row)
    return rows


def save_sorted_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["use_alpha"] = "true" if row["use_alpha"] else "false"
            writer.writerow(serializable)


def print_top_runs(rows: list[dict[str, Any]], top_k: int = 10) -> None:
    print(f"\nTop {min(top_k, len(rows))} runs by best_val_acc:")
    for rank, row in enumerate(rows[:top_k], start=1):
        print(
            f"{rank:02d}. acc={row['best_val_acc']:.4f} loss={row['best_val_loss']:.4f} "
            f"pgm={row['pgm_smoother']} lambda={row['lambda_smooth']} "
            f"classifier={row['classifier']} alpha={str(row['use_alpha']).lower()} "
            f"run={row['run_name']}"
        )


def group_average(rows: list[dict[str, Any]], key: str) -> list[tuple[Any, float, int]]:
    grouped: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row["best_val_acc"])
    return sorted((key_value, mean(values), len(values)) for key_value, values in grouped.items())


def print_group_averages(rows: list[dict[str, Any]]) -> None:
    for key in ["pgm_smoother", "classifier", "lambda_smooth", "use_alpha"]:
        print(f"\nAverage best_val_acc grouped by {key}:")
        for key_value, avg_acc, count in group_average(rows, key):
            if isinstance(key_value, bool):
                key_value = str(key_value).lower()
            print(f"  {key_value}: {avg_acc:.4f} over {count} run(s)")


def print_best_lambda_by_pgm_classifier(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, bool, float], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["pgm_smoother"], row["classifier"], row["use_alpha"], row["lambda_smooth"])
        ].append(row["best_val_acc"])

    candidates: dict[tuple[str, str, bool], list[tuple[float, float]]] = defaultdict(list)
    for (pgm_type, classifier, use_alpha, lambda_smooth), values in grouped.items():
        candidates[(pgm_type, classifier, use_alpha)].append((lambda_smooth, mean(values)))

    print("\nBest lambda for each PGM/classifier/alpha setting:")
    for (pgm_type, classifier, use_alpha), values in sorted(candidates.items()):
        best_lambda, best_acc = max(values, key=lambda item: item[1])
        print(
            f"  pgm={pgm_type} classifier={classifier} alpha={str(use_alpha).lower()}: "
            f"lambda={best_lambda} avg_best_val_acc={best_acc:.4f}"
        )


def maybe_plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("\nmatplotlib not installed; skipping plots.")
        return

    grouped: dict[tuple[str, str, bool], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["pgm_smoother"], row["classifier"], row["use_alpha"])][row["lambda_smooth"]].append(
            row["best_val_acc"]
        )

    plt.figure(figsize=(9, 5))
    for (pgm_type, classifier, use_alpha), lambda_map in sorted(grouped.items()):
        xs = sorted(lambda_map)
        ys = [mean(lambda_map[x]) for x in xs]
        label = f"{pgm_type}, {classifier}, alpha={str(use_alpha).lower()}"
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("lambda_smooth")
    plt.ylabel("best_val_acc")
    plt.title("Diving48 V2 Controlled Grid")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    output_path = output_dir / "lambda_vs_best_val_acc.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"\nSaved plot: {output_path}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.results_csv}")

    sorted_rows = sorted(rows, key=lambda row: (-row["best_val_acc"], row["best_val_loss"]))
    sorted_csv = args.results_csv.with_name("results_sorted.csv")
    save_sorted_csv(sorted_rows, sorted_csv)

    print(f"Loaded {len(rows)} result row(s) from {args.results_csv}")
    print(f"Saved sorted CSV: {sorted_csv}")
    print_top_runs(sorted_rows)
    print_group_averages(rows)
    print_best_lambda_by_pgm_classifier(rows)
    maybe_plot(rows, args.results_csv.parent)


if __name__ == "__main__":
    main()


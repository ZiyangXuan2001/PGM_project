"""Run a controlled grid over precomputed Diving48 embedding files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config  # noqa: E402
from utils.experiment_manager import ABLATION_INFO  # noqa: E402
from utils.training import append_csv_row, prepare_training_config, train_from_config  # noqa: E402


RESULT_FIELDS = [
    "run_name",
    "ablation_id",
    "model_variant",
    "pgm_smoother",
    "lambda_smooth",
    "classifier",
    "use_alpha",
    "best_val_acc",
    "best_val_loss",
    "final_train_acc",
    "final_train_loss",
    "checkpoint_path",
    "seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled DiffTraj-PGM grid.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment_grid.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--stage", choices=["debug", "full"], default="debug")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--overfit_samples", type=int, default=None)
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_existing_results(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["run_name"] for row in csv.DictReader(handle) if row.get("run_name")}


def bool_name(value: bool) -> str:
    return "true" if value else "false"


def value_name(value: Any) -> str:
    return str(value).replace("/", "_")


def make_grid_name(ablation_id: str, lambda_smooth: float, use_alpha: bool) -> str:
    variant = ABLATION_INFO[ablation_id]["variant"]
    return (
        f"{ablation_id}_{variant}"
        f"__lambda={value_name(lambda_smooth)}"
        f"__alpha={bool_name(use_alpha)}"
    )


def build_runs(config: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    grid_config = config["grid"]
    if stage == "debug":
        ablations = ["E0", "E3", "E4"]
        lambda_values = [float(config.get("pgm_smoother", {}).get("lambda_smooth", 1.0))]
        use_alpha_values = [bool(config.get("information_matrix", {}).get("use_alpha", True))]
    else:
        ablations = grid_config["ablations"]
        lambda_values = grid_config["lambda_values"]
        use_alpha_values = grid_config["use_alpha_values"]

    runs = []
    for ablation_id, lambda_smooth, use_alpha in product(
        ablations,
        lambda_values,
        use_alpha_values,
    ):
        if ablation_id not in ABLATION_INFO:
            raise ValueError(f"unknown ablation: {ablation_id}")
        variant = ABLATION_INFO[ablation_id]["variant"]
        pgm_type = "none" if ablation_id in {"E0", "E1"} else "gaussian_chain"
        classifier_type = "attention_pool" if ablation_id == "E4" else "mlp"
        alpha = bool(use_alpha) if ablation_id in {"E3", "E4"} else False
        runs.append(
            {
                "ablation_id": ablation_id,
                "model_variant": variant,
                "pgm_smoother": str(pgm_type),
                "lambda_smooth": float(lambda_smooth),
                "classifier": str(classifier_type),
                "use_alpha": alpha,
                "run_name": make_grid_name(ablation_id, float(lambda_smooth), alpha),
            }
        )
    return runs


def row_from_metrics(run: dict[str, Any], metrics: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "run_name": run["run_name"],
        "ablation_id": run["ablation_id"],
        "model_variant": run["model_variant"],
        "pgm_smoother": run["pgm_smoother"],
        "lambda_smooth": run["lambda_smooth"],
        "classifier": run["classifier"],
        "use_alpha": bool_name(run["use_alpha"]),
        "best_val_acc": metrics["best_val_acc"],
        "best_val_loss": metrics["best_val_loss"],
        "final_train_acc": metrics["final_train_acc"],
        "final_train_loss": metrics["final_train_loss"],
        "checkpoint_path": metrics["checkpoint_path"],
        "seed": seed,
    }


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_config_from_base(config: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    run_config = deepcopy(config)
    run_config.setdefault("model", {})["ablation_id"] = run["ablation_id"]
    run_config["model"]["variant"] = run["model_variant"]
    run_config.setdefault("pgm_smoother", {})["type"] = run["pgm_smoother"]
    run_config["pgm_smoother"]["lambda_smooth"] = run["lambda_smooth"]
    run_config.setdefault("classifier", {})["type"] = run["classifier"]
    run_config.setdefault("information_matrix", {})["use_alpha"] = run["use_alpha"]
    return run_config


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    runs = build_runs(config, args.stage)
    print(f"Stage: {args.stage}")
    print(f"Total runs: {len(runs)}")

    dataset_config = config["dataset"]
    output_config = config["output"]
    embedding_root = resolve_path(dataset_config["embedding_root"])
    output_root = resolve_path(output_config["root"])
    results_csv = resolve_path(output_config["results_csv"])
    output_root.mkdir(parents=True, exist_ok=True)

    train_file = embedding_root / dataset_config["train_file"]
    val_file = embedding_root / dataset_config["val_file"]
    test_file = embedding_root / dataset_config["test_file"]
    if not train_file.is_file() or not val_file.is_file() or not test_file.is_file():
        raise FileNotFoundError(
            "Missing embedding files:\n"
            f"  {train_file}\n  {val_file}\n  {test_file}"
        )

    existing_result_names = read_existing_results(results_csv)
    seed = int(config["training"].get("seed", 0))

    for index, run in enumerate(runs, start=1):
        run_dir = output_root / run["run_name"]
        metrics_path = run_dir / "metrics.json"

        print(f"\n[{index}/{len(runs)}] {run['run_name']}")
        if metrics_path.is_file():
            print("  metrics.json exists; skipping training")
            metrics = load_metrics(metrics_path)
            if run["run_name"] not in existing_result_names:
                append_csv_row(results_csv, row_from_metrics(run, metrics, seed), RESULT_FIELDS)
                existing_result_names.add(run["run_name"])
            continue

        run_config = run_config_from_base(config, run)
        run_config = prepare_training_config(
            base_config=run_config,
            train_file=train_file,
            val_file=val_file,
            run_dir=run_dir,
            ablation_id=run["ablation_id"],
            model_variant=run["model_variant"],
            epochs=args.epochs,
            device=args.device,
        )
        run_config.setdefault("dataset", {})["test_file"] = str(test_file)
        if args.overfit_samples is not None:
            run_config.setdefault("training", {})["overfit_samples"] = args.overfit_samples

        metrics = train_from_config(
            config=run_config,
            train_file=train_file,
            val_file=val_file,
            run_dir=run_dir,
            overfit_samples=args.overfit_samples,
        )
        append_csv_row(results_csv, row_from_metrics(run, metrics, seed), RESULT_FIELDS)
        existing_result_names.add(run["run_name"])

    print(f"\nGrid complete. Results: {results_csv}")


if __name__ == "__main__":
    main()

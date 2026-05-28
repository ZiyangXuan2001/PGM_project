"""Run the P-series lambda sweep with the direct attention classifier.

This keeps the original noPGM / prePGM / postPGM lambda design and changes
only the classifier head:

    R [B, T-1, d_r] -> trajectory_matrix_attention -> logits

No InformationMatrixAccumulator is used in these runs.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config  # noqa: E402
from utils.training import train_from_config  # noqa: E402


SWEEP_CONFIGS = [
    "P3_trajectory_matrix_linear_noPGM.yaml",
    "P4_trajectory_matrix_linear_pgm_lam005.yaml",
    "P5_trajectory_matrix_linear_pgm_lam010.yaml",
    "P8_trajectory_matrix_linear_pgm_lam020.yaml",
    "P9_trajectory_matrix_linear_pgm_lam030.yaml",
    "P10_trajectory_matrix_linear_pgm_lam040.yaml",
    "P14_trajectory_matrix_linear_pgm_lam050.yaml",
    "P15_trajectory_matrix_linear_pgm_lam060.yaml",
    "P16_trajectory_matrix_linear_pgm_lam070.yaml",
    "P17_trajectory_matrix_linear_pgm_lam080.yaml",
    "P18_trajectory_matrix_linear_pgm_lam090.yaml",
    "P19_trajectory_matrix_linear_pgm_lam100.yaml",
    "P6_prePGM_lam005_trajectory_matrix_linear.yaml",
    "P7_prePGM_lam010_trajectory_matrix_linear.yaml",
    "P11_prePGM_lam020_trajectory_matrix_linear.yaml",
    "P12_prePGM_lam030_trajectory_matrix_linear.yaml",
    "P13_prePGM_lam040_trajectory_matrix_linear.yaml",
    "P20_prePGM_lam050_trajectory_matrix_linear.yaml",
    "P21_prePGM_lam060_trajectory_matrix_linear.yaml",
    "P22_prePGM_lam070_trajectory_matrix_linear.yaml",
    "P23_prePGM_lam080_trajectory_matrix_linear.yaml",
    "P24_prePGM_lam090_trajectory_matrix_linear.yaml",
    "P25_prePGM_lam100_trajectory_matrix_linear.yaml",
]


SUMMARY_FIELDS = [
    "source_config",
    "ablation_id",
    "model_variant",
    "classifier_type",
    "lambda_smooth",
    "frame_pgm_type",
    "frame_lambda_smooth",
    "best_val_top1",
    "best_val_epoch",
    "best_val_loss",
    "epochs_trained",
    "final_train_top1",
    "final_val_top1",
    "run_dir",
    "checkpoint_best",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P-series attention-head lambda sweep.")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path(r"C:\data\diving48_embeddings\clip_vit_b16\train.pt"),
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path(r"C:\data\diving48_embeddings\clip_vit_b16\test.pt"),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "p_series_attention_sweep")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "p_series_attention_sweep_summary.csv",
    )
    parser.add_argument(
        "--configs",
        default="all",
        help="Use 'all' or a comma-separated subset of config stems/files.",
    )
    return parser.parse_args()


def selected_config_names(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return SWEEP_CONFIGS
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise SystemExit("--configs must be 'all' or a comma-separated list.")
    by_stem = {Path(name).stem: name for name in SWEEP_CONFIGS}
    by_name = {name: name for name in SWEEP_CONFIGS}
    selected: list[str] = []
    for item in requested:
        if item in by_name:
            selected.append(by_name[item])
        elif item in by_stem:
            selected.append(by_stem[item])
        else:
            valid = ", ".join(Path(name).stem for name in SWEEP_CONFIGS)
            raise SystemExit(f"Unknown config {item!r}. Valid stems: {valid}")
    return selected


def make_attention_config(source_config: dict[str, Any], source_name: str, args: argparse.Namespace) -> dict[str, Any]:
    config = deepcopy(source_config)
    config.setdefault("metadata", {})["source_config"] = source_name
    config["metadata"]["trajectory_head"] = "trajectory_matrix_attention"
    config.setdefault("classifier", {})["type"] = "trajectory_matrix_attention"
    config["classifier"].setdefault("hidden_dim", 256)
    config["classifier"].setdefault("num_heads", 4)
    config["classifier"].setdefault("dropout", 0.3)
    config.setdefault("training", {})["device"] = args.device
    if args.seed is not None:
        config["training"]["seed"] = int(args.seed)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.early_stop_patience is not None:
        config["training"]["early_stop_patience"] = int(args.early_stop_patience)
    config.setdefault("output", {})["root"] = str(args.output_root)
    config["notes"] = (
        f"Attention-head lambda sweep copied from {source_name}. "
        "Same PGM placement/lambda as the source config; classifier is direct "
        "trajectory_matrix_attention over DiffNet relation tokens R. "
        "No InformationMatrixAccumulator."
    )
    return config


def append_summary(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def main() -> None:
    args = parse_args()
    config_dir = PROJECT_ROOT / "configs" / "p_series"
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    config_names = selected_config_names(args.configs)
    print(f"Running {len(config_names)} P-series attention sweep configs.", flush=True)
    print(f"train_file={args.train_file}", flush=True)
    print(f"val_file={args.val_file}", flush=True)
    print(f"output_root={args.output_root}", flush=True)
    print(f"summary_path={args.summary_path}", flush=True)

    for index, config_name in enumerate(config_names, start=1):
        source_path = config_dir / config_name
        print(f"\n[{index}/{len(config_names)}] {config_name}", flush=True)
        start_time = time.perf_counter()
        source_config = load_config(str(source_path))
        config = make_attention_config(source_config, config_name, args)
        metrics = train_from_config(
            config=config,
            train_file=args.train_file,
            val_file=args.val_file,
            run_dir=None,
        )
        elapsed = time.perf_counter() - start_time
        row = {
            "source_config": config_name,
            "ablation_id": metrics.get("ablation_id"),
            "model_variant": metrics.get("model_variant"),
            "classifier_type": metrics.get("classifier_type"),
            "lambda_smooth": metrics.get("lambda_smooth"),
            "frame_pgm_type": metrics.get("frame_pgm_type"),
            "frame_lambda_smooth": metrics.get("frame_lambda_smooth"),
            "best_val_top1": metrics.get("best_val_top1"),
            "best_val_epoch": metrics.get("best_val_epoch"),
            "best_val_loss": metrics.get("best_val_loss"),
            "epochs_trained": metrics.get("epochs_trained"),
            "final_train_top1": metrics.get("final_train_top1"),
            "final_val_top1": metrics.get("final_val_top1"),
            "run_dir": config.get("output", {}).get("run_dir"),
            "checkpoint_best": metrics.get("checkpoint_best"),
        }
        append_summary(args.summary_path, row)
        print(
            f"finished {config_name}: best_val_top1={metrics['best_val_top1']:.4f} "
            f"best_epoch={metrics['best_val_epoch']} elapsed={elapsed:.1f}s",
            flush=True,
        )

    print("\nP-series attention sweep finished.", flush=True)


if __name__ == "__main__":
    main()

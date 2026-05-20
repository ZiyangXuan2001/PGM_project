"""Local experiment management for controlled Diving48 runs."""

from __future__ import annotations

import csv
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REGISTRY_PATH = EXPERIMENTS_DIR / "experiment_registry.csv"

REGISTRY_FIELDS = [
    "run_name",
    "date",
    "ablation_id",
    "model_variant",
    "backbone",
    "input_type",
    "num_frames",
    "pgm_smoother",
    "lambda_smooth",
    "information_matrix",
    "use_alpha",
    "classifier",
    "batch_size",
    "lr",
    "epochs",
    "best_val_top1",
    "best_val_epoch",
    "test_top1",
    "checkpoint_best",
    "run_dir",
    "notes",
]

TRAIN_LOG_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_top1",
    "val_top1",
    "lr",
    "epoch_time_sec",
]


ABLATION_INFO: dict[str, dict[str, str]] = {
    "E0": {
        "variant": "mean_pool_baseline",
        "purpose": "Basic frozen CLIP frame-embedding baseline with no explicit temporal modeling.",
        "architecture": (
            "X {input_shape}\n"
            "-> mean_pool over T\n"
            "-> MLP classifier\n"
            "-> logits [B, 48]"
        ),
    },
    "E1": {
        "variant": "diff_only",
        "purpose": "Test whether learned adjacent-frame difference embeddings help.",
        "architecture": (
            "X {input_shape}\n"
            "-> PairwiseDiffNet\n"
            "-> R [B, T-1, d_y]\n"
            "-> mean_pool over time\n"
            "-> MLP classifier\n"
            "-> logits [B, 48]"
        ),
    },
    "E2": {
        "variant": "diff_pgm",
        "purpose": "Test whether Gaussian-chain PGM smoothing improves temporal difference features.",
        "architecture": (
            "X {input_shape}\n"
            "-> PairwiseDiffNet\n"
            "-> R [B, T-1, d_y]\n"
            "-> GaussianPGMSmoother(lambda_smooth = {lambda_smooth})\n"
            "-> Y [B, T-1, d_y]\n"
            "-> mean_pool over time\n"
            "-> MLP classifier\n"
            "-> logits [B, 48]"
        ),
    },
    "E3": {
        "variant": "diff_pgm_info",
        "purpose": "Test whether the information matrix accumulation improves over pooled smoothed temporal features.",
        "architecture": (
            "X {input_shape}\n"
            "-> PairwiseDiffNet\n"
            "-> R [B, T-1, d_y]\n"
            "-> GaussianPGMSmoother(lambda_smooth = {lambda_smooth})\n"
            "-> Y [B, T-1, d_y]\n"
            "-> InformationMatrixAccumulator(K = {K}, d_h = {d_h}, use_alpha = {use_alpha})\n"
            "-> mean_pool over K\n"
            "-> MLP classifier\n"
            "-> logits [B, 48]"
        ),
    },
    "E4": {
        "variant": "diff_pgm_info_attention",
        "purpose": "Test whether attention pooling over the information matrix improves over simple mean pooling.",
        "architecture": (
            "X {input_shape}\n"
            "-> PairwiseDiffNet\n"
            "-> R [B, T-1, d_y]\n"
            "-> GaussianPGMSmoother(lambda_smooth = {lambda_smooth})\n"
            "-> Y [B, T-1, d_y]\n"
            "-> InformationMatrixAccumulator(K = {K}, d_h = {d_h}, use_alpha = {use_alpha})\n"
            "-> Attention pooling classifier head\n"
            "-> logits [B, 48]"
        ),
    },
}


@dataclass
class RunPaths:
    run_name: str
    run_dir: Path
    checkpoints_dir: Path
    config_path: Path
    metrics_path: Path
    train_log_path: Path
    model_summary_path: Path
    experiment_card_path: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path


def today_string() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def now_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def short_id() -> str:
    return uuid.uuid4().hex[:4]


def get_ablation_id(config: dict[str, Any]) -> str:
    model_config = config.get("model", {})
    variant = model_config.get("variant")
    if model_config.get("ablation_id"):
        return str(model_config["ablation_id"])
    for ablation_id, info in ABLATION_INFO.items():
        if info["variant"] == variant:
            return ablation_id
    raise ValueError(f"unknown model.variant: {variant!r}")


def get_model_variant(config: dict[str, Any]) -> str:
    ablation_id = get_ablation_id(config)
    return str(config.get("model", {}).get("variant", ABLATION_INFO[ablation_id]["variant"]))


def get_lambda_value(config: dict[str, Any]) -> str:
    ablation_id = get_ablation_id(config)
    if ablation_id in {"E0", "E1"}:
        return "none"
    return str(config.get("pgm_smoother", {}).get("lambda_smooth", "none"))


def get_classifier_label(config: dict[str, Any]) -> str:
    variant = get_model_variant(config)
    if variant == "diff_pgm_info_attention":
        return "attention"
    return "mlp"


def build_run_name(config: dict[str, Any]) -> str:
    ablation_id = get_ablation_id(config)
    variant = get_model_variant(config)
    lambda_value = get_lambda_value(config)
    use_alpha = bool(config.get("information_matrix", {}).get("use_alpha", False))
    if ablation_id in {"E0", "E1", "E2"}:
        use_alpha = False
    classifier = get_classifier_label(config)
    return (
        f"{today_string()}_{ablation_id}_{variant}_lambda{lambda_value}_"
        f"alpha{str(use_alpha).lower()}_{classifier}_{short_id()}"
    )


def create_run_dir(config: dict[str, Any]) -> RunPaths:
    """Create the run directory and checkpoints directory."""

    output_config = config.setdefault("output", {})
    explicit_run_dir = output_config.get("run_dir")
    if explicit_run_dir:
        run_dir = Path(explicit_run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        run_name = run_dir.name
    else:
        run_root = Path(output_config.get("root", DEFAULT_RUN_ROOT))
        if not run_root.is_absolute():
            run_root = PROJECT_ROOT / run_root
        run_name = output_config.get("run_name") or build_run_name(config)
        run_dir = run_root / run_name

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    config.setdefault("output", {})["run_name"] = run_name
    config["output"]["run_dir"] = str(run_dir)
    config["output"]["checkpoints_dir"] = str(checkpoints_dir)
    return RunPaths(
        run_name=run_name,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        config_path=run_dir / "config_resolved.yaml",
        metrics_path=run_dir / "metrics.json",
        train_log_path=run_dir / "train_log.csv",
        model_summary_path=run_dir / "model_summary.txt",
        experiment_card_path=run_dir / "experiment_card.md",
        best_checkpoint_path=checkpoints_dir / "best.pt",
        last_checkpoint_path=checkpoints_dir / "last.pt",
    )


def save_resolved_config(config: dict[str, Any], run_dir: Path) -> None:
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def architecture_text(config: dict[str, Any]) -> str:
    ablation_id = get_ablation_id(config)
    info = ABLATION_INFO[ablation_id]
    backbone_config = config.get("backbone", {})
    input_dim = backbone_config.get("input_dim", 512)
    if backbone_config.get("feature_format") == "spatial_map":
        input_shape = f"[B, T, {backbone_config.get('spatial_tokens', 49)}, {input_dim}]"
    else:
        input_shape = f"[B, T, {input_dim}]"
    return info["architecture"].format(
        input_shape=input_shape,
        lambda_smooth=config.get("pgm_smoother", {}).get("lambda_smooth", "none"),
        K=config.get("information_matrix", {}).get("K", 8),
        d_h=config.get("information_matrix", {}).get("d_h", 128),
        use_alpha=str(bool(config.get("information_matrix", {}).get("use_alpha", False))).lower(),
    )


def write_model_summary(config: dict[str, Any], run_dir: Path) -> None:
    ablation_id = get_ablation_id(config)
    variant = get_model_variant(config)
    dataset = config.get("dataset", {}).get("name", "diving48_v2")
    num_classes = config.get("dataset", {}).get("num_classes", 48)
    num_frames = config.get("backbone", {}).get("num_frames", 16)
    input_dim = config.get("backbone", {}).get("input_dim", 512)
    backbone = config.get("backbone", {}).get("name", "precomputed_clip_vit_b16")
    if config.get("backbone", {}).get("feature_format") == "spatial_map":
        input_shape = f"X [B, {num_frames}, {config.get('backbone', {}).get('spatial_tokens', 49)}, {input_dim}]"
    else:
        input_shape = f"X [B, {num_frames}, {input_dim}]"
    lines = [
        f"Run name: {config.get('output', {}).get('run_name', '')}",
        f"Ablation ID: {ablation_id}",
        f"Model variant: {variant}",
        f"Dataset: {dataset}",
        f"Input type: {backbone} frame embeddings",
        f"Input shape: {input_shape}",
        f"Number of classes: {num_classes}",
        "",
        "Architecture:",
        architecture_text(config),
        "",
        "Purpose:",
        ABLATION_INFO[ablation_id]["purpose"],
        "",
    ]
    (run_dir / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def init_train_log(run_dir: Path) -> None:
    path = run_dir / "train_log.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAIN_LOG_FIELDS)
        writer.writeheader()


def append_train_log(run_dir: Path, row: dict[str, Any]) -> None:
    path = run_dir / "train_log.csv"
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAIN_LOG_FIELDS)
        writer.writerow({field: row.get(field, "") for field in TRAIN_LOG_FIELDS})


def save_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def value_or_null(value: Any) -> str:
    return "null" if value is None else str(value)


def write_experiment_card(config: dict[str, Any], metrics: dict[str, Any], run_dir: Path) -> None:
    ablation_id = get_ablation_id(config)
    training = config.get("training", {})
    info = config.get("information_matrix", {})
    classifier = config.get("classifier", {})
    pgm = config.get("pgm_smoother", {})
    output = config.get("output", {})
    backbone = config.get("backbone", {}).get("name", "precomputed_clip_vit_b16")
    card = f"""# Experiment Card

## Run identity

- Run name: {metrics.get("run_name")}
- Date: {metrics.get("date")}
- Ablation ID: {ablation_id}
- Model variant: {get_model_variant(config)}
- Dataset: {config.get("dataset", {}).get("name", "diving48_v2")}
- Input type: {backbone} frame embeddings
- Number of frames: {config.get("backbone", {}).get("num_frames", 16)}
- Random seed: {training.get("seed", 0)}

## Architecture

Pipeline:
{architecture_text(config)}

## Main purpose

{ABLATION_INFO[ablation_id]["purpose"]}

## Key config

| Field | Value |
|---|---|
| model.variant | {get_model_variant(config)} |
| pgm_smoother.lambda_smooth | {pgm.get("lambda_smooth")} |
| information_matrix.use_alpha | {info.get("use_alpha")} |
| information_matrix.K | {info.get("K")} |
| information_matrix.d_h | {info.get("d_h")} |
| classifier.type | {classifier.get("type")} |
| training.lr | {training.get("lr")} |
| training.batch_size | {training.get("batch_size")} |
| training.epochs | {training.get("epochs")} |

## Results

| Metric | Value |
|---|---|
| best_val_top1 | {value_or_null(metrics.get("best_val_top1"))} |
| best_val_epoch | {value_or_null(metrics.get("best_val_epoch"))} |
| final_train_loss | {value_or_null(metrics.get("final_train_loss"))} |
| final_val_loss | {value_or_null(metrics.get("final_val_loss"))} |
| final_train_top1 | {value_or_null(metrics.get("final_train_top1"))} |
| final_val_top1 | {value_or_null(metrics.get("final_val_top1"))} |
| test_top1 | {value_or_null(metrics.get("test_top1"))} |

## Checkpoints

- best: {output.get("checkpoint_best", metrics.get("checkpoint_best"))}
- last: {output.get("checkpoint_last", metrics.get("checkpoint_last"))}

## Notes

- What worked:
- What failed:
- What to try next: {metrics.get("notes", "")}
"""
    (run_dir / "experiment_card.md").write_text(card, encoding="utf-8")


def append_registry(config: dict[str, Any], metrics: dict[str, Any], run_dir: Path) -> None:
    registry_path = Path(config.get("output", {}).get("registry_path", REGISTRY_PATH))
    if not registry_path.is_absolute():
        registry_path = PROJECT_ROOT / registry_path
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    exists = registry_path.is_file()
    ablation_id = get_ablation_id(config)
    info_config = config.get("information_matrix", {})
    info_enabled = bool(info_config.get("enabled", ablation_id in {"E3", "E4"}))
    row = {
        "run_name": metrics.get("run_name"),
        "date": metrics.get("date"),
        "ablation_id": metrics.get("ablation_id"),
        "model_variant": metrics.get("model_variant"),
        "backbone": config.get("backbone", {}).get("name", "precomputed_clip_vit_b16"),
        "input_type": f"{config.get('backbone', {}).get('name', 'precomputed_clip_vit_b16')}_frame_embeddings",
        "num_frames": config.get("backbone", {}).get("num_frames", 16),
        "pgm_smoother": config.get("pgm_smoother", {}).get("type"),
        "lambda_smooth": metrics.get("lambda_smooth"),
        "information_matrix": info_config.get("type", "accumulator") if info_enabled else "none",
        "use_alpha": metrics.get("use_alpha"),
        "classifier": metrics.get("classifier_type"),
        "batch_size": config.get("training", {}).get("batch_size"),
        "lr": config.get("training", {}).get("lr"),
        "epochs": config.get("training", {}).get("epochs"),
        "best_val_top1": metrics.get("best_val_top1"),
        "best_val_epoch": metrics.get("best_val_epoch"),
        "test_top1": metrics.get("test_top1"),
        "checkpoint_best": metrics.get("checkpoint_best"),
        "run_dir": str(run_dir),
        "notes": metrics.get("notes", ""),
    }
    with registry_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in REGISTRY_FIELDS})


def empty_registry() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        with REGISTRY_PATH.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS).writeheader()

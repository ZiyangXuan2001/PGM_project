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
    "val_correction_magnitude",
    "val_observation_residual",
    "val_smoothness_energy",
    "val_frame_pgm_correction_magnitude",
    "val_frame_pgm_observation_residual",
    "val_frame_pgm_smoothness_energy",
    "lr",
    "epoch_time_sec",
]


ABLATION_INFO: dict[str, dict[str, str]] = {
    "E0": {
        "variant": "feature_mean",
        "purpose": "Frozen feature mean baseline with no DiffNet, PGM, or accumulator.",
        "architecture": (
            "X {input_shape}\n"
            "-> mean_pool over T and spatial tokens if present\n"
            "-> MLP classifier\n"
            "-> logits [B, 48]"
        ),
    },
    "E1": {
        "variant": "diff_mean",
        "purpose": "Test whether learned adjacent-frame DiffNet observations help.",
        "architecture": (
            "X {input_shape}\n"
            "-> PairwiseDiffNet\n"
            "-> R [B, T-1, d_y]\n"
            "-> mean_pool over time\n"
            "-> MLP classifier\n"
            "-> logits [B, 48]"
        ),
    },
    "E1.5": {
        "variant": "diff_info_accum",
        "purpose": "Control for the InformationMatrixAccumulator without PGM inference.",
        "architecture": (
            "X {input_shape}\n"
            "-> PairwiseDiffNet\n"
            "-> R [B, T-1, d_y]\n"
            "-> InformationMatrixAccumulator(K = {K}, d_h = {d_h}, use_alpha = {use_alpha})\n"
            "-> mean_pool over K\n"
            "-> MLP classifier\n"
            "-> logits [B, 48]"
        ),
    },
    "E2": {
        "variant": "diff_pgm_mean",
        "purpose": "Test whether Gaussian PGM smoothing improves mean-pooled DiffNet observations.",
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
        "variant": "diff_pgm_info_accum",
        "purpose": "Test whether PGM MAP evidence improves over the no-PGM accumulator baseline.",
        "architecture": (
            "X {input_shape}\n"
            "-> PairwiseDiffNet\n"
            "-> R [B, T-1, d_y]\n"
            "-> GaussianPGMSmoother(lambda_smooth = {lambda_smooth})\n"
            "-> Y [B, T-1, d_y]\n"
            "-> local PGM evidence U [B, T-1, 3*d_y+2]\n"
            "-> InformationMatrixAccumulator(K = {K}, d_h = {d_h}, use_alpha = {use_alpha})\n"
            "-> mean_pool over K\n"
            "-> MLP classifier\n"
            "-> logits [B, 48]"
        ),
    },
    "P-noPGM": {
        "variant": "p_traj_no_pgm",
        "purpose": "Projected CLIP trajectory-matrix baseline with no frame PGM.",
        "architecture": "P-series trajectory-matrix baseline.",
    },
    "P-PGM": {
        "variant": "p_traj_pgm",
        "purpose": "Projected CLIP trajectory-matrix model with only fixed frame-level Gaussian PGM smoothing.",
        "architecture": "P-series trajectory-matrix model with frame PGM.",
    },
    "P-prePGM": {
        "variant": "p_traj_pre_pgm",
        "purpose": "Test fixed Gaussian PGM smoothing on raw CLIP frame features before the trajectory-matrix-linear path.",
        "architecture": "P-series trajectory-matrix model with pre-trajectory frame PGM.",
    },
}

LEGACY_VARIANT_ALIASES = {
    "mean_pool_baseline": "feature_mean",
    "diff_only": "diff_mean",
    "diff_pgm": "diff_pgm_mean",
    "diff_pgm_info": "diff_pgm_info_accum",
    "diff_pgm_info_attention": "diff_pgm_info_accum",
}

LEGACY_ABLATION_ALIASES = {"E4": "E3"}


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


def is_p_series_config(config: dict[str, Any]) -> bool:
    model_config = config.get("model", {})
    model_name = model_config.get("name", model_config.get("type"))
    return str(model_name) in {"p_series_trajectory_matrix", "p_series_trajectory"}


def p_series_uses_pgm(config: dict[str, Any]) -> bool:
    model_config = config.get("model", {})
    pgm_config = config.get("pgm", {})
    lambda_frame = float(pgm_config.get("lambda_frame", model_config.get("lambda_frame", 0.0)) or 0.0)
    return bool(model_config.get("use_pgm", False)) and lambda_frame > 0.0


def p_series_uses_pre_pgm(config: dict[str, Any]) -> bool:
    model_config = config.get("model", {})
    pre_pgm_config = config.get("pre_pgm", {})
    lambda_frame = float(pre_pgm_config.get("lambda_frame", model_config.get("pre_lambda_frame", 0.0)) or 0.0)
    return bool(model_config.get("use_pre_pgm", False)) and lambda_frame > 0.0


def get_ablation_id(config: dict[str, Any]) -> str:
    model_config = config.get("model", {})
    if is_p_series_config(config) and not model_config.get("ablation_id"):
        if p_series_uses_pre_pgm(config):
            return "P-prePGM"
        return "P-PGM" if p_series_uses_pgm(config) else "P-noPGM"
    variant = LEGACY_VARIANT_ALIASES.get(str(model_config.get("variant")), model_config.get("variant"))
    if model_config.get("ablation_id"):
        return LEGACY_ABLATION_ALIASES.get(str(model_config["ablation_id"]), str(model_config["ablation_id"]))
    for ablation_id, info in ABLATION_INFO.items():
        if info["variant"] == variant:
            return ablation_id
    raise ValueError(f"unknown model.variant: {variant!r}")


def get_model_variant(config: dict[str, Any]) -> str:
    ablation_id = get_ablation_id(config)
    variant = str(config.get("model", {}).get("variant", ABLATION_INFO[ablation_id]["variant"]))
    return LEGACY_VARIANT_ALIASES.get(variant, variant)


def get_lambda_value(config: dict[str, Any]) -> str:
    ablation_id = get_ablation_id(config)
    if is_p_series_config(config):
        if p_series_uses_pre_pgm(config):
            return f"preframefixed{config.get('pre_pgm', {}).get('lambda_frame', 'none')}"
        if not p_series_uses_pgm(config):
            return "none"
        return f"framefixed{config.get('pgm', {}).get('lambda_frame', 'none')}"
    frame_pgm = config.get("frame_pgm_smoother", {})
    frame_pgm_type = str(frame_pgm.get("type", "none"))
    if frame_pgm_type != "none":
        type_label = "learn" if frame_pgm_type == "learnable_gaussian_chain" else "fixed"
        return f"frame{type_label}{frame_pgm.get('lambda_smooth', 'none')}"
    if ablation_id in {"E0", "E1", "E1.5", "P-noPGM", "P-PGM"}:
        return "none"
    return str(config.get("pgm_smoother", {}).get("lambda_smooth", "none"))


def get_classifier_label(config: dict[str, Any]) -> str:
    return str(config.get("classifier", {}).get("type", "mlp"))


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
    diff_config = config.get("diff_nn", {})
    diff_net_type = diff_config.get("diff_net_type", diff_config.get("type", "pairwise_diff_net"))
    diff_net_label = "SimpleConcatPairwiseDiffNet" if diff_net_type == "simple_concat_pairwise" else "PairwiseDiffNet"
    input_dim = config.get("model", {}).get("clip_dim", backbone_config.get("input_dim", 512))
    if backbone_config.get("feature_format") == "spatial_map":
        input_shape = f"[B, T, {backbone_config.get('spatial_tokens', 49)}, {input_dim}]"
    else:
        input_shape = f"[B, T, {input_dim}]"
    if ablation_id in {"P-noPGM", "P-PGM", "P-prePGM"}:
        model_config = config.get("model", {})
        pgm_config = config.get("pgm", {})
        pre_pgm_config = config.get("pre_pgm", {})
        projection_dim = model_config.get("d_z", config.get("temporal_projection", {}).get("output_dim", 128))
        relation_dim = model_config.get("d_r", diff_config.get("d_y", diff_config.get("output_dim", 128)))
        num_frames = model_config.get("num_frames", backbone_config.get("num_frames", 16))
        num_pairs = int(num_frames) - 1
        use_pgm = p_series_uses_pgm(config)
        use_pre_pgm = p_series_uses_pre_pgm(config)
        lines = [
            f"X {input_shape}",
        ]
        if use_pre_pgm:
            lines.extend(
                [
                    (
                        "-> GaussianFramePGMSmoother"
                        f"(pre, lambda_frame = {pre_pgm_config.get('lambda_frame', 'none')})"
                    ),
                    f"-> X_smooth {input_shape}",
                ]
            )
        lines.extend(
            [
                "-> TemporalProjection",
                f"-> U [B, T, {projection_dim}]",
            ]
        )
        if use_pgm:
            lines.extend(
                [
                    (
                        "-> GaussianFramePGMSmoother"
                        f"(lambda_frame = {pgm_config.get('lambda_frame', 'none')})"
                    ),
                    f"-> Z [B, T, {projection_dim}]",
                ]
            )
        lines.extend(
            [
                "-> ProjectedPairwiseDiffNet",
                f"-> R [B, T-1, {relation_dim}]",
                f"-> flatten R [B, {num_pairs * int(relation_dim)}]",
                "-> TrajectoryMatrixClassifier",
                "-> logits [B, 48]",
            ]
        )
        return "\n".join(lines)
    architecture = info["architecture"].format(
        input_shape=input_shape,
        lambda_smooth=config.get("pgm_smoother", {}).get("lambda_smooth", "none"),
        K=config.get("information_matrix", {}).get("K", 8),
        d_h=config.get("information_matrix", {}).get("d_h", 128),
        use_alpha=str(bool(config.get("information_matrix", {}).get("use_alpha", False))).lower(),
    )
    frame_pgm = config.get("frame_pgm_smoother", {})
    if frame_pgm.get("type", "none") != "none":
        architecture = architecture.replace(
            f"X {input_shape}\n",
            (
                f"X {input_shape}\n"
                f"-> FrameGaussianPGMSmoother(lambda_smooth = {frame_pgm.get('lambda_smooth', 'none')})\n"
                f"-> Z {input_shape}\n"
            ),
            1,
        )
    return architecture.replace("PairwiseDiffNet", diff_net_label)


def write_model_summary(config: dict[str, Any], run_dir: Path) -> None:
    ablation_id = get_ablation_id(config)
    variant = get_model_variant(config)
    dataset = config.get("dataset", {}).get("name", "diving48_v2")
    num_classes = config.get("dataset", {}).get("num_classes", 48)
    num_frames = config.get("backbone", {}).get("num_frames", 16)
    input_dim = config.get("backbone", {}).get("input_dim", 512)
    backbone = config.get("backbone", {}).get("name", "precomputed_clip_vit_b16")
    diff_config = config.get("diff_nn", {})
    diff_net_type = diff_config.get("diff_net_type", diff_config.get("type", "pairwise_diff_net"))
    frame_pgm = config.get("frame_pgm_smoother", {})
    frame_pgm_type = frame_pgm.get("type", "none")
    frame_pgm_lambda = frame_pgm.get("lambda_smooth", "none")
    if is_p_series_config(config):
        model_config = config.get("model", {})
        if p_series_uses_pre_pgm(config):
            frame_pgm_type = "pre_gaussian_chain"
            frame_pgm_lambda = config.get("pre_pgm", {}).get("lambda_frame", "none")
        elif p_series_uses_pgm(config):
            frame_pgm_type = "post_projected_gaussian_chain"
            frame_pgm_lambda = config.get("pgm", {}).get("lambda_frame", "none")
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
        f"Frame PGM smoother: {frame_pgm_type}",
        f"Frame PGM lambda_smooth: {frame_pgm_lambda}",
        f"DiffNet type: {diff_net_type}",
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
    frame_pgm = config.get("frame_pgm_smoother", {})
    pre_pgm = config.get("pre_pgm", {})
    p_series_pgm = config.get("pgm", {})
    model_config = config.get("model", {})
    output = config.get("output", {})
    backbone = config.get("backbone", {}).get("name", "precomputed_clip_vit_b16")
    diff_config = config.get("diff_nn", {})
    diff_net_type = diff_config.get("diff_net_type", diff_config.get("type", "pairwise_diff_net"))
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

PGM interpretation:
If frame_pgm_smoother is enabled, frozen frame embeddings X_t are treated as noisy observations of latent clean frame states Z_t before DiffNet. If pgm_smoother is enabled after DiffNet, R_t is treated as noisy pairwise temporal evidence and smoothed into Y_t. The true Gaussian PGM information matrix is A = alpha I + lambda L, where L is the temporal path graph Laplacian. The learned InformationMatrixAccumulator is a learned sequential evidence accumulator, not the same object as A.

## Main purpose

{ABLATION_INFO[ablation_id]["purpose"]}

## Key config

| Field | Value |
|---|---|
| model.variant | {get_model_variant(config)} |
| diff_nn.diff_net_type | {diff_net_type} |
| diff_nn.hidden_dim | {diff_config.get("hidden_dim")} |
| diff_nn.d_y | {diff_config.get("d_y")} |
| diff_nn.dropout | {diff_config.get("dropout")} |
| model.use_pre_pgm | {model_config.get("use_pre_pgm")} |
| pre_pgm.lambda_frame | {pre_pgm.get("lambda_frame")} |
| model.use_pgm | {model_config.get("use_pgm")} |
| pgm.lambda_frame | {p_series_pgm.get("lambda_frame")} |
| frame_pgm_smoother.type | {frame_pgm.get("type", "none")} |
| frame_pgm_smoother.lambda_smooth | {frame_pgm.get("lambda_smooth", "none")} |
| pgm_smoother.lambda_smooth | {pgm.get("lambda_smooth")} |
| information_matrix.use_alpha | {info.get("use_alpha")} |
| information_matrix.K | {info.get("K")} |
| information_matrix.d_h | {info.get("d_h")} |
| classifier.type | {classifier.get("type")} |
| training.lr | {training.get("lr")} |
| training.batch_size | {training.get("batch_size")} |
| training.epochs | {training.get("epochs")} |
| training.early_stop_patience | {training.get("early_stop_patience")} |

## Results

| Metric | Value |
|---|---|
| best_val_top1 | {value_or_null(metrics.get("best_val_top1"))} |
| best_val_epoch | {value_or_null(metrics.get("best_val_epoch"))} |
| epochs_trained | {value_or_null(metrics.get("epochs_trained"))} |
| early_stopped | {value_or_null(metrics.get("early_stopped"))} |
| stop_reason | {value_or_null(metrics.get("stop_reason"))} |
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
    info_enabled = bool(info_config.get("enabled", ablation_id in {"E1.5", "E3"}))
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

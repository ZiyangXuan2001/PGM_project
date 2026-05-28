"""Smoke tests for the simplified P-series trajectory-matrix model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import GaussianFramePGMSmoother, PSeriesTrajectoryMatrixModel  # noqa: E402
from utils.config import load_config  # noqa: E402


def _p_series_index(path: Path) -> int:
    prefix = path.stem.split("_", 1)[0]
    return int(prefix[1:])


def discover_configs() -> dict[str, Path]:
    config_dir = PROJECT_ROOT / "configs" / "p_series"
    configs: dict[str, Path] = {}
    candidates = [
        config_path
        for config_path in config_dir.glob("*.yaml")
        if config_path.stem.startswith("P") and len(config_path.stem) > 1 and config_path.stem[1].isdigit()
    ]
    for config_path in sorted(candidates, key=_p_series_index):
        if (
            "trajectory_matrix_linear" not in config_path.name
            and "trajectory_matrix_attention" not in config_path.name
        ):
            continue
        configs[config_path.stem] = config_path
    return configs


CONFIGS = discover_configs()

LOCAL_TRAIN_FILE = r"C:\data\diving48_embeddings\clip_vit_b16\train.pt"
LOCAL_VAL_FILE = r"C:\data\diving48_embeddings\clip_vit_b16\test.pt"


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise AssertionError(f"{name} shape mismatch: expected {expected}, got {actual}")
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf values")


def build_model(config_path: Path) -> PSeriesTrajectoryMatrixModel:
    return PSeriesTrajectoryMatrixModel.from_config(load_config(str(config_path)))


def check_forward(
    model: PSeriesTrajectoryMatrixModel,
    X: torch.Tensor,
    label: str,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        debug = model(X, return_debug=True)
    assert_shape(debug["X_smooth"], (2, 16, 512), f"{label} X_smooth")
    assert_shape(debug["U"], (2, 16, 128), f"{label} U")
    assert_shape(debug["Z"], (2, 16, 128), f"{label} Z")
    assert_shape(debug["R"], (2, 15, 128), f"{label} R")
    assert_shape(debug["logits"], (2, 48), f"{label} logits")
    return debug


def print_training_commands() -> None:
    python_executable = sys.executable
    print("\nManual training commands:")
    for label, config_path in CONFIGS.items():
        command = (
            f"{python_executable} -u scripts/train_embeddings.py "
            f"--config {config_path.relative_to(PROJECT_ROOT)} "
            f"--train_file {LOCAL_TRAIN_FILE} "
            f"--val_file {LOCAL_VAL_FILE} "
            "--device auto"
        )
        print(f"  {label}:")
        print(f"    {command}")

    print("\nRecommended tmux command:")
    print("  tmux new -s p_series")
    print("  bash scripts/run_p_series.sh")

    print("\nRun artifacts:")
    print("  outputs/runs/<run_name>/config_resolved.yaml")
    print("  outputs/runs/<run_name>/train_log.csv")
    print("  outputs/runs/<run_name>/metrics.json")
    print("  outputs/runs/<run_name>/model_summary.txt")
    print("  outputs/runs/<run_name>/experiment_card.md")
    print("  outputs/runs/<run_name>/checkpoints/best.pt")
    print("  outputs/runs/<run_name>/checkpoints/last.pt")
    print("  experiments/experiment_registry.csv")


def main() -> None:
    torch.manual_seed(20260527)
    X = torch.randn(2, 16, 512)

    models_by_label = {
        label: build_model(config_path)
        for label, config_path in CONFIGS.items()
    }
    debug_by_label = {
        label: check_forward(model, X, label)
        for label, model in models_by_label.items()
    }
    baseline_label = next(label for label in CONFIGS if label.startswith("P3_"))
    no_pgm = models_by_label[baseline_label]
    debug_no_pgm = debug_by_label[baseline_label]

    identity_smoother = GaussianFramePGMSmoother(use_pgm=True, alpha=1.0, lambda_frame=0.0)
    identity_Z = identity_smoother(debug_no_pgm["U"])
    identity_max_error = (identity_Z - debug_no_pgm["U"]).abs().max().item()
    if identity_max_error >= 1e-6:
        raise AssertionError(f"lambda_frame=0 should return U unchanged; max error={identity_max_error:.3e}")

    parameter_counts = {
        label: trainable_parameter_count(model)
        for label, model in models_by_label.items()
    }
    parameter_counts_by_head: dict[str, set[int]] = {}
    for label, model in models_by_label.items():
        parameter_counts_by_head.setdefault(model.classifier_type, set()).add(parameter_counts[label])
    mismatched = {
        head: counts
        for head, counts in parameter_counts_by_head.items()
        if len(counts) != 1
    }
    if mismatched:
        raise AssertionError(
            "P-series trainable parameter counts should match within each classifier head: "
            f"{mismatched}"
        )

    for label, model in models_by_label.items():
        pgm_params = trainable_parameter_count(model.frame_smoother)
        pre_pgm_params = trainable_parameter_count(model.pre_frame_smoother)
        if pgm_params != 0 or pre_pgm_params != 0:
            raise AssertionError(
                f"fixed PGM smoothers should add 0 trainable params for {label}; "
                f"post={pgm_params}, pre={pre_pgm_params}"
            )

    print("P-series trajectory-matrix smoke test passed.")
    for label, debug in debug_by_label.items():
        print(f"\n{label}")
        print(f"  X_smooth: {tuple(debug['X_smooth'].shape)}")
        print(f"  U: {tuple(debug['U'].shape)}")
        print(f"  Z: {tuple(debug['Z'].shape)}")
        print(f"  R: {tuple(debug['R'].shape)}")
        print(f"  logits: {tuple(debug['logits'].shape)}")
        print(f"  classifier type: {models_by_label[label].classifier_type}")
        print(f"  total trainable params: {trainable_parameter_count(models_by_label[label]):,}")
        print(f"  classifier trainable params: {trainable_parameter_count(models_by_label[label].classifier):,}")
        print(f"  post-PGM trainable params: {trainable_parameter_count(models_by_label[label].frame_smoother):,}")
        print(f"  pre-PGM trainable params: {trainable_parameter_count(models_by_label[label].pre_frame_smoother):,}")
        print(f"  selected lambda_frame: {float(debug['lambda_frame']):.4f}")
        print(f"  pre_lambda_frame: {float(debug['pre_lambda_frame']):.4f}")
        print(f"  post_lambda_frame: {float(debug['post_lambda_frame']):.4f}")

    print(f"\nlambda_frame=0 identity max error: {identity_max_error:.3e}")
    print("P-series parameter counts match within each classifier head:")
    for head, counts in sorted(parameter_counts_by_head.items()):
        only_count = next(iter(counts))
        print(f"  {head}: {only_count:,}")
    print("\nNo training was started.")
    print_training_commands()


if __name__ == "__main__":
    main()

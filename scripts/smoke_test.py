"""Smoke tests for E0-E4 controlled Diving48 DiffTraj-PGM variants."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import EmbeddingDifferencePGMModel  # noqa: E402
from utils import assert_shape, check_finite, load_config  # noqa: E402
from utils.experiment_manager import ABLATION_INFO  # noqa: E402


VARIANTS = [
    ("E0", "mean_pool_baseline"),
    ("E1", "diff_only"),
    ("E2", "diff_pgm"),
    ("E3", "diff_pgm_info"),
    ("E4", "diff_pgm_info_attention"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E0-E4 controlled model smoke tests.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML config.",
    )
    return parser.parse_args()


def config_for_variant(base_config: dict[str, Any], ablation_id: str, variant: str) -> dict[str, Any]:
    config = deepcopy(base_config)
    config.setdefault("model", {})["ablation_id"] = ablation_id
    config["model"]["variant"] = variant
    if ablation_id in {"E0", "E1"}:
        config.setdefault("pgm_smoother", {})["type"] = "none"
    else:
        config.setdefault("pgm_smoother", {})["type"] = "gaussian_chain"
    config.setdefault("classifier", {})["type"] = "attention_pool" if ablation_id == "E4" else "mlp"
    return config


def tensor_shape(value: torch.Tensor | None) -> tuple[int, ...] | None:
    return None if value is None else tuple(value.shape)


def check_tensor(value: torch.Tensor | None, name: str) -> None:
    if value is not None:
        assert_shape(value, name=name)
        check_finite(value, name=name)


def check_variant_shapes(debug: dict[str, Any], config: dict[str, Any], batch_size: int, num_frames: int) -> None:
    ablation_id = config["model"]["ablation_id"]
    d_y = int(config["diff_nn"]["d_y"])
    K = int(config["information_matrix"]["K"])
    d_h = int(config["information_matrix"]["d_h"])
    num_classes = int(config["dataset"]["num_classes"])
    length = num_frames - 1

    assert debug["logits"].shape == (batch_size, num_classes)
    if ablation_id == "E0":
        assert debug["R"] is None
        assert debug["Y"] is None
        assert debug["H_final"] is None
        assert debug["pooled"].shape == (batch_size, int(config["backbone"]["input_dim"]))
    elif ablation_id in {"E1", "E2"}:
        assert debug["R"].shape == (batch_size, length, d_y)
        assert debug["Y"].shape == (batch_size, length, d_y)
        assert debug["H_final"] is None
        assert debug["pooled"].shape == (batch_size, d_y)
    else:
        assert debug["R"].shape == (batch_size, length, d_y)
        assert debug["Y"].shape == (batch_size, length, d_y)
        assert debug["H_final"].shape == (batch_size, K, d_h)

    for key in ["logits", "R", "Y", "H_final", "alpha", "pooled"]:
        check_tensor(debug.get(key), key)
    check_tensor(debug.get("lambda_smooth"), "lambda_smooth")


def run_variant(base_config: dict[str, Any], ablation_id: str, variant: str) -> None:
    config = config_for_variant(base_config, ablation_id, variant)
    batch_size = 2
    num_frames = int(config["backbone"].get("num_frames", 16))
    input_dim = int(config["backbone"].get("input_dim", 512))
    num_classes = int(config["dataset"]["num_classes"])

    torch.manual_seed(1000 + int(ablation_id[1:]))
    X = torch.randn(batch_size, num_frames, input_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    model = EmbeddingDifferencePGMModel.from_config(config)
    debug = model(X, return_debug=True)

    print(f"\n{ablation_id}: {variant}")
    print(f"  purpose: {ABLATION_INFO[ablation_id]['purpose']}")
    print(f"  R: {tensor_shape(debug.get('R'))}")
    print(f"  Y: {tensor_shape(debug.get('Y'))}")
    print(f"  H_final: {tensor_shape(debug.get('H_final'))}")
    print(f"  logits: {tensor_shape(debug.get('logits'))}")
    check_variant_shapes(debug, config, batch_size=batch_size, num_frames=num_frames)

    loss = F.cross_entropy(debug["logits"], labels)
    loss.backward()
    print("  backward ok")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    for ablation_id, variant in VARIANTS:
        run_variant(config, ablation_id, variant)
    print("\nAll E0-E4 smoke tests finished.")


if __name__ == "__main__":
    main()


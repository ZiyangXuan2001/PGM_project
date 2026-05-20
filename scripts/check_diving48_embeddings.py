"""Validate saved Diving48 V2 CLIP embedding tensors."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMB_DIR = PROJECT_ROOT / "data" / "diving48_v2" / "embeddings" / "clip_vit_b16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Diving48 V2 embedding files.")
    parser.add_argument("--emb_dir", type=Path, default=DEFAULT_EMB_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--expected_T", type=int, default=16)
    parser.add_argument("--expected_K", type=int, default=None)
    parser.add_argument("--expected_D", type=int, default=512)
    parser.add_argument("--expected_num_classes", type=int, default=48)
    return parser.parse_args()


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def compact_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record.get("sample_id"),
        "split": record.get("split"),
        "vid_name": record.get("vid_name"),
        "label": record.get("label"),
        "label_name": record.get("label_name"),
        "T": record.get("T"),
        "source_type": record.get("source_type"),
        "source_frame_count": record.get("source_frame_count"),
    }


def check_split(
    emb_dir: Path,
    split: str,
    expected_T: int,
    expected_K: int | None,
    expected_D: int,
    expected_num_classes: int,
) -> int:
    errors: list[str] = []
    path = emb_dir / f"{split}.pt"
    if not path.is_file():
        print(f"\n{split}: missing {path}")
        return 1

    payload = torch_load(path)
    X = payload.get("X")
    labels = payload.get("labels")
    label_names = payload.get("label_names", [])
    metadata = payload.get("metadata", [])
    backbone_name = payload.get("backbone_name")
    feature_format = payload.get("feature_format")

    print(f"\n{split}: {path}")
    if not isinstance(X, torch.Tensor):
        errors.append("X is missing or is not a tensor")
    if not isinstance(labels, torch.Tensor):
        errors.append("labels is missing or is not a tensor")
    if not isinstance(label_names, list):
        errors.append("label_names is missing or is not a list")
        label_names = []
    if not isinstance(metadata, list):
        errors.append("metadata is missing or is not a list")
        metadata = []

    if errors:
        for error in errors:
            print(f"  - {error}")
        return len(errors)

    assert isinstance(X, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    print(f"  X shape: {tuple(X.shape)}")
    print(f"  labels shape: {tuple(labels.shape)}")
    if backbone_name:
        print(f"  backbone_name: {backbone_name}")
    if feature_format:
        print(f"  feature_format: {feature_format}")

    if X.ndim == 3:
        if X.shape[1:] != (expected_T, expected_D):
            errors.append(f"expected X shape [N, {expected_T}, {expected_D}], got {tuple(X.shape)}")
    elif X.ndim == 4:
        expected_K = expected_K if expected_K is not None else X.shape[2]
        if X.shape[1:] != (expected_T, expected_K, expected_D):
            errors.append(
                f"expected X shape [N, {expected_T}, {expected_K}, {expected_D}], got {tuple(X.shape)}"
            )
    else:
        errors.append(f"X must have shape [N, T, D] or [N, T, K, D], got {tuple(X.shape)}")

    if labels.ndim != 1:
        errors.append(f"labels must have shape [N], got {tuple(labels.shape)}")
    elif X.ndim in {3, 4} and labels.shape[0] != X.shape[0]:
        errors.append(f"labels length {labels.shape[0]} does not match X N={X.shape[0]}")

    if len(label_names) != expected_num_classes:
        errors.append(f"expected {expected_num_classes} label names, got {len(label_names)}")

    if X.ndim in {3, 4} and len(metadata) != X.shape[0]:
        errors.append(f"metadata length {len(metadata)} does not match X N={X.shape[0]}")

    if not torch.isfinite(X).all():
        errors.append("X contains NaN or Inf")

    if labels.numel() > 0:
        min_label = int(labels.min().item())
        max_label = int(labels.max().item())
        if min_label < 0 or max_label >= expected_num_classes:
            errors.append(f"label range [{min_label}, {max_label}] is outside [0, {expected_num_classes - 1}]")

    X_float = X.float()
    print(f"  embedding mean: {X_float.mean().item():.6f}")
    print(f"  embedding std: {X_float.std().item():.6f}")

    label_counter = Counter(int(label) for label in labels.tolist())
    print(f"  observed classes: {len(label_counter)} / {expected_num_classes}")
    print("  first label counts:")
    for label, count in sorted(label_counter.items())[:10]:
        name = label_names[label] if label < len(label_names) else str(label)
        print(f"    {label} {name}: {count}")

    print("  sample metadata:")
    for record in metadata[:2]:
        print(json.dumps(compact_metadata(record), indent=4, ensure_ascii=False)[:800])

    if errors:
        print(f"  errors: {len(errors)}")
        for error in errors[:20]:
            print(f"    - {error}")
    else:
        print("  checks passed")
    return len(errors)


def main() -> None:
    args = parse_args()
    total_errors = 0
    for split in args.splits:
        total_errors += check_split(
            emb_dir=args.emb_dir,
            split=split,
            expected_T=args.expected_T,
            expected_K=args.expected_K,
            expected_D=args.expected_D,
            expected_num_classes=args.expected_num_classes,
        )

    if total_errors:
        raise SystemExit(f"\nEmbedding check failed with {total_errors} error(s).")
    print("\nDiving48 V2 embedding checks passed.")


if __name__ == "__main__":
    main()

"""Validate saved warm-up motion CLIP embedding tensors."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMB_DIR = PROJECT_ROOT / "data" / "warmup_motion_manim" / "embeddings"
SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check warm-up motion embedding files.")
    parser.add_argument("--emb_dir", type=Path, default=DEFAULT_EMB_DIR)
    parser.add_argument("--expected_T", type=int, default=16)
    parser.add_argument("--expected_D", type=int, default=512)
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
        "label": record.get("label"),
        "label_name": record.get("label_name"),
        "T": record.get("T"),
        "first_frame": (record.get("frame_paths") or [None])[0],
    }


def check_split(emb_dir: Path, split: str, expected_T: int, expected_D: int) -> int:
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

    print(f"\n{split}: {path}")
    if not isinstance(X, torch.Tensor):
        errors.append("X is missing or is not a tensor")
    if not isinstance(labels, torch.Tensor):
        errors.append("labels is missing or is not a tensor")
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

    if X.ndim != 3:
        errors.append(f"X must have shape [N, T, D], got {tuple(X.shape)}")
    elif X.shape[1:] != (expected_T, expected_D):
        errors.append(f"expected X shape [N, {expected_T}, {expected_D}], got {tuple(X.shape)}")

    if labels.ndim != 1:
        errors.append(f"labels must have shape [N], got {tuple(labels.shape)}")
    elif X.ndim == 3 and labels.shape[0] != X.shape[0]:
        errors.append(f"labels length {labels.shape[0]} does not match X N={X.shape[0]}")

    if X.ndim == 3 and len(metadata) != X.shape[0]:
        errors.append(f"metadata length {len(metadata)} does not match X N={X.shape[0]}")

    if not torch.isfinite(X).all():
        errors.append("X contains NaN or Inf")

    X_float = X.float()
    print(f"  embedding mean: {X_float.mean().item():.6f}")
    print(f"  embedding std: {X_float.std().item():.6f}")

    label_counter = Counter(int(label) for label in labels.tolist())
    print("  label distribution:")
    for label, count in sorted(label_counter.items()):
        name = label_names[label] if isinstance(label_names, list) and label < len(label_names) else str(label)
        print(f"    {label} {name}: {count}")

    print("  sample metadata:")
    for record in metadata[:2]:
        print(json.dumps(compact_metadata(record), indent=4)[:800])

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
    for split in SPLITS:
        total_errors += check_split(
            emb_dir=args.emb_dir,
            split=split,
            expected_T=args.expected_T,
            expected_D=args.expected_D,
        )

    if total_errors:
        raise SystemExit(f"\nEmbedding check failed with {total_errors} error(s).")
    print("\nWarmup embedding checks passed.")


if __name__ == "__main__":
    main()

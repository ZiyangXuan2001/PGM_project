"""Validate the raw warm-up motion dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "warmup_motion_manim" / "raw"
LABEL_NAMES = [
    "clockwise",
    "counter_clockwise",
    "horizontal_oscillation",
    "vertical_oscillation",
    "stationary",
]
SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check warm-up motion raw dataset.")
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of metadata records")
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError(f"{path} contains a non-object metadata entry")
    return data


def compact_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record.get("sample_id"),
        "split": record.get("split"),
        "label": record.get("label"),
        "label_name": record.get("label_name"),
        "T": record.get("T"),
        "first_frame": (record.get("frame_paths") or [None])[0],
        "params": record.get("params"),
    }


def check_split(raw_dir: Path, split: str) -> int:
    errors: list[str] = []
    split_dir = raw_dir / split
    metadata_path = raw_dir / f"metadata_{split}.json"

    if not split_dir.is_dir():
        errors.append(f"missing split directory: {split_dir}")
    if not metadata_path.is_file():
        errors.append(f"missing metadata file: {metadata_path}")
        print(f"\n{split}: unable to continue without metadata.")
        return len(errors)

    try:
        records = load_metadata(metadata_path)
    except Exception as exc:  # noqa: BLE001
        print(f"\n{split}: failed to load metadata: {exc}")
        return len(errors) + 1

    label_counts: Counter[int] = Counter()
    image_size: tuple[int, int] | None = None

    for idx, record in enumerate(records):
        sample_id = record.get("sample_id")
        label = record.get("label")
        label_name = record.get("label_name")
        T = record.get("T")
        frame_paths = record.get("frame_paths")

        if label not in range(len(LABEL_NAMES)):
            errors.append(f"{split}[{idx}] invalid label: {label!r}")
        else:
            label_counts[int(label)] += 1
            expected_name = LABEL_NAMES[int(label)]
            if label_name != expected_name:
                errors.append(
                    f"{split}[{idx}] label_name {label_name!r} does not match {expected_name!r}"
                )

        if not isinstance(sample_id, str):
            errors.append(f"{split}[{idx}] missing string sample_id")
            sample_id = f"unknown_{idx}"

        if not isinstance(T, int) or T <= 0:
            errors.append(f"{split}[{idx}] invalid T: {T!r}")
            T = 0

        if not isinstance(frame_paths, list):
            errors.append(f"{split}[{idx}] frame_paths must be a list")
            frame_paths = []

        if len(frame_paths) != T:
            errors.append(
                f"{split}/{sample_id} expected {T} frame paths, got {len(frame_paths)}"
            )

        sample_dir = split_dir / str(sample_id)
        if sample_dir.exists():
            frame_files = sorted(sample_dir.glob("frame_*.png"))
            if len(frame_files) != T:
                errors.append(
                    f"{split}/{sample_id} expected {T} PNG files, got {len(frame_files)}"
                )
        else:
            errors.append(f"missing sample directory: {sample_dir}")

        for frame_rel in frame_paths:
            frame_path = raw_dir / frame_rel
            if not frame_path.is_file():
                errors.append(f"missing frame: {frame_path}")
                continue
            try:
                with Image.open(frame_path) as image:
                    size = image.size
            except Exception as exc:  # noqa: BLE001
                errors.append(f"failed to open {frame_path}: {exc}")
                continue

            if image_size is None:
                image_size = size
            elif size != image_size:
                errors.append(f"{frame_path} has size {size}, expected {image_size}")

    print(f"\n{split}: {len(records)} samples")
    print("  label distribution:")
    for label, label_name in enumerate(LABEL_NAMES):
        print(f"    {label} {label_name}: {label_counts[label]}")
    print(f"  image size: {image_size}")
    print("  sample metadata:")
    for record in records[:2]:
        print(json.dumps(compact_metadata(record), indent=4)[:1200])

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
        total_errors += check_split(args.raw_dir, split)

    if total_errors:
        raise SystemExit(f"\nDataset check failed with {total_errors} error(s).")
    print("\nRaw warm-up motion dataset checks passed.")


if __name__ == "__main__":
    main()

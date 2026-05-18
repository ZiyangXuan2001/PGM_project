"""Validate a local Diving48 V2 dataset layout without downloading data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "diving48_v2"
SPLITS = {
    "train": "Diving48_V2_train.json",
    "test": "Diving48_V2_test.json",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".mov", ".webm"]


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local Diving48 V2 files.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--annotation_dir", type=Path, default=None)
    parser.add_argument("--video_dir", type=Path, default=None)
    parser.add_argument("--rawframes_dir", type=Path, default=None)
    parser.add_argument("--input_format", choices=["auto", "videos", "rawframes"], default="auto")
    parser.add_argument("--train_annotation", default=SPLITS["train"])
    parser.add_argument("--test_annotation", default=SPLITS["test"])
    parser.add_argument("--max_source_checks", type=int, default=100)
    parser.add_argument("--decode_one", type=str_to_bool, default=False)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of annotation records")
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"{path}[{idx}] must be an object")
    return data


def video_path_for(record: dict[str, Any], video_dir: Path) -> Path | None:
    vid_name = str(record.get("vid_name", ""))
    direct = video_dir / vid_name
    if direct.is_file():
        return direct
    for extension in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{vid_name}{extension}"
        if candidate.is_file():
            return candidate
    return None


def rawframe_paths_for(record: dict[str, Any], rawframes_dir: Path) -> list[Path]:
    vid_name = str(record.get("vid_name", ""))
    frame_dir = rawframes_dir / vid_name
    if not frame_dir.is_dir():
        return []
    return sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_source(
    record: dict[str, Any],
    input_format: str,
    video_dir: Path,
    rawframes_dir: Path,
) -> tuple[str, Path | None, list[Path]]:
    if input_format in {"auto", "rawframes"}:
        frame_paths = rawframe_paths_for(record, rawframes_dir)
        if frame_paths:
            return "rawframes", rawframes_dir / str(record.get("vid_name", "")), frame_paths
        if input_format == "rawframes":
            return "rawframes", None, []

    video_path = video_path_for(record, video_dir)
    if video_path is not None:
        return "videos", video_path, []
    return "videos", None, []


def decode_one_source(source_type: str, source_path: Path, frame_paths: list[Path]) -> str:
    if source_type == "rawframes":
        if not frame_paths:
            return "no frames"
        with Image.open(frame_paths[0]) as image:
            return f"opened {frame_paths[0].name} size={image.size}"

    try:
        import cv2  # type: ignore
    except ImportError:
        return "opencv-python is not installed; skipped video decode"

    capture = cv2.VideoCapture(str(source_path))
    try:
        if not capture.isOpened():
            return "OpenCV could not open video"
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = capture.read()
        if not ok or frame is None:
            return f"OpenCV opened video but could not read frame_count={frame_count}"
        return f"opened video frame_count={frame_count} first_frame={frame.shape}"
    finally:
        capture.release()


def validate_records(split: str, records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for idx, record in enumerate(records):
        vid_name = record.get("vid_name")
        label = record.get("label")
        if not isinstance(vid_name, str) or not vid_name:
            errors.append(f"{split}[{idx}] invalid vid_name: {vid_name!r}")
        if not isinstance(label, int) or not 0 <= label < 48:
            errors.append(f"{split}[{idx}] invalid label: {label!r}")
    return errors


def check_split(
    split: str,
    annotation_path: Path,
    input_format: str,
    video_dir: Path,
    rawframes_dir: Path,
    max_source_checks: int,
    decode_one: bool,
) -> int:
    print(f"\n{split}: {annotation_path}")
    records = load_records(annotation_path)
    errors = validate_records(split, records)
    label_counts = Counter(int(record["label"]) for record in records if isinstance(record.get("label"), int))

    print(f"  records: {len(records)}")
    print(f"  labels: {len(label_counts)} / 48")
    print(f"  first labels: {dict(sorted(label_counts.items())[:5])}")

    missing_sources: list[str] = []
    first_source: tuple[str, Path, list[Path]] | None = None
    for record in records[:max_source_checks]:
        source_type, source_path, frame_paths = resolve_source(
            record=record,
            input_format=input_format,
            video_dir=video_dir,
            rawframes_dir=rawframes_dir,
        )
        if source_path is None:
            missing_sources.append(str(record.get("vid_name")))
        elif first_source is None:
            first_source = (source_type, source_path, frame_paths)

    if missing_sources:
        errors.append(
            f"missing source files for {len(missing_sources)}/{min(max_source_checks, len(records))} "
            f"checked records; first missing: {missing_sources[:5]}"
        )
    else:
        print(f"  source check: first {min(max_source_checks, len(records))} record(s) found")

    if decode_one and first_source is not None:
        source_type, source_path, frame_paths = first_source
        print(f"  decode: {decode_one_source(source_type, source_path, frame_paths)}")

    sample = records[0] if records else {}
    compact_sample = {
        "vid_name": sample.get("vid_name"),
        "label": sample.get("label"),
        "label_name": sample.get("label_name"),
        "start_frame": sample.get("start_frame"),
        "end_frame": sample.get("end_frame"),
    }
    print(f"  sample: {json.dumps(compact_sample, ensure_ascii=False)}")

    if errors:
        print(f"  errors: {len(errors)}")
        for error in errors[:20]:
            print(f"    - {error}")
    else:
        print("  checks passed")
    return len(errors)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    annotation_dir = args.annotation_dir or dataset_root / "annotations"
    video_dir = args.video_dir or dataset_root / "videos"
    rawframes_dir = args.rawframes_dir or dataset_root / "rawframes"
    split_files = {
        "train": args.train_annotation,
        "test": args.test_annotation,
    }

    print(f"dataset_root: {dataset_root}")
    print(f"annotation_dir: {annotation_dir}")
    print(f"input_format: {args.input_format}")
    print(f"video_dir: {video_dir}")
    print(f"rawframes_dir: {rawframes_dir}")

    total_errors = 0
    for split, filename in split_files.items():
        annotation_path = annotation_dir / filename
        if not annotation_path.is_file():
            print(f"\n{split}: missing annotation file {annotation_path}")
            total_errors += 1
            continue
        total_errors += check_split(
            split=split,
            annotation_path=annotation_path,
            input_format=args.input_format,
            video_dir=video_dir,
            rawframes_dir=rawframes_dir,
            max_source_checks=args.max_source_checks,
            decode_one=args.decode_one,
        )

    if total_errors:
        raise SystemExit(f"\nDiving48 V2 check failed with {total_errors} error(s).")
    print("\nDiving48 V2 dataset checks passed.")


if __name__ == "__main__":
    main()

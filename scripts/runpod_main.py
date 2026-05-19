"""One-command RunPod smoke pipeline for Diving48 DiffTraj-PGM."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/workspace/data") if Path("/workspace").exists() else PROJECT_ROOT / "data"
DEFAULT_DATASET_ROOT = DEFAULT_DATA_ROOT / "diving48_v2"
DEFAULT_EMBEDDINGS_ROOT = DEFAULT_DATA_ROOT / "diving48_embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RunPod main smoke pipeline.")
    parser.add_argument("--stage", choices=["check", "fake_small", "dataset_small"], default="fake_small")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--download-videos", action="store_true", help="Download the full Diving48 video archive.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing dataset files instead of downloading.")
    parser.add_argument(
        "--skip-annotations",
        action="store_true",
        help="When downloading, skip annotation JSON download and use existing uploaded files.",
    )
    parser.add_argument("--skip-extract-embeddings", action="store_true", help="Use an existing small train.pt file.")
    parser.add_argument("--train-url", default=None)
    parser.add_argument("--test-url", default=None)
    parser.add_argument("--vocab-url", default=None)
    parser.add_argument("--video-url", default=None)
    parser.add_argument("--max-source-checks", type=int, default=5)
    parser.add_argument("--max-extract-samples", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--variants", default="E0,E4")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument("--subset-dir", type=Path, default=None)
    return parser.parse_args()


def selected_device_name(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_command(cmd: list[str], description: str) -> None:
    print(f"\n--- {description} ---")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def python_cmd() -> str:
    return sys.executable


def run_environment_check(args: argparse.Namespace) -> None:
    run_command(
        [
            python_cmd(),
            "scripts/check_runpod_environment.py",
            "--config",
            str(args.config),
            "--device",
            args.device,
        ],
        "environment check",
    )


def run_fake_small(args: argparse.Namespace) -> None:
    cmd = [
        python_cmd(),
        "scripts/runpod_small_start.py",
        "--mode",
        "fake",
        "--max-samples",
        str(args.max_train_samples),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--variants",
        args.variants,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    add_small_output_args(cmd, args)
    run_command(cmd, "fake small training")


def run_download(args: argparse.Namespace) -> None:
    if args.skip_download:
        print("\n--- dataset download skipped ---")
        return
    cmd = [
        python_cmd(),
        "scripts/download_diving48_v2.py",
        "--dataset-root",
        str(args.dataset_root),
        "--extract",
    ]
    if not args.download_videos:
        cmd.append("--skip-videos")
    if args.skip_annotations:
        cmd.append("--skip-annotations")
    for flag, value in [
        ("--train-url", args.train_url),
        ("--test-url", args.test_url),
        ("--vocab-url", args.vocab_url),
        ("--video-url", args.video_url),
    ]:
        if value:
            cmd.extend([flag, value])
    run_command(cmd, "dataset download / prepare")


def run_dataset_check(args: argparse.Namespace) -> None:
    run_command(
        [
            python_cmd(),
            "scripts/check_diving48_dataset.py",
            "--dataset_root",
            str(args.dataset_root),
            "--input_format",
            "auto",
            "--max_source_checks",
            str(args.max_source_checks),
            "--decode_one",
            "false",
        ],
        "dataset layout check",
    )


def run_small_embedding_extract(args: argparse.Namespace, device: str) -> Path:
    embedding_subdir = "small_clip_vit_b16"
    out_dir = args.embeddings_root / embedding_subdir
    train_file = out_dir / "train.pt"
    if args.skip_extract_embeddings and train_file.is_file():
        print(f"\n--- embedding extraction skipped, using {train_file} ---")
        return train_file

    run_command(
        [
            python_cmd(),
            "scripts/extract_diving48_clip_embeddings.py",
            "--dataset_root",
            str(args.dataset_root),
            "--out_dir",
            str(args.embeddings_root),
            "--embedding_subdir",
            embedding_subdir,
            "--input_format",
            "auto",
            "--num_frames",
            "16",
            "--max_samples_per_split",
            str(args.max_extract_samples),
            "--device",
            device,
            "--batch_size",
            "64",
        ],
        "small CLIP embedding extraction",
    )
    return train_file


def run_real_small(args: argparse.Namespace, train_file: Path) -> None:
    cmd = [
        python_cmd(),
        "scripts/runpod_small_start.py",
        "--mode",
        "real",
        "--embeddings-path",
        str(train_file),
        "--max-samples",
        str(args.max_train_samples),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--variants",
        args.variants,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    add_small_output_args(cmd, args)
    run_command(cmd, "real small training")


def add_small_output_args(cmd: list[str], args: argparse.Namespace) -> None:
    if args.output_root is not None:
        cmd.extend(["--output-root", str(args.output_root)])
    if args.registry_path is not None:
        cmd.extend(["--registry-path", str(args.registry_path)])
    if args.subset_dir is not None:
        cmd.extend(["--subset-dir", str(args.subset_dir)])


def run_dataset_small(args: argparse.Namespace) -> None:
    device = selected_device_name(args.device)
    run_download(args)
    run_dataset_check(args)
    train_file = run_small_embedding_extract(args, device)
    run_real_small(args, train_file)


def main() -> None:
    args = parse_args()
    try:
        run_environment_check(args)
        if args.stage == "fake_small":
            run_fake_small(args)
        elif args.stage == "dataset_small":
            run_dataset_small(args)
        print("\nRUNPOD_MAIN_STATUS: OK")
    except Exception as exc:
        print(f"\nRUNPOD_MAIN_STATUS: FAILED")
        print(f"Reason: {exc}")
        raise


if __name__ == "__main__":
    main()

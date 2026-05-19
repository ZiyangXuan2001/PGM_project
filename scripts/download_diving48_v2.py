"""Download and arrange Diving48 V2 files for local/RunPod training."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/workspace/data/diving48_v2") if Path("/workspace").exists() else PROJECT_ROOT / "data" / "diving48_v2"

DEFAULT_TRAIN_URL = "http://www.svcl.ucsd.edu/projects/resound/Diving48_V2_train.json"
DEFAULT_TEST_URL = "http://www.svcl.ucsd.edu/projects/resound/Diving48_V2_test.json"
DEFAULT_VOCAB_URL = "http://www.svcl.ucsd.edu/projects/resound/Diving48_vocab.json"
DEFAULT_VIDEO_URL = "http://www.svcl.ucsd.edu/projects/resound/Diving48_rgb.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Diving48 V2 annotations/videos.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-url", default=DEFAULT_TRAIN_URL)
    parser.add_argument("--test-url", default=DEFAULT_TEST_URL)
    parser.add_argument("--vocab-url", default=DEFAULT_VOCAB_URL)
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--skip-videos", action="store_true", help="Only download annotation files.")
    parser.add_argument("--extract", action="store_true", help="Extract the video archive after download.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the downloaded video archive after extraction.")
    return parser.parse_args()


def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_num * block_size, total_size)
    pct = downloaded / total_size * 100.0
    if block_num == 0 or downloaded == total_size or block_num % 200 == 0:
        print(f"  {downloaded / (1024 ** 3):.2f} / {total_size / (1024 ** 3):.2f} GB ({pct:.1f}%)")


def download_file(url: str, dest: Path, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force:
        print(f"exists, skip: {dest}")
        return
    print(f"download: {url}")
    print(f"to: {dest}")
    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)


def safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_resolved = dest_dir.resolve()
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = (dest_dir / member.name).resolve()
            if not str(member_path).startswith(str(dest_resolved)):
                raise RuntimeError(f"unsafe archive member path: {member.name}")
        tar.extractall(dest_dir)


def extract_archive(archive_path: Path, dataset_root: Path) -> None:
    print(f"extract: {archive_path}")
    if tarfile.is_tarfile(archive_path):
        safe_extract_tar(archive_path, dataset_root)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dataset_root)
    else:
        raise ValueError(f"unsupported archive format: {archive_path}")

    rgb_dir = dataset_root / "rgb"
    videos_dir = dataset_root / "videos"
    if rgb_dir.is_dir() and not videos_dir.exists():
        rgb_dir.rename(videos_dir)
        print(f"renamed {rgb_dir} -> {videos_dir}")
    elif rgb_dir.is_dir():
        print(f"found extracted rgb dir: {rgb_dir}; videos dir already exists, leaving both in place")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    annotation_dir = dataset_root / "annotations"
    download_dir = dataset_root / "downloads"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset_root: {dataset_root}")
    download_file(args.train_url, annotation_dir / "Diving48_V2_train.json", args.force)
    download_file(args.test_url, annotation_dir / "Diving48_V2_test.json", args.force)
    download_file(args.vocab_url, annotation_dir / "Diving48_vocab.json", args.force)

    if not args.skip_videos:
        archive_name = Path(args.video_url.split("?")[0]).name or "Diving48_rgb.tar.gz"
        archive_path = download_dir / archive_name
        download_file(args.video_url, archive_path, args.force)
        if args.extract:
            extract_archive(archive_path, dataset_root)
            if not args.keep_archive:
                archive_path.unlink(missing_ok=True)
                print(f"removed archive: {archive_path}")

    print("Diving48 V2 download step finished.")


if __name__ == "__main__":
    main()

"""Download and arrange Diving48 V2 files for local/RunPod training."""

from __future__ import annotations

import argparse
import contextlib
import json
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/workspace/data/diving48_v2") if Path("/workspace").exists() else PROJECT_ROOT / "data" / "diving48_v2"

DEFAULT_TRAIN_URL = "http://www.svcl.ucsd.edu/projects/resound/Diving48_V2_train.json"
DEFAULT_TEST_URL = "http://www.svcl.ucsd.edu/projects/resound/Diving48_V2_test.json"
DEFAULT_VOCAB_URL = "http://www.svcl.ucsd.edu/projects/resound/Diving48_vocab.json"
DEFAULT_VIDEO_URL = "https://huggingface.co/datasets/bkprocovid19/diving48/resolve/main/Diving48_rgb.tar.gz"
OFFICIAL_VIDEO_URL = "http://www.svcl.ucsd.edu/projects/resound/Diving48_rgb.tar.gz"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_HF_DATASET = "mteb/diving48"
DEFAULT_HF_CONFIG = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Diving48 V2 annotations/videos.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-url", default=DEFAULT_TRAIN_URL)
    parser.add_argument("--test-url", default=DEFAULT_TEST_URL)
    parser.add_argument("--vocab-url", default=DEFAULT_VOCAB_URL)
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument(
        "--annotation-source",
        choices=["hf", "ucsd"],
        default="hf",
        help="Download annotations from Hugging Face row metadata or the legacy UCSD JSON URLs.",
    )
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--hf-page-size", type=int, default=100, help="Hugging Face rows API page size, max 100.")
    parser.add_argument("--hf-max-retries", type=int, default=12)
    parser.add_argument(
        "--official-video-url",
        action="store_true",
        help="Use the original UCSD video URL instead of the Hugging Face mirror.",
    )
    parser.add_argument(
        "--skip-annotations",
        action="store_true",
        help="Do not download annotation JSON files; assume they are already present.",
    )
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
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )
    try:
        with contextlib.closing(urllib.request.urlopen(request, timeout=60)) as response:
            total_size = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            block_size = 1024 * 1024
            with dest.open("wb") as handle:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and (downloaded == total_size or downloaded // block_size % 200 == 0):
                        pct = downloaded / total_size * 100.0
                        print(
                            f"  {downloaded / (1024 ** 3):.2f} / "
                            f"{total_size / (1024 ** 3):.2f} GB ({pct:.1f}%)"
                        )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {url}: {exc}") from exc


def normalize_label_name(value: Any) -> str:
    if isinstance(value, list):
        return "_".join(str(part) for part in value)
    return str(value)


def hf_rows_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return f"{HF_ROWS_URL}?{query}"


def fetch_hf_rows(args: argparse.Namespace, split: str, offset: int, length: int) -> dict[str, Any]:
    url = hf_rows_url(args.hf_dataset, args.hf_config, split, offset, length)
    last_error: Exception | None = None
    for attempt in range(1, args.hf_max_retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        try:
            with contextlib.closing(urllib.request.urlopen(request, timeout=90)) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = exc
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if exc.code not in {429, 500, 502, 503, 504}:
                break
            sleep_sec = int(retry_after) if retry_after and retry_after.isdigit() else min(120, 5 * attempt)
            print(
                f"  {split} offset={offset}: HTTP {exc.code}, retry "
                f"{attempt}/{args.hf_max_retries} after {sleep_sec}s"
            )
            time.sleep(sleep_sec)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            sleep_sec = min(120, 5 * attempt)
            print(f"  {split} offset={offset}: {exc}, retry {attempt}/{args.hf_max_retries} after {sleep_sec}s")
            time.sleep(sleep_sec)
    raise RuntimeError(f"failed to fetch Hugging Face rows for {split} offset={offset}: {last_error}")


def convert_hf_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "vid_name": str(row["vid_name"]),
        "label": int(row["label"]),
        "label_name": normalize_label_name(row.get("label_name")),
        "start_frame": row.get("start_frame"),
        "end_frame": row.get("end_frame"),
    }


def write_json_atomic(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def download_hf_annotations(args: argparse.Namespace, annotation_dir: Path) -> None:
    output_files = [
        annotation_dir / "Diving48_V2_train.json",
        annotation_dir / "Diving48_V2_test.json",
        annotation_dir / "Diving48_vocab.json",
    ]
    if all(path.is_file() for path in output_files) and not args.force:
        for path in output_files:
            print(f"exists, skip: {path}")
        return

    page_size = max(1, min(100, args.hf_page_size))
    print(f"annotation source: Hugging Face {args.hf_dataset}")
    print("The HF dataset stores labels/metadata in rows, so this step generates Diving48 JSON files.")

    all_records: list[dict[str, Any]] = []
    for split, filename in [("train", "Diving48_V2_train.json"), ("test", "Diving48_V2_test.json")]:
        first_page = fetch_hf_rows(args, split, 0, page_size)
        total = int(first_page["num_rows_total"])
        records = [convert_hf_record(item["row"]) for item in first_page["rows"]]
        print(f"{split}: total={total}, page_size={page_size}")
        next_progress = max(page_size, 1000)
        for offset in range(page_size, total, page_size):
            data = fetch_hf_rows(args, split, offset, min(page_size, total - offset))
            records.extend(convert_hf_record(item["row"]) for item in data["rows"])
            if len(records) >= next_progress or len(records) == total:
                print(f"  {split}: fetched {min(len(records), total)}/{total}")
                next_progress += 1000

        if len(records) != total:
            raise RuntimeError(f"{split}: expected {total} records, got {len(records)}")
        out_path = annotation_dir / filename
        write_json_atomic(out_path, records)
        all_records.extend(records)
        print(f"wrote {out_path}")

    vocab_by_label: dict[int, str] = {}
    for record in all_records:
        label = int(record["label"])
        vocab_by_label.setdefault(label, str(record["label_name"]))
    missing = sorted(set(range(48)) - set(vocab_by_label))
    if missing:
        raise RuntimeError(f"missing label names for labels: {missing}")
    vocab = {vocab_by_label[label]: label for label in range(48)}
    vocab_path = annotation_dir / "Diving48_vocab.json"
    write_json_atomic(vocab_path, vocab)
    print(f"wrote {vocab_path}")


def annotation_failure_message(error: Exception, annotation_dir: Path) -> str:
    return f"""
Could not prepare Diving48 V2 annotations.

Reason:
  {error}

The RunPod CUDA/model environment may still be OK. The default annotation
source is Hugging Face row metadata. The legacy UCSD annotation URLs may return
HTTP 403 from cloud machines.

To continue, use one of these options:

Option A: retry the Hugging Face metadata source
  python scripts/download_diving48_v2.py --dataset-root <DATASET_ROOT> --skip-videos

Option B: use OpenDataLab/MMAction2 downloader on RunPod
  pip install -U openmim opendatalab
  odl login
  mim download mmaction2 --dataset diving48

Then copy or symlink these files into:
  {annotation_dir}/Diving48_V2_train.json
  {annotation_dir}/Diving48_V2_test.json
  {annotation_dir}/Diving48_vocab.json

Option C: upload the annotation files from your local machine:
  scp -P <PORT> Diving48_V2_train.json root@<RUNPOD_HOST>:{annotation_dir}/
  scp -P <PORT> Diving48_V2_test.json  root@<RUNPOD_HOST>:{annotation_dir}/
  scp -P <PORT> Diving48_vocab.json    root@<RUNPOD_HOST>:{annotation_dir}/

Option D: if you already have CLIP embeddings, skip dataset download and run:
  python scripts/runpod_small_start.py --mode real --embeddings-path /workspace/data/diving48_embeddings/train_embeddings.pt --samples-per-class 2 --epochs 3 --batch-size 16 --variants all
"""


def download_annotations(args: argparse.Namespace, annotation_dir: Path) -> None:
    try:
        if args.annotation_source == "hf":
            download_hf_annotations(args, annotation_dir)
        else:
            download_file(args.train_url, annotation_dir / "Diving48_V2_train.json", args.force)
            download_file(args.test_url, annotation_dir / "Diving48_V2_test.json", args.force)
            download_file(args.vocab_url, annotation_dir / "Diving48_vocab.json", args.force)
    except RuntimeError as exc:
        message = annotation_failure_message(exc, annotation_dir)
        (annotation_dir / "DOWNLOAD_FAILED.txt").write_text(message, encoding="utf-8")
        raise SystemExit(message) from exc


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
    if args.skip_annotations:
        print(f"annotation download skipped; expecting existing files under {annotation_dir}")
    else:
        download_annotations(args, annotation_dir)

    if not args.skip_videos:
        if args.official_video_url:
            args.video_url = OFFICIAL_VIDEO_URL
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

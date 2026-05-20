"""Extract CLIP frame embeddings for a local Diving48 V2 dataset.

This script expects the dataset to already exist on disk. It does not download
annotations, videos, or frames.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "diving48_v2"
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".mov", ".webm"]
FALLBACK_LABEL_NAMES = [f"class_{idx:02d}" for idx in range(48)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings for Diving48 V2.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--annotation_dir", type=Path, default=None)
    parser.add_argument("--video_dir", type=Path, default=None)
    parser.add_argument("--rawframes_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--input_format", choices=["auto", "videos", "rawframes"], default="auto")
    parser.add_argument("--train_annotation", default="Diving48_V2_train.json")
    parser.add_argument("--test_annotation", default="Diving48_V2_test.json")
    parser.add_argument("--vocab_file", default="Diving48_vocab.json")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--val_fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--backbone_name", default="ViT-B/16")
    parser.add_argument("--embedding_subdir", default="clip_vit_b16")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available.")
    return torch.device(device_name)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of annotation records")
    records: list[dict[str, Any]] = []
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"{path}[{idx}] must be an object")
        vid_name = record.get("vid_name")
        label = record.get("label")
        if not isinstance(vid_name, str) or not vid_name:
            raise ValueError(f"{path}[{idx}] invalid vid_name: {vid_name!r}")
        if not isinstance(label, int) or not 0 <= label < 48:
            raise ValueError(f"{path}[{idx}] invalid label: {label!r}")
        records.append(record)
    return records


def normalize_label_name(value: Any) -> str:
    if isinstance(value, list):
        return "-".join(str(part) for part in value)
    if isinstance(value, str):
        return value
    return str(value)


def label_names_from_vocab(vocab_path: Path) -> list[str] | None:
    if not vocab_path.is_file():
        return None
    vocab = load_json(vocab_path)
    if isinstance(vocab, dict):
        names = FALLBACK_LABEL_NAMES.copy()
        for key, value in vocab.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                if isinstance(value, int) and 0 <= value < 48:
                    names[value] = normalize_label_name(key)
                continue
            if 0 <= idx < 48:
                names[idx] = normalize_label_name(value)
        return names
    if isinstance(vocab, list) and len(vocab) >= 48:
        return [normalize_label_name(value) for value in vocab[:48]]
    raise ValueError(f"{vocab_path} has an unsupported vocabulary format")


def derive_label_names(records: list[dict[str, Any]], vocab_path: Path) -> list[str]:
    vocab_names = label_names_from_vocab(vocab_path)
    if vocab_names is not None:
        return vocab_names

    names = FALLBACK_LABEL_NAMES.copy()
    for record in records:
        label_name = record.get("label_name")
        if label_name is not None:
            names[int(record["label"])] = normalize_label_name(label_name)
    return names


def stratified_train_val_split(
    records: list[dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0.0 <= val_fraction < 1.0")
    if val_fraction == 0.0:
        return records, []

    rng = random.Random(seed)
    by_label: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(int(record["label"]), []).append(record)

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for label_records in by_label.values():
        shuffled = label_records.copy()
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_fraction))
        if len(shuffled) > 1:
            val_count = max(1, min(val_count, len(shuffled) - 1))
        else:
            val_count = 0
        val_records.extend(shuffled[:val_count])
        train_records.extend(shuffled[val_count:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def preprocessing_stats(backbone_name: str) -> tuple[int, torch.Tensor, torch.Tensor]:
    if backbone_name == "resnet50":
        return 224, IMAGENET_MEAN, IMAGENET_STD
    return CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD


def preprocess_pil_image(
    image: Image.Image,
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    image = image.convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), RESAMPLE_BICUBIC)
    image_bytes = image.tobytes()
    tensor = torch.frombuffer(image_bytes, dtype=torch.uint8).clone()
    tensor = tensor.view(image_size, image_size, 3).permute(2, 0, 1)
    tensor = tensor.float().div(255.0)
    return (tensor - mean) / std


def preprocess_image_path(
    path: Path,
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess_pil_image(image, image_size=image_size, mean=mean, std=std)


def load_encoder(
    device: torch.device,
    backbone_name: str,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    if backbone_name == "resnet50":
        try:
            from models import ResNetFrameEncoder

            encoder = ResNetFrameEncoder(
                backbone_name=backbone_name,
                freeze=True,
                normalize=True,
                device=device,
            )
            encoder.eval()

            def encode_with_resnet(frames: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    return encoder(frames.unsqueeze(1)).squeeze(1).detach().cpu()

            return encode_with_resnet, "models.ResNetFrameEncoder(resnet50)"
        except ImportError:
            raise

    try:
        from models import CLIPFrameEncoder

        encoder = CLIPFrameEncoder(
            backbone_name=backbone_name,
            freeze=True,
            normalize=True,
            device=device,
        )
        encoder.eval()

        def encode_with_project_encoder(frames: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return encoder(frames.unsqueeze(1)).squeeze(1).detach().cpu()

        return encode_with_project_encoder, f"models.CLIPFrameEncoder({backbone_name})"
    except ImportError as exc:
        if "OpenAI CLIP is required" in str(exc) or "No module named 'clip'" in str(exc):
            raise

    try:
        import clip  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenAI CLIP is required for embedding extraction.\n"
            "pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    model, _ = clip.load(backbone_name, device=device, jit=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    def encode_direct(frames: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = model.encode_image(frames.to(device))
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
            return embeddings.detach().cpu()

    return encode_direct, f"direct OpenAI CLIP({backbone_name})"


def uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        raise ValueError("cannot sample from an empty frame sequence")
    if count <= 0:
        raise ValueError("num_frames must be positive")
    if count == 1:
        return [length // 2]
    positions = torch.linspace(0, length - 1, steps=count)
    return [min(length - 1, max(0, int(round(pos.item())))) for pos in positions]


def rawframe_paths_for(record: dict[str, Any], rawframes_dir: Path) -> list[Path]:
    frame_dir = rawframes_dir / str(record["vid_name"])
    if not frame_dir.is_dir():
        return []
    return sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def video_path_for(record: dict[str, Any], video_dir: Path) -> Path | None:
    vid_name = str(record["vid_name"])
    direct = video_dir / vid_name
    if direct.is_file():
        return direct
    for extension in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{vid_name}{extension}"
        if candidate.is_file():
            return candidate
    return None


def load_rawframes(
    record: dict[str, Any],
    rawframes_dir: Path,
    num_frames: int,
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    frame_paths = rawframe_paths_for(record, rawframes_dir)
    if not frame_paths:
        raise FileNotFoundError(f"missing rawframes for {record['vid_name']} under {rawframes_dir}")
    indices = uniform_indices(len(frame_paths), num_frames)
    sampled_paths = [frame_paths[idx] for idx in indices]
    frames = torch.stack(
        [
            preprocess_image_path(path, image_size=image_size, mean=mean, std=std)
            for path in sampled_paths
        ],
        dim=0,
    )
    metadata = {
        "source_type": "rawframes",
        "source_path": str(rawframes_dir / str(record["vid_name"])),
        "source_frame_count": len(frame_paths),
        "sampled_frame_indices": indices,
        "sampled_frame_paths": [str(path.relative_to(rawframes_dir)) for path in sampled_paths],
    }
    return frames, metadata


def load_video_frames(
    record: dict[str, Any],
    video_dir: Path,
    num_frames: int,
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    video_path = video_path_for(record, video_dir)
    if video_path is None:
        raise FileNotFoundError(f"missing video for {record['vid_name']} under {video_dir}")

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required when extracting from mp4 videos. "
            "Install it on the training machine or pass --input_format rawframes."
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"OpenCV could not open {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"{video_path} reports no frames")
        indices = uniform_indices(frame_count, num_frames)
        frames: list[torch.Tensor] = []
        for frame_idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise ValueError(f"failed to read frame {frame_idx} from {video_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(
                preprocess_pil_image(
                    Image.fromarray(frame_rgb),
                    image_size=image_size,
                    mean=mean,
                    std=std,
                )
            )
    finally:
        capture.release()

    metadata = {
        "source_type": "videos",
        "source_path": str(video_path),
        "source_frame_count": frame_count,
        "sampled_frame_indices": indices,
    }
    return torch.stack(frames, dim=0), metadata


def load_sequence_frames(
    record: dict[str, Any],
    input_format: str,
    video_dir: Path,
    rawframes_dir: Path,
    num_frames: int,
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if input_format in {"auto", "rawframes"}:
        try:
            return load_rawframes(
                record,
                rawframes_dir,
                num_frames,
                image_size=image_size,
                mean=mean,
                std=std,
            )
        except FileNotFoundError:
            if input_format == "rawframes":
                raise

    return load_video_frames(
        record,
        video_dir,
        num_frames,
        image_size=image_size,
        mean=mean,
        std=std,
    )


def encode_sequence(
    frames: torch.Tensor,
    encode_batch: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, frames.shape[0], batch_size):
        chunks.append(encode_batch(frames[start : start + batch_size]))
    return torch.cat(chunks, dim=0)


def compact_metadata(
    record: dict[str, Any],
    split: str,
    label_names: list[str],
    frame_metadata: dict[str, Any],
    num_frames: int,
) -> dict[str, Any]:
    label = int(record["label"])
    return {
        "sample_id": str(record["vid_name"]),
        "split": split,
        "vid_name": str(record["vid_name"]),
        "label": label,
        "label_name": normalize_label_name(record.get("label_name", label_names[label])),
        "T": num_frames,
        "start_frame": record.get("start_frame"),
        "end_frame": record.get("end_frame"),
        **frame_metadata,
    }


def process_split(
    split: str,
    records: list[dict[str, Any]],
    label_names: list[str],
    input_format: str,
    video_dir: Path,
    rawframes_dir: Path,
    out_dir: Path,
    encode_batch: Callable[[torch.Tensor], torch.Tensor],
    backbone_name: str,
    num_frames: int,
    batch_size: int,
    max_samples: int | None,
) -> None:
    if max_samples is not None:
        records = records[:max_samples]
    if not records:
        raise ValueError(f"{split} contains no samples")

    embeddings: list[torch.Tensor] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    embedding_dim: int | None = None
    image_size, mean, std = preprocessing_stats(backbone_name)
    for idx, record in enumerate(records, start=1):
        frames, frame_metadata = load_sequence_frames(
            record=record,
            input_format=input_format,
            video_dir=video_dir,
            rawframes_dir=rawframes_dir,
            num_frames=num_frames,
            image_size=image_size,
            mean=mean,
            std=std,
        )
        sequence_embeddings = encode_sequence(frames, encode_batch, batch_size)
        if sequence_embeddings.ndim != 2 or sequence_embeddings.shape[0] != num_frames:
            raise ValueError(
                f"expected {record['vid_name']} embeddings [{num_frames}, D], "
                f"got {tuple(sequence_embeddings.shape)}"
            )
        current_dim = int(sequence_embeddings.shape[-1])
        if embedding_dim is None:
            embedding_dim = current_dim
        elif current_dim != embedding_dim:
            raise ValueError(
                f"mixed embedding dims in {split}: first D={embedding_dim}, "
                f"{record['vid_name']} D={current_dim}"
            )
        embeddings.append(sequence_embeddings)
        labels.append(int(record["label"]))
        metadata.append(
            compact_metadata(
                record=record,
                split=split,
                label_names=label_names,
                frame_metadata=frame_metadata,
                num_frames=num_frames,
            )
        )
        if idx == 1 or idx % 50 == 0 or idx == len(records):
            print(f"  {split}: encoded {idx}/{len(records)}")

    X = torch.stack(embeddings, dim=0).contiguous()
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    payload = {
        "X": X,
        "labels": labels_tensor,
        "label_names": label_names,
        "metadata": metadata,
        "backbone_name": backbone_name,
        "embedding_dim": int(X.shape[-1]),
        "num_frames": int(X.shape[1]),
    }
    out_path = out_dir / f"{split}.pt"
    torch.save(payload, out_path)
    print(f"  saved {out_path} with X={tuple(X.shape)}, labels={tuple(labels_tensor.shape)}")


def main() -> None:
    args = parse_args()
    if args.num_frames <= 0:
        raise SystemExit("num_frames must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("batch_size must be positive.")
    if args.max_samples_per_split is not None and args.max_samples_per_split <= 0:
        raise SystemExit("max_samples_per_split must be positive when provided.")

    dataset_root = args.dataset_root
    annotation_dir = args.annotation_dir or dataset_root / "annotations"
    video_dir = args.video_dir or dataset_root / "videos"
    rawframes_dir = args.rawframes_dir or dataset_root / "rawframes"
    out_root = args.out_dir or dataset_root / "embeddings"
    out_dir = out_root / args.embedding_subdir if args.embedding_subdir else out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records_all = load_records(annotation_dir / args.train_annotation)
    test_records = load_records(annotation_dir / args.test_annotation)
    train_records, val_records = stratified_train_val_split(
        train_records_all,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    split_records = {"train": train_records}
    if val_records:
        split_records["val"] = val_records
    split_records["test"] = test_records

    label_names = derive_label_names(train_records_all + test_records, annotation_dir / args.vocab_file)
    device = resolve_device(args.device)
    try:
        encode_batch, encoder_name = load_encoder(device, args.backbone_name)
    except ImportError as exc:
        print(exc)
        raise SystemExit(1) from exc

    print(f"Using {encoder_name} on {device}.")
    print(f"dataset_root: {dataset_root}")
    print(f"input_format: {args.input_format}")
    print(f"num_frames: {args.num_frames}")
    print(f"out_dir: {out_dir}")
    if not val_records:
        print("No held-out val split was created; configs use test.pt as val_file by default.")

    for split, records in split_records.items():
        process_split(
            split=split,
            records=records,
            label_names=label_names,
            input_format=args.input_format,
            video_dir=video_dir,
            rawframes_dir=rawframes_dir,
            out_dir=out_dir,
            encode_batch=encode_batch,
            backbone_name=args.backbone_name,
            num_frames=args.num_frames,
            batch_size=args.batch_size,
            max_samples=args.max_samples_per_split,
        )


if __name__ == "__main__":
    main()

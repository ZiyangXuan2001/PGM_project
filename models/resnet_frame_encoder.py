"""Optional torchvision ResNet frame encoder for precomputed frame embeddings."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class ResNetFrameEncoder(nn.Module):
    """Encode each frame independently with a frozen torchvision ResNet.

    The default ``resnet50`` encoder returns one 2048-dimensional feature vector
    per frame after global average pooling, preserving the project input shape
    convention ``[B, T, D]``.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        freeze: bool = True,
        normalize: bool = True,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        if backbone_name != "resnet50":
            raise ValueError(f"ResNetFrameEncoder only supports resnet50, got {backbone_name!r}")
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for ResNetFrameEncoder. Install it with:\n"
                "  pip install torchvision"
            ) from exc

        self.backbone_name = backbone_name
        self.freeze = freeze
        self.normalize = normalize
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        try:
            weights = ResNet50_Weights.IMAGENET1K_V2
        except AttributeError:
            weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        self.feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        self.model = model.to(self.device)

        if self.freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames of shape ``[B, T, 3, H, W]`` into ``[B, T, 2048]``."""

        if frames.ndim != 5:
            raise ValueError(
                f"frames must have shape [B, T, 3, H, W], got {tuple(frames.shape)}"
            )
        batch_size, num_frames, channels, height, width = frames.shape
        if channels != 3:
            raise ValueError(f"frames must have 3 channels, got {channels}")

        flat_frames = frames.reshape(batch_size * num_frames, channels, height, width)
        flat_frames = flat_frames.to(self.device)

        if self.freeze:
            with torch.no_grad():
                embeddings = self.model(flat_frames)
        else:
            embeddings = self.model(flat_frames)

        embeddings = embeddings.float()
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.reshape(batch_size, num_frames, -1)

"""Optional CLIP frame encoder.

This module deliberately performs no temporal reasoning. It maps each raw frame
independently into a CLIP image embedding and reshapes the result back to a
sequence.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class CLIPFrameEncoder(nn.Module):
    """Encode raw video frames with OpenAI CLIP ViT-B/16 by default.

    Parameters
    ----------
    backbone_name:
        CLIP backbone name passed to ``clip.load``.
    freeze:
        If true, CLIP parameters are frozen and encoding runs under no-grad.
    normalize:
        If true, L2-normalize output embeddings along the feature dimension.
    device:
        Optional device used when loading CLIP.
    """

    def __init__(
        self,
        backbone_name: str = "ViT-B/16",
        freeze: bool = True,
        normalize: bool = True,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        try:
            import clip  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenAI CLIP is required for CLIPFrameEncoder. Install it with:\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        self.backbone_name = backbone_name
        self.freeze = freeze
        self.normalize = normalize
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        clip_model, _preprocess = clip.load(backbone_name, device=self.device)
        self.clip_model = clip_model

        if self.freeze:
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames of shape ``[B, T, 3, H, W]`` into ``[B, T, D_clip]``."""

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
                embeddings = self.clip_model.encode_image(flat_frames)
        else:
            embeddings = self.clip_model.encode_image(flat_frames)

        embeddings = embeddings.float()
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.reshape(batch_size, num_frames, -1)

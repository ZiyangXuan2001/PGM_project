"""Classifier heads used by the controlled DiffTraj-PGM model."""

from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Generic MLP classifier for pooled vector representations."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must have shape [B, input_dim], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {x.shape[-1]}")
        return self.classifier(x)


class InformationMatrixClassifier(nn.Module):
    """Flatten an information matrix and produce class logits."""

    def __init__(
        self,
        K: int = 8,
        d_h: int = 128,
        num_classes: int = 48,
        classifier_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.K = K
        self.d_h = d_h
        self.num_classes = num_classes

        self.classifier = MLPClassifier(
            input_dim=K * d_h,
            num_classes=num_classes,
            hidden_dim=classifier_hidden,
            dropout=dropout,
        )

    def forward(self, H_final: torch.Tensor) -> torch.Tensor:
        if H_final.ndim != 3:
            raise ValueError(
                f"H_final must have shape [B, K, d_h], got {tuple(H_final.shape)}"
            )
        if H_final.shape[1:] != (self.K, self.d_h):
            raise ValueError(
                f"expected H_final shape [B, {self.K}, {self.d_h}], "
                f"got {tuple(H_final.shape)}"
            )

        return self.classifier(H_final.flatten(start_dim=1))


class MeanPooledInformationMatrixClassifier(nn.Module):
    """Mean-pool information rows before an MLP classifier."""

    def __init__(
        self,
        d_h: int = 128,
        num_classes: int = 48,
        classifier_hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.d_h = d_h
        self.classifier = MLPClassifier(
            input_dim=d_h,
            num_classes=num_classes,
            hidden_dim=classifier_hidden,
            dropout=dropout,
        )

    def forward(self, H_final: torch.Tensor) -> torch.Tensor:
        if H_final.ndim != 3:
            raise ValueError(f"H_final must have shape [B, K, d_h], got {tuple(H_final.shape)}")
        if H_final.shape[-1] != self.d_h:
            raise ValueError(f"expected d_h={self.d_h}, got {H_final.shape[-1]}")
        return self.classifier(H_final.mean(dim=1))


class AttentionPooledInformationMatrixClassifier(nn.Module):
    """Learnable-query attention pooling over information matrix rows."""

    def __init__(
        self,
        d_h: int = 128,
        num_classes: int = 48,
        classifier_hidden: int = 256,
        dropout: float = 0.2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if d_h % num_heads != 0:
            raise ValueError("d_h must be divisible by num_heads")
        self.d_h = d_h
        self.query = nn.Parameter(torch.zeros(1, 1, d_h))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_h,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = MLPClassifier(
            input_dim=d_h,
            num_classes=num_classes,
            hidden_dim=classifier_hidden,
            dropout=dropout,
        )

    def forward(self, H_final: torch.Tensor) -> torch.Tensor:
        if H_final.ndim != 3:
            raise ValueError(f"H_final must have shape [B, K, d_h], got {tuple(H_final.shape)}")
        if H_final.shape[-1] != self.d_h:
            raise ValueError(f"expected d_h={self.d_h}, got {H_final.shape[-1]}")
        query = self.query.expand(H_final.shape[0], -1, -1)
        pooled, _ = self.attention(query=query, key=H_final, value=H_final, need_weights=False)
        return self.classifier(pooled.squeeze(1))

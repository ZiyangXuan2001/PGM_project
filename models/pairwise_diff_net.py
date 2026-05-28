"""Pairwise embedding difference network."""

from __future__ import annotations

import torch
from torch import nn


class PairwiseDiffNet(nn.Module):
    """Compute local difference embeddings for adjacent frame embeddings."""

    def __init__(
        self,
        D: int = 512,
        d_y: int = 128,
        pair_hidden: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.D = D
        self.d_y = d_y

        self.mlp = nn.Sequential(
            nn.Linear(4 * D, pair_hidden),
            nn.GELU(),
            nn.LayerNorm(pair_hidden),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden, d_y),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return pairwise embeddings ``R`` with shape ``[B, T-1, d_y]``."""

        if X.ndim != 3:
            raise ValueError(f"X must have shape [B, T, D], got {tuple(X.shape)}")
        if X.shape[-1] != self.D:
            raise ValueError(f"expected embedding dim D={self.D}, got {X.shape[-1]}")
        if X.shape[1] < 2:
            raise ValueError("X must contain at least two frames so T-1 >= 1")

        x_t = X[:, :-1, :]
        x_next = X[:, 1:, :]
        diff = x_next - x_t
        U = torch.cat([x_t, x_next, diff, diff.abs()], dim=-1)
        return self.mlp(U)


class SimpleConcatPairwiseDiffNet(nn.Module):
    """Order-aware adjacent-pair network using only ``[x_t, x_{t+1}]``.

    Supports both global sequences ``[B, T, D]`` and spatial-token sequences
    ``[B, T, K, D]``. Linear and LayerNorm modules operate over the last
    dimension, so the same MLP handles either input rank.
    """

    def __init__(
        self,
        D: int = 64,
        d_y: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.D = D
        self.d_y = d_y
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(2 * D),
            nn.Linear(2 * D, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_y),
            nn.LayerNorm(d_y),
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return relation features for adjacent pairs.

        Global input:

        ```text
        X: [B, T, D] -> R: [B, T-1, d_y]
        ```

        Spatial input:

        ```text
        X: [B, T, K, D] -> R: [B, T-1, K, d_y]
        ```
        """

        if X.ndim not in {3, 4}:
            raise ValueError(f"X must have shape [B, T, D] or [B, T, K, D], got {tuple(X.shape)}")
        if X.shape[-1] != self.D:
            raise ValueError(f"expected embedding dim D={self.D}, got {X.shape[-1]}")
        if X.shape[1] < 2:
            raise ValueError("X must contain at least two frames so T-1 >= 1")

        x_t = X[:, :-1, ...]
        x_next = X[:, 1:, ...]
        z = torch.cat([x_t, x_next], dim=-1)
        return self.mlp(z)

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

"""Information matrix accumulator for variable-length trajectories."""

from __future__ import annotations

import torch
from torch import nn


class InformationMatrixAccumulator(nn.Module):
    """Accumulate a smoothed trajectory into a fixed-size information matrix."""

    def __init__(
        self,
        d_y: int = 128,
        K: int = 8,
        d_h: int = 128,
        eta: float = 0.1,
        eps: float = 1e-6,
        use_alpha: bool = True,
        normalize_delta: bool = True,
    ) -> None:
        super().__init__()
        self.d_y = d_y
        self.K = K
        self.d_h = d_h
        self.eta = eta
        self.eps = eps
        self.use_alpha = use_alpha
        self.normalize_delta = normalize_delta

        self.H0 = nn.Parameter(torch.zeros(K, d_h))
        input_dim = d_y + d_h

        self.mlp_alpha = nn.Sequential(
            nn.Linear(input_dim, d_h),
            nn.GELU(),
            nn.Linear(d_h, 1),
        )
        self.mlp_delta = nn.Sequential(
            nn.Linear(input_dim, d_h),
            nn.GELU(),
            nn.Linear(d_h, K * d_h),
        )
        self.layer_norm = nn.LayerNorm(d_h)

    def forward(self, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``H_final`` and per-step alpha values."""

        if Y.ndim != 3:
            raise ValueError(f"Y must have shape [B, L, d_y], got {tuple(Y.shape)}")
        if Y.shape[-1] != self.d_y:
            raise ValueError(f"expected d_y={self.d_y}, got {Y.shape[-1]}")

        batch_size, length, _ = Y.shape
        H = self.H0.unsqueeze(0).expand(batch_size, -1, -1)
        alpha_values = []

        for t in range(length):
            summary = H.mean(dim=1)
            update_input = torch.cat([Y[:, t, :], summary], dim=-1)

            if self.use_alpha:
                alpha = torch.sigmoid(self.mlp_alpha(update_input))
            else:
                alpha = torch.ones(batch_size, 1, device=Y.device, dtype=Y.dtype)

            delta_raw = self.mlp_delta(update_input).reshape(batch_size, self.K, self.d_h)
            if self.normalize_delta:
                fro_norm = torch.linalg.vector_norm(delta_raw, ord=2, dim=(1, 2), keepdim=True)
                delta = delta_raw / (fro_norm + self.eps)
            else:
                delta = delta_raw

            H = self.layer_norm(H + self.eta * alpha.unsqueeze(-1) * delta)
            alpha_values.append(alpha)

        alpha_tensor = torch.stack(alpha_values, dim=1)
        return H, alpha_tensor

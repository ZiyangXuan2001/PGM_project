"""Differentiable Gaussian-chain PGM smoother."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GaussianPGMSmoother(nn.Module):
    """Smooth local difference embeddings with a path-graph Gaussian prior.

    The MAP solution minimizes:

        sum_t ||Y_t - R_t||^2 + lambda_smooth * sum_t ||Y_t - Y_{t-1}||^2

    which yields the linear system ``(I + lambda_smooth * L_path)Y = R``.
    """

    def __init__(
        self,
        lambda_init: float = 0.5,
        learnable_lambda: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if lambda_init < 0:
            raise ValueError("lambda_init must be non-negative")
        if learnable_lambda and lambda_init <= 0:
            raise ValueError("learnable lambda_init must be positive")

        self.learnable_lambda = learnable_lambda
        self.eps = eps

        if learnable_lambda:
            # Inverse softplus so softplus(lambda_raw) starts near lambda_init.
            lambda_raw_init = math.log(math.expm1(lambda_init))
            self.lambda_raw = nn.Parameter(torch.tensor(lambda_raw_init))
            self.register_buffer("_fixed_lambda", torch.empty(0), persistent=False)
        else:
            self.register_parameter("lambda_raw", None)
            self.register_buffer("_fixed_lambda", torch.tensor(float(lambda_init)))

    @property
    def lambda_smooth(self) -> torch.Tensor:
        """Current non-negative smoothness strength."""

        if self.learnable_lambda:
            return F.softplus(self.lambda_raw) + self.eps
        return self._fixed_lambda

    def _path_laplacian(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if length < 1:
            raise ValueError("length must be >= 1")

        laplacian = torch.zeros(length, length, device=device, dtype=dtype)
        if length == 1:
            return laplacian

        indices = torch.arange(length, device=device)
        laplacian[indices, indices] = 2.0
        laplacian[0, 0] = 1.0
        laplacian[-1, -1] = 1.0
        off_diag = torch.arange(length - 1, device=device)
        laplacian[off_diag, off_diag + 1] = -1.0
        laplacian[off_diag + 1, off_diag] = -1.0
        return laplacian

    def forward(self, R: torch.Tensor, return_lambda: bool = False):
        """Smooth ``R`` of shape ``[B, L, d_y]`` into ``Y`` with same shape."""

        if R.ndim != 3:
            raise ValueError(f"R must have shape [B, L, d_y], got {tuple(R.shape)}")
        length = R.shape[1]
        original_dtype = R.dtype
        solve_dtype = torch.float32 if R.dtype in {torch.float16, torch.bfloat16} else R.dtype
        R_solve = R.to(dtype=solve_dtype)
        lam = self.lambda_smooth.to(device=R.device, dtype=solve_dtype)

        identity = torch.eye(length, device=R.device, dtype=solve_dtype)
        laplacian = self._path_laplacian(length, R.device, solve_dtype)
        system = identity + lam * laplacian

        # torch.linalg.solve supports batched RHS. Here [L, L] solves [L, B*d_y].
        rhs = R_solve.permute(1, 0, 2).reshape(length, -1)
        solution = torch.linalg.solve(system, rhs)
        Y = solution.reshape(length, R.shape[0], R.shape[2]).permute(1, 0, 2)
        Y = Y.to(dtype=original_dtype)

        if return_lambda:
            return Y, lam.to(dtype=original_dtype)
        return Y

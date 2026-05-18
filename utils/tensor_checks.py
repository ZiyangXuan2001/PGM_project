"""Small tensor validation helpers used by smoke tests."""

from __future__ import annotations

import torch


def assert_shape(
    tensor: torch.Tensor,
    expected_rank: int | None = None,
    name: str = "tensor",
) -> None:
    """Assert that a tensor has an expected rank."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
    if expected_rank is not None and tensor.ndim != expected_rank:
        raise AssertionError(
            f"{name} must have rank {expected_rank}, got shape {tuple(tensor.shape)}"
        )


def check_finite(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert that all tensor values are finite."""

    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf values")

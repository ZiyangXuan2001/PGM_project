"""Utility helpers."""

from .config import load_config
from .tensor_checks import assert_shape, check_finite

__all__ = ["assert_shape", "check_finite", "load_config"]

"""Simplified P-series trajectory-matrix model for frozen CLIP features."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TemporalProjection(nn.Module):
    """Per-frame CLIP projection with no temporal mixing."""

    def __init__(self, clip_dim: int = 512, d_z: int = 128) -> None:
        super().__init__()
        self.clip_dim = clip_dim
        self.d_z = d_z
        self.net = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, d_z),
            nn.GELU(),
            nn.LayerNorm(d_z),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 3:
            raise ValueError(f"X must have shape [B, T, clip_dim], got {tuple(X.shape)}")
        if X.shape[-1] != self.clip_dim:
            raise ValueError(f"expected clip_dim={self.clip_dim}, got {X.shape[-1]}")
        return self.net(X)


class GaussianFramePGMSmoother(nn.Module):
    """Fixed Gaussian-chain MAP smoother over projected frame states."""

    def __init__(
        self,
        use_pgm: bool = True,
        alpha: float = 1.0,
        lambda_frame: float = 0.05,
    ) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if lambda_frame < 0:
            raise ValueError("lambda_frame must be non-negative")
        self.use_pgm = use_pgm
        self.register_buffer("_alpha", torch.tensor(float(alpha)))
        self.register_buffer("_lambda_frame", torch.tensor(float(lambda_frame)))

    @property
    def alpha(self) -> torch.Tensor:
        return self._alpha

    @property
    def lambda_frame(self) -> torch.Tensor:
        return self._lambda_frame

    def _chain_laplacian(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if T < 1:
            raise ValueError("T must be >= 1")
        L = torch.zeros(T, T, device=device, dtype=dtype)
        if T == 1:
            return L
        indices = torch.arange(T, device=device)
        L[indices, indices] = 2.0
        L[0, 0] = 1.0
        L[-1, -1] = 1.0
        off_diag = torch.arange(T - 1, device=device)
        L[off_diag, off_diag + 1] = -1.0
        L[off_diag + 1, off_diag] = -1.0
        return L

    def forward(self, U: torch.Tensor, return_lambda: bool = False):
        if U.ndim != 3:
            raise ValueError(f"U must have shape [B, T, d_z], got {tuple(U.shape)}")

        alpha = self.alpha.to(device=U.device, dtype=U.dtype)
        lambda_frame = self.lambda_frame.to(device=U.device, dtype=U.dtype)
        if not self.use_pgm or float(lambda_frame.detach().cpu()) == 0.0:
            if return_lambda:
                return U, lambda_frame
            return U

        T = U.shape[1]
        I = torch.eye(T, device=U.device, dtype=U.dtype)
        L = self._chain_laplacian(T, U.device, U.dtype)
        A = alpha * I + lambda_frame * L
        rhs = alpha * U.permute(1, 0, 2).reshape(T, -1)
        Z = torch.linalg.solve(A, rhs)
        Z = Z.reshape(T, U.shape[0], U.shape[2]).permute(1, 0, 2)

        if return_lambda:
            return Z, lambda_frame
        return Z


class ProjectedPairwiseDiffNet(nn.Module):
    """Adjacent-pair DiffNet over projected frame states."""

    def __init__(
        self,
        d_z: int = 128,
        d_r: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.d_z = d_z
        self.d_r = d_r
        self.net = nn.Sequential(
            nn.Linear(4 * d_z, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_r),
        )

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        if F.ndim != 3:
            raise ValueError(f"F must have shape [B, T, d_z], got {tuple(F.shape)}")
        if F.shape[-1] != self.d_z:
            raise ValueError(f"expected d_z={self.d_z}, got {F.shape[-1]}")
        if F.shape[1] < 2:
            raise ValueError("F must contain at least two frames")
        f_t = F[:, :-1, :]
        f_next = F[:, 1:, :]
        diff = f_next - f_t
        abs_diff = diff.abs()
        pair = torch.cat([f_t, f_next, diff, abs_diff], dim=-1)
        return self.net(pair)


class TrajectoryMatrixClassifier(nn.Module):
    """Linear classifier over the flattened temporal relation matrix."""

    def __init__(
        self,
        num_frames: int = 16,
        d_r: int = 128,
        num_classes: int = 48,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_pairs = num_frames - 1
        self.d_r = d_r
        self.num_classes = num_classes
        input_dim = self.num_pairs * d_r
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        if R.ndim != 3:
            raise ValueError(f"R must have shape [B, T-1, d_r], got {tuple(R.shape)}")
        expected_tail = (self.num_pairs, self.d_r)
        if R.shape[1:] != expected_tail:
            raise ValueError(f"expected R tail shape {expected_tail}, got {tuple(R.shape[1:])}")
        return self.net(R.flatten(start_dim=1))


class TrajectoryMatrixAttentionClassifier(nn.Module):
    """Attention classifier over temporal relation tokens.

    This head reads the DiffNet relation sequence ``R [B, T-1, d_r]`` directly.
    It does not use the InformationMatrixAccumulator; the only PGM component is
    the optional Gaussian smoother before this classifier.
    """

    def __init__(
        self,
        num_frames: int = 16,
        d_r: int = 128,
        num_classes: int = 48,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if d_r % num_heads != 0:
            raise ValueError("d_r must be divisible by num_heads")
        self.num_pairs = num_frames - 1
        self.d_r = d_r
        self.num_classes = num_classes
        self.input_norm = nn.LayerNorm(d_r)
        self.query = nn.Parameter(torch.zeros(1, 1, d_r))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_r,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.net = nn.Sequential(
            nn.LayerNorm(3 * d_r),
            nn.Dropout(dropout),
            nn.Linear(3 * d_r, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        if R.ndim != 3:
            raise ValueError(f"R must have shape [B, T-1, d_r], got {tuple(R.shape)}")
        expected_tail = (self.num_pairs, self.d_r)
        if R.shape[1:] != expected_tail:
            raise ValueError(f"expected R tail shape {expected_tail}, got {tuple(R.shape[1:])}")

        tokens = self.input_norm(R)
        query = self.query.expand(R.shape[0], -1, -1)
        attended, _ = self.attention(query=query, key=tokens, value=tokens, need_weights=False)
        features = torch.cat(
            [
                attended.squeeze(1),
                tokens.mean(dim=1),
                tokens.amax(dim=1),
            ],
            dim=-1,
        )
        return self.net(features)


class PSeriesTrajectoryMatrixModel(nn.Module):
    """P-series trajectory-matrix classifier for frozen CLIP frames."""

    def __init__(
        self,
        clip_dim: int = 512,
        num_frames: int = 16,
        d_z: int = 128,
        d_r: int = 128,
        num_classes: int = 48,
        use_pgm: bool = False,
        alpha: float = 1.0,
        lambda_frame: float = 0.0,
        use_pre_pgm: bool = False,
        pre_pgm_alpha: float = 1.0,
        pre_pgm_lambda_frame: float = 0.0,
        diff_hidden_dim: int = 256,
        diff_dropout: float = 0.25,
        classifier_dropout: float = 0.3,
        classifier_hidden_dim: int = 256,
        classifier_num_heads: int = 4,
        classifier_type: str = "trajectory_matrix_linear",
    ) -> None:
        super().__init__()
        if num_frames < 2:
            raise ValueError("num_frames must be at least 2")
        valid_classifier_types = {"trajectory_matrix_linear", "trajectory_matrix_attention"}
        if classifier_type not in valid_classifier_types:
            raise ValueError(
                f"classifier.type must be one of {sorted(valid_classifier_types)}, "
                f"got {classifier_type!r}"
            )
        self.clip_dim = clip_dim
        self.num_frames = num_frames
        self.d_z = d_z
        self.d_r = d_r
        self.num_classes = num_classes
        self.use_pgm = use_pgm
        self.lambda_frame_value = float(lambda_frame)
        self.use_pre_pgm = use_pre_pgm
        self.pre_lambda_frame_value = float(pre_pgm_lambda_frame)
        self.classifier_type = classifier_type

        self.pre_frame_smoother = GaussianFramePGMSmoother(
            use_pgm=use_pre_pgm,
            alpha=pre_pgm_alpha,
            lambda_frame=pre_pgm_lambda_frame,
        )
        self.projection = TemporalProjection(clip_dim=clip_dim, d_z=d_z)
        self.frame_smoother = GaussianFramePGMSmoother(
            use_pgm=use_pgm,
            alpha=alpha,
            lambda_frame=lambda_frame,
        )
        self.diffnet = ProjectedPairwiseDiffNet(
            d_z=d_z,
            d_r=d_r,
            hidden_dim=diff_hidden_dim,
            dropout=diff_dropout,
        )
        if classifier_type == "trajectory_matrix_attention":
            self.classifier = TrajectoryMatrixAttentionClassifier(
                num_frames=num_frames,
                d_r=d_r,
                num_classes=num_classes,
                hidden_dim=classifier_hidden_dim,
                dropout=classifier_dropout,
                num_heads=classifier_num_heads,
            )
        else:
            self.classifier = TrajectoryMatrixClassifier(
                num_frames=num_frames,
                d_r=d_r,
                num_classes=num_classes,
                dropout=classifier_dropout,
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PSeriesTrajectoryMatrixModel":
        model_config = config.get("model", {})
        pgm_config = config.get("pgm", {})
        pre_pgm_config = config.get("pre_pgm", {})
        diffnet_config = config.get("diffnet", {})
        classifier_config = config.get("classifier", {})
        backbone_config = config.get("backbone", {})
        dataset_config = config.get("dataset", {})

        clip_dim = int(model_config.get("clip_dim", backbone_config.get("input_dim", 512)))
        num_frames = int(model_config.get("num_frames", backbone_config.get("num_frames", 16)))
        d_z = int(model_config.get("d_z", 128))
        d_r = int(model_config.get("d_r", 128))
        num_classes = int(model_config.get("num_classes", dataset_config.get("num_classes", 48)))
        use_pgm = bool(model_config.get("use_pgm", False))
        lambda_frame = float(pgm_config.get("lambda_frame", model_config.get("lambda_frame", 0.0)) or 0.0)
        use_pre_pgm = bool(model_config.get("use_pre_pgm", False))
        pre_pgm_lambda_frame = float(
            pre_pgm_config.get("lambda_frame", model_config.get("pre_lambda_frame", 0.0)) or 0.0
        )
        return cls(
            clip_dim=clip_dim,
            num_frames=num_frames,
            d_z=d_z,
            d_r=d_r,
            num_classes=num_classes,
            use_pgm=use_pgm,
            alpha=float(pgm_config.get("alpha", 1.0)),
            lambda_frame=lambda_frame,
            use_pre_pgm=use_pre_pgm,
            pre_pgm_alpha=float(pre_pgm_config.get("alpha", pgm_config.get("alpha", 1.0))),
            pre_pgm_lambda_frame=pre_pgm_lambda_frame,
            diff_hidden_dim=int(diffnet_config.get("hidden_dim", 256)),
            diff_dropout=float(diffnet_config.get("dropout", 0.25)),
            classifier_dropout=float(classifier_config.get("dropout", 0.3)),
            classifier_hidden_dim=int(classifier_config.get("hidden_dim", 256)),
            classifier_num_heads=int(classifier_config.get("num_heads", 4)),
            classifier_type=str(classifier_config.get("type", "trajectory_matrix_linear")),
        )

    def _validate_input(self, X: torch.Tensor) -> None:
        if X.ndim != 3:
            raise ValueError(f"X must have shape [B, T, clip_dim], got {tuple(X.shape)}")
        if X.shape[1] != self.num_frames:
            raise ValueError(f"expected num_frames={self.num_frames}, got {X.shape[1]}")
        if X.shape[-1] != self.clip_dim:
            raise ValueError(f"expected clip_dim={self.clip_dim}, got {X.shape[-1]}")

    def _frame_pgm_diagnostics(self, U: torch.Tensor, Z: torch.Tensor) -> dict[str, torch.Tensor]:
        delta = Z - U
        return {
            "correction_magnitude": torch.linalg.vector_norm(delta, ord=2, dim=-1).mean(),
            "observation_residual": delta.square().mean(),
            "smoothness_energy": (Z[:, 1:, :] - Z[:, :-1, :]).square().mean(),
        }

    def forward(self, X: torch.Tensor, return_debug: bool = False) -> torch.Tensor | dict[str, Any]:
        self._validate_input(X)
        X_smooth, pre_lambda_frame = self.pre_frame_smoother(X, return_lambda=True)
        U = self.projection(X_smooth)
        Z, lambda_frame = self.frame_smoother(U, return_lambda=True)
        R = self.diffnet(Z)
        logits = self.classifier(R)
        if return_debug:
            pre_pgm_enabled = self.use_pre_pgm and self.pre_lambda_frame_value > 0
            post_pgm_enabled = self.use_pgm and self.lambda_frame_value > 0
            frame_pgm_diagnostics = {}
            if pre_pgm_enabled:
                frame_pgm_diagnostics = self._frame_pgm_diagnostics(X, X_smooth)
            elif post_pgm_enabled:
                frame_pgm_diagnostics = self._frame_pgm_diagnostics(U, Z)
            return {
                "logits": logits,
                "X_smooth": X_smooth,
                "U": U,
                "Z": Z,
                "R": R,
                "lambda_frame": pre_lambda_frame if pre_pgm_enabled else lambda_frame,
                "pre_lambda_frame": pre_lambda_frame,
                "post_lambda_frame": lambda_frame,
                "frame_pgm_diagnostics": frame_pgm_diagnostics,
                "pre_frame_pgm_diagnostics": (
                    self._frame_pgm_diagnostics(X, X_smooth) if pre_pgm_enabled else {}
                ),
                "post_frame_pgm_diagnostics": (
                    self._frame_pgm_diagnostics(U, Z) if post_pgm_enabled else {}
                ),
                "ablation_id": "P-prePGM" if pre_pgm_enabled else ("P-PGM" if post_pgm_enabled else "P-noPGM"),
                "model_variant": (
                    "p_traj_pre_pgm" if pre_pgm_enabled else ("p_traj_pgm" if post_pgm_enabled else "p_traj_no_pgm")
                ),
                "classifier_type": self.classifier_type,
            }
        return logits


# Backward-compatible alias for the first local P-series implementation.
PSeriesTrajectoryModel = PSeriesTrajectoryMatrixModel

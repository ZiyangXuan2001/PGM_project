"""Controlled frozen-feature DiffNet/PGM ablations for Diving48 V2."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .classifier import (
    AttentionPooledInformationMatrixClassifier,
    MeanPooledInformationMatrixClassifier,
    MLPClassifier,
    TemporalEvidenceInformationClassifier,
)
from .gaussian_pgm_smoother import GaussianPGMSmoother
from .information_matrix_accumulator import InformationMatrixAccumulator
from .pairwise_diff_net import PairwiseDiffNet, SimpleConcatPairwiseDiffNet


SUPPORTED_BACKBONES = {
    "precomputed_clip_vit_b16",
    "precomputed_resnet50",
    "precomputed_resnet50_layer3",
    "precomputed_resnet50_layer4",
    # Future documentation-only options. Feature extraction is not implemented here.
    "dinov2_s14_frame",
    "clip_vit_b32_frame",
    "clip_vit_b16_frame",
    "resnet50_frame",
}

SUPPORTED_DIFF_NET_TYPES = {"pairwise_diff_net", "simple_concat_pairwise"}


ABLATION_VARIANTS: dict[str, dict[str, str]] = {
    "E0": {
        "variant": "feature_mean",
        "purpose": "Frozen feature mean baseline with no DiffNet, PGM, or accumulator.",
    },
    "E1": {
        "variant": "diff_mean",
        "purpose": "DiffNet temporal observations with mean pooling.",
    },
    "E1.5": {
        "variant": "diff_info_accum",
        "purpose": "DiffNet temporal observations with the accumulator but no PGM.",
    },
    "E2": {
        "variant": "diff_pgm_mean",
        "purpose": "DiffNet observations smoothed by Gaussian PGM and mean pooled.",
    },
    "E3": {
        "variant": "diff_pgm_info_accum",
        "purpose": "PGM evidence read by the learned InformationMatrixAccumulator.",
    },
}

VARIANT_TO_ABLATION = {value["variant"]: key for key, value in ABLATION_VARIANTS.items()}

LEGACY_VARIANT_ALIASES = {
    "mean_pool_baseline": "feature_mean",
    "diff_only": "diff_mean",
    "diff_pgm": "diff_pgm_mean",
    "diff_pgm_info": "diff_pgm_info_accum",
    "diff_pgm_info_attention": "diff_pgm_info_accum",
}

LEGACY_ABLATION_ALIASES = {"E4": "E3"}


def canonical_variant(model_variant: str) -> str:
    return LEGACY_VARIANT_ALIASES.get(model_variant, model_variant)


def canonical_ablation_id(ablation_id: str) -> str:
    return LEGACY_ABLATION_ALIASES.get(ablation_id, ablation_id)


class CLIPMeanPoolBaseline(nn.Module):
    """Simple CLIP embedding baseline: mean-pool frames, then classify."""

    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 48,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.classifier = MLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 3:
            raise ValueError(f"X must have shape [B, T, d_x], got {tuple(X.shape)}")
        if X.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {X.shape[-1]}")
        return self.classifier(X.mean(dim=1))


class EmbeddingDifferencePGMModel(nn.Module):
    """Frozen features -> DiffNet/PGM/accumulator ablation head.

    DiffNet produces pairwise temporal observations ``R_t``. The Gaussian PGM
    smoother treats ``R_t`` as noisy observations and performs MAP inference
    over latent clean temporal states ``Y_t``. The learned accumulator is not
    the analytic Gaussian information matrix ``A = alpha I + lambda L``; it is
    a small sequential evidence reader. E1.5 controls for the accumulator
    alone, while E3 tests PGM evidence beyond that accumulator.
    """

    def __init__(
        self,
        input_dim: int = 512,
        d_y: int = 128,
        K: int = 8,
        d_h: int = 128,
        num_classes: int = 48,
        pair_hidden: int = 256,
        diff_dropout: float = 0.1,
        pgm_type: str = "gaussian_chain",
        lambda_smooth: float = 1.0,
        use_alpha: bool = True,
        eta: float = 1.0,
        classifier_type: str = "mlp",
        classifier_hidden: int = 256,
        classifier_dropout: float = 0.2,
        attention_heads: int = 4,
        temporal_layers: int = 1,
        frame_pgm_type: str = "none",
        frame_lambda_smooth: float = 1.0,
        backbone_name: str = "precomputed_clip_vit_b16",
        feature_format: str = "vector",
        spatial_tokens: int = 49,
        spatial_token_dim: int = 64,
        diff_net_type: str = "pairwise_diff_net",
        model_variant: str = "diff_pgm_info_accum",
        ablation_id: str = "E3",
    ) -> None:
        super().__init__()
        model_variant = canonical_variant(model_variant)
        ablation_id = canonical_ablation_id(ablation_id)
        if model_variant not in VARIANT_TO_ABLATION:
            valid = ", ".join(sorted(VARIANT_TO_ABLATION))
            raise ValueError(f"unsupported model variant {model_variant!r}; valid options: {valid}")
        expected_ablation = VARIANT_TO_ABLATION[model_variant]
        if ablation_id != expected_ablation:
            raise ValueError(
                f"ablation_id={ablation_id!r} does not match model_variant={model_variant!r}; "
                f"expected {expected_ablation!r}"
            )
        if backbone_name not in SUPPORTED_BACKBONES:
            valid = ", ".join(sorted(SUPPORTED_BACKBONES))
            raise ValueError(f"unsupported backbone {backbone_name!r}; valid options: {valid}")
        if backbone_name not in {
            "precomputed_clip_vit_b16",
            "precomputed_resnet50",
            "precomputed_resnet50_layer3",
            "precomputed_resnet50_layer4",
        }:
            raise NotImplementedError(
                f"{backbone_name} is documented as a future backbone option; "
                "only precomputed frame inputs are implemented now."
            )
        if feature_format not in {"vector", "spatial_map"}:
            raise ValueError(f"feature_format must be 'vector' or 'spatial_map', got {feature_format!r}")
        if feature_format == "spatial_map" and spatial_tokens <= 0:
            raise ValueError("spatial_tokens must be positive for spatial_map inputs")
        if pgm_type not in {"none", "gaussian_chain", "learnable_gaussian_chain"}:
            raise ValueError(
                "pgm_smoother.type must be one of: none, gaussian_chain, "
                f"learnable_gaussian_chain; got {pgm_type!r}"
            )
        if frame_pgm_type not in {"none", "gaussian_chain", "learnable_gaussian_chain"}:
            raise ValueError(
                "frame_pgm_smoother.type must be one of: none, gaussian_chain, "
                f"learnable_gaussian_chain; got {frame_pgm_type!r}"
            )
        if classifier_type not in {"mlp", "attention_pool", "temporal_evidence_attention"}:
            raise ValueError(
                "classifier.type must be 'mlp', 'attention_pool', or "
                f"'temporal_evidence_attention', got {classifier_type!r}"
            )
        if diff_net_type not in SUPPORTED_DIFF_NET_TYPES:
            valid = ", ".join(sorted(SUPPORTED_DIFF_NET_TYPES))
            raise ValueError(f"unsupported diff_net_type {diff_net_type!r}; valid options: {valid}")

        self.input_dim = input_dim
        self.D = input_dim
        self.d_y = d_y
        self.K = K
        self.d_h = d_h
        self.num_classes = num_classes
        self.pgm_type = pgm_type
        self.frame_pgm_type = frame_pgm_type
        self.classifier_type = classifier_type
        self.backbone_name = backbone_name
        self.feature_format = feature_format
        self.spatial_tokens = spatial_tokens
        self.spatial_token_dim = spatial_token_dim
        self.diff_net_type = diff_net_type
        self.model_variant = model_variant
        self.ablation_id = ablation_id
        self.pairwise_input_dim = spatial_token_dim if feature_format == "spatial_map" else input_dim
        self.uses_diff_net = model_variant != "feature_mean"
        self.uses_pgm = model_variant in {"diff_pgm_mean", "diff_pgm_info_accum"}
        self.uses_accumulator = model_variant in {"diff_info_accum", "diff_pgm_info_accum"}
        self.uses_pgm_evidence = model_variant == "diff_pgm_info_accum"
        self.evidence_dim = 3 * d_y + 2
        self.accumulator_input_dim = self.evidence_dim if self.uses_pgm_evidence else d_y
        self.uses_frame_pgm = frame_pgm_type != "none"

        self.spatial_projector = None
        if feature_format == "spatial_map":
            self.spatial_projector = nn.Sequential(
                nn.Linear(input_dim, spatial_token_dim),
                nn.GELU(),
                nn.LayerNorm(spatial_token_dim),
                nn.Dropout(diff_dropout),
            )

        self.frame_smoother = None
        if self.uses_frame_pgm:
            self.frame_smoother = GaussianPGMSmoother(
                lambda_init=frame_lambda_smooth,
                learnable_lambda=frame_pgm_type == "learnable_gaussian_chain",
            )

        self.pairwise_diff: nn.Module | None = None
        if diff_net_type == "simple_concat_pairwise":
            if self.uses_diff_net:
                self.pairwise_diff = SimpleConcatPairwiseDiffNet(
                    D=self.pairwise_input_dim,
                    d_y=d_y,
                    hidden_dim=pair_hidden,
                    dropout=diff_dropout,
                )
        elif self.uses_diff_net:
            self.pairwise_diff = PairwiseDiffNet(
                D=self.pairwise_input_dim,
                d_y=d_y,
                pair_hidden=pair_hidden,
                dropout=diff_dropout,
            )

        self.baseline_classifier: MLPClassifier | None = None
        if model_variant == "feature_mean":
            self.baseline_classifier = MLPClassifier(
                input_dim=self.pairwise_input_dim,
                num_classes=num_classes,
                hidden_dim=classifier_hidden,
                dropout=classifier_dropout,
            )

        self.smoother = None
        if self.uses_pgm:
            self.smoother = GaussianPGMSmoother(
                lambda_init=lambda_smooth,
                learnable_lambda=pgm_type == "learnable_gaussian_chain",
            )

        self.diff_pool_classifier: MLPClassifier | None = None
        if model_variant in {"diff_mean", "diff_pgm_mean"}:
            self.diff_pool_classifier = MLPClassifier(
                input_dim=d_y,
                num_classes=num_classes,
                hidden_dim=classifier_hidden,
                dropout=classifier_dropout,
            )

        self.accumulator: InformationMatrixAccumulator | None = None
        self.info_classifier: nn.Module | None = None
        if self.uses_accumulator:
            self.accumulator = InformationMatrixAccumulator(
                d_y=self.accumulator_input_dim,
                K=K,
                d_h=d_h,
                eta=eta,
                use_alpha=use_alpha,
                normalize_delta=True,
            )
            if classifier_type == "attention_pool":
                self.info_classifier = AttentionPooledInformationMatrixClassifier(
                    d_h=d_h,
                    num_classes=num_classes,
                    classifier_hidden=classifier_hidden,
                    dropout=classifier_dropout,
                    num_heads=attention_heads,
                )
            elif classifier_type == "temporal_evidence_attention":
                self.info_classifier = TemporalEvidenceInformationClassifier(
                    evidence_dim=self.accumulator_input_dim,
                    d_h=d_h,
                    num_classes=num_classes,
                    classifier_hidden=classifier_hidden,
                    dropout=classifier_dropout,
                    num_heads=attention_heads,
                    temporal_layers=temporal_layers,
                )
            else:
                self.info_classifier = MeanPooledInformationMatrixClassifier(
                    d_h=d_h,
                    num_classes=num_classes,
                    classifier_hidden=classifier_hidden,
                    dropout=classifier_dropout,
                )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EmbeddingDifferencePGMModel":
        """Build the controlled model from ``configs/default.yaml`` style config."""

        dataset_config = config.get("dataset", {})
        model_config = config.get("model", {})
        backbone_config = config.get("backbone", {})
        diff_config = config.get("diff_nn", {})
        pgm_config = config.get("pgm_smoother", {})
        frame_pgm_config = config.get("frame_pgm_smoother", {})
        info_config = config.get("information_matrix", {})
        classifier_config = config.get("classifier", {})

        diff_net_type = str(diff_config.get("diff_net_type", diff_config.get("type", "pairwise_diff_net")))
        if diff_net_type not in SUPPORTED_DIFF_NET_TYPES:
            valid = ", ".join(sorted(SUPPORTED_DIFF_NET_TYPES))
            raise ValueError(f"diff_nn.diff_net_type must be one of: {valid}; got {diff_net_type!r}")
        info_type = info_config.get("type", "accumulator")
        if info_type != "accumulator":
            raise ValueError(f'information_matrix.type must be "accumulator", got {info_type!r}')

        model_variant = canonical_variant(str(model_config.get("variant", "diff_pgm_info_accum")))
        ablation_id = canonical_ablation_id(str(model_config.get("ablation_id", VARIANT_TO_ABLATION.get(model_variant, "E3"))))
        if model_variant in {"feature_mean", "diff_mean", "diff_info_accum"}:
            pgm_type = "none"
        elif model_variant in {"diff_pgm_mean", "diff_pgm_info_accum"}:
            pgm_type = str(pgm_config.get("type", "gaussian_chain"))
        else:
            pgm_type = str(pgm_config.get("type", "gaussian_chain"))

        classifier_type = str(classifier_config.get("type", "mlp"))
        lambda_config = pgm_config.get("lambda_smooth", 1.0)
        lambda_smooth = 1.0 if lambda_config is None else float(lambda_config)
        frame_lambda_config = frame_pgm_config.get("lambda_smooth", 1.0)
        frame_lambda_smooth = 1.0 if frame_lambda_config is None else float(frame_lambda_config)

        return cls(
            input_dim=int(backbone_config.get("input_dim", 512)),
            d_y=int(diff_config.get("d_y", 128)),
            K=int(info_config.get("K", 8)),
            d_h=int(info_config.get("d_h", 128)),
            num_classes=int(dataset_config.get("num_classes", classifier_config.get("num_classes", 48))),
            pair_hidden=int(diff_config.get("hidden_dim", 256)),
            diff_dropout=float(diff_config.get("dropout", 0.1)),
            pgm_type=pgm_type,
            lambda_smooth=lambda_smooth,
            use_alpha=bool(info_config.get("use_alpha", True)),
            eta=float(info_config.get("eta", 1.0)),
            classifier_type=classifier_type,
            classifier_hidden=int(classifier_config.get("hidden_dim", 256)),
            classifier_dropout=float(classifier_config.get("dropout", 0.2)),
            attention_heads=int(classifier_config.get("num_heads", 4)),
            temporal_layers=int(classifier_config.get("temporal_layers", 1)),
            frame_pgm_type=str(frame_pgm_config.get("type", "none")),
            frame_lambda_smooth=frame_lambda_smooth,
            backbone_name=str(backbone_config.get("name", "precomputed_clip_vit_b16")),
            feature_format=str(backbone_config.get("feature_format", "vector")),
            spatial_tokens=int(backbone_config.get("spatial_tokens", 49)),
            spatial_token_dim=int(backbone_config.get("spatial_token_dim", min(128, int(diff_config.get("d_y", 128))))),
            diff_net_type=diff_net_type,
            model_variant=model_variant,
            ablation_id=ablation_id,
        )

    def _validate_input(self, X: torch.Tensor) -> None:
        if self.feature_format == "vector":
            if X.ndim != 3:
                raise ValueError(f"X must have shape [B, T, d_x], got {tuple(X.shape)}")
            if X.shape[-1] != self.input_dim:
                raise ValueError(f"expected input_dim={self.input_dim}, got {X.shape[-1]}")
            if X.shape[1] < 2:
                raise ValueError("X must contain at least two frames")
            return

        if X.ndim != 4:
            raise ValueError(f"spatial_map X must have shape [B, T, S, d_x], got {tuple(X.shape)}")
        if X.shape[2] != self.spatial_tokens:
            raise ValueError(f"expected spatial_tokens={self.spatial_tokens}, got {X.shape[2]}")
        if X.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {X.shape[-1]}")
        if X.shape[1] < 2:
            raise ValueError("X must contain at least two frames")

    def _prepare_frame_sequence(self, X: torch.Tensor) -> torch.Tensor:
        if self.feature_format == "vector":
            return X
        assert self.spatial_projector is not None
        Z = self.spatial_projector(X)
        return Z

    def _frame_pgm_diagnostics(self, observed: torch.Tensor, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        delta = latent - observed
        if latent.shape[1] > 1:
            smoothness_energy = (latent[:, 1:, ...] - latent[:, :-1, ...]).square().mean()
        else:
            smoothness_energy = latent.new_zeros(())
        return {
            "correction_magnitude": torch.linalg.vector_norm(delta, ord=2, dim=-1).mean(),
            "observation_residual": delta.square().mean(),
            "smoothness_energy": smoothness_energy,
        }

    def _apply_frame_pgm(
        self,
        prepared: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        if self.frame_smoother is None:
            return prepared, None, {}
        if prepared.ndim == 3:
            latent, lambda_smooth = self.frame_smoother(prepared, return_lambda=True)
            return latent, lambda_smooth, self._frame_pgm_diagnostics(prepared, latent)

        batch_size, num_frames, num_tokens, token_dim = prepared.shape
        token_sequences = prepared.permute(0, 2, 1, 3).reshape(batch_size * num_tokens, num_frames, token_dim)
        latent_tokens, lambda_smooth = self.frame_smoother(token_sequences, return_lambda=True)
        latent = latent_tokens.reshape(batch_size, num_tokens, num_frames, token_dim).permute(0, 2, 1, 3)
        return latent, lambda_smooth, self._frame_pgm_diagnostics(prepared, latent)

    def _run_pairwise_diff(self, prepared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.pairwise_diff is None:
            raise RuntimeError("pairwise_diff is only available for DiffNet variants")
        if self.feature_format == "vector":
            return self.pairwise_diff(prepared), None

        if self.diff_net_type == "simple_concat_pairwise":
            token_R = self.pairwise_diff(prepared)
            return token_R.mean(dim=2), token_R

        batch_size, num_frames, num_tokens, token_dim = prepared.shape
        token_sequences = prepared.permute(0, 2, 1, 3).reshape(batch_size * num_tokens, num_frames, token_dim)
        token_R = self.pairwise_diff(token_sequences)
        token_R = token_R.reshape(batch_size, num_tokens, num_frames - 1, self.d_y).permute(0, 2, 1, 3)
        return token_R.mean(dim=2), token_R

    def _build_pgm_evidence(self, R: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Build local PGM evidence ``[R_t, Y_t, |Y_t-R_t|, residual_t, smoothness_t]``."""

        delta = Y - R
        abs_delta = delta.abs()
        residual = delta.square().mean(dim=-1, keepdim=True)
        smoothness = torch.zeros_like(residual)
        if Y.shape[1] > 1:
            smoothness[:, 1:, :] = (Y[:, 1:, :] - Y[:, :-1, :]).square().mean(dim=-1, keepdim=True)
        return torch.cat([R, Y, abs_delta, residual, smoothness], dim=-1)

    def _pgm_diagnostics(self, R: torch.Tensor, Y: torch.Tensor) -> dict[str, torch.Tensor]:
        delta = Y - R
        if Y.shape[1] > 1:
            smoothness_energy = (Y[:, 1:, :] - Y[:, :-1, :]).square().mean()
        else:
            smoothness_energy = Y.new_zeros(())
        return {
            "correction_magnitude": torch.linalg.vector_norm(delta, ord=2, dim=-1).mean(),
            "observation_residual": delta.square().mean(),
            "smoothness_energy": smoothness_energy,
        }

    def forward(self, X: torch.Tensor, return_debug: bool = False) -> torch.Tensor | dict[str, Any]:
        """Run DiffTraj-PGM on ``[B,T,D]`` vectors or ``[B,T,S,D]`` spatial tokens."""

        self._validate_input(X)
        prepared = self._prepare_frame_sequence(X)
        prepared_observed = prepared
        prepared, frame_lambda_smooth, frame_diagnostics = self._apply_frame_pgm(prepared)
        if self.model_variant == "feature_mean":
            assert self.baseline_classifier is not None
            pooled = prepared.mean(dim=1) if prepared.ndim == 3 else prepared.mean(dim=(1, 2))
            logits = self.baseline_classifier(pooled)
            if return_debug:
                return {
                    "logits": logits,
                    "R": None,
                    "Y": None,
                    "U": None,
                    "H_final": None,
                    "alpha": None,
                    "lambda_smooth": None,
                    "frame_lambda_smooth": frame_lambda_smooth,
                    "pooled": pooled,
                    "pgm_diagnostics": frame_diagnostics,
                    "frame_pgm_diagnostics": frame_diagnostics,
                    "Z": prepared if self.uses_frame_pgm else None,
                    "X_prepared": prepared_observed,
                    "ablation_id": self.ablation_id,
                    "model_variant": self.model_variant,
                    "classifier_type": "mlp",
                    "diff_net_type": self.diff_net_type,
                    "feature_format": self.feature_format,
                }
            return logits

        R, R_tokens = self._run_pairwise_diff(prepared)
        if self.model_variant == "diff_mean":
            assert self.diff_pool_classifier is not None
            pooled = R.mean(dim=1)
            logits = self.diff_pool_classifier(pooled)
            if return_debug:
                return {
                    "logits": logits,
                    "R": R,
                    "Y": None,
                    "U": None,
                    "H_final": None,
                    "alpha": None,
                    "lambda_smooth": None,
                    "frame_lambda_smooth": frame_lambda_smooth,
                    "pooled": pooled,
                    "R_tokens": R_tokens,
                    "pgm_diagnostics": frame_diagnostics,
                    "frame_pgm_diagnostics": frame_diagnostics,
                    "Z": prepared if self.uses_frame_pgm else None,
                    "X_prepared": prepared_observed,
                    "ablation_id": self.ablation_id,
                    "model_variant": self.model_variant,
                    "classifier_type": "mlp",
                    "diff_net_type": self.diff_net_type,
                    "feature_format": self.feature_format,
                }
            return logits

        if self.model_variant == "diff_info_accum":
            assert self.accumulator is not None
            assert self.info_classifier is not None
            H_final, alpha = self.accumulator(R)
            if self.classifier_type == "temporal_evidence_attention":
                logits = self.info_classifier(H_final, R)
            else:
                logits = self.info_classifier(H_final)
            if return_debug:
                return {
                    "logits": logits,
                    "R": R,
                    "Y": None,
                    "U": None,
                    "H_final": H_final,
                    "alpha": alpha,
                    "lambda_smooth": None,
                    "frame_lambda_smooth": frame_lambda_smooth,
                    "pooled": None,
                    "R_tokens": R_tokens,
                    "pgm_diagnostics": frame_diagnostics,
                    "frame_pgm_diagnostics": frame_diagnostics,
                    "Z": prepared if self.uses_frame_pgm else None,
                    "X_prepared": prepared_observed,
                    "ablation_id": self.ablation_id,
                    "model_variant": self.model_variant,
                    "classifier_type": self.classifier_type,
                    "diff_net_type": self.diff_net_type,
                    "feature_format": self.feature_format,
                    "accumulator_input_dim": self.accumulator_input_dim,
                }
            return logits

        if self.model_variant == "diff_pgm_mean":
            assert self.smoother is not None
            assert self.diff_pool_classifier is not None
            Y, lambda_smooth = self.smoother(R, return_lambda=True)
            diagnostics = self._pgm_diagnostics(R, Y)
            pooled = Y.mean(dim=1)
            logits = self.diff_pool_classifier(pooled)
            if return_debug:
                return {
                    "logits": logits,
                    "R": R,
                    "Y": Y,
                    "U": None,
                    "H_final": None,
                    "alpha": None,
                    "lambda_smooth": lambda_smooth,
                    "frame_lambda_smooth": frame_lambda_smooth,
                    "pooled": pooled,
                    "R_tokens": R_tokens,
                    "pgm_diagnostics": diagnostics,
                    "frame_pgm_diagnostics": frame_diagnostics,
                    "Z": prepared if self.uses_frame_pgm else None,
                    "X_prepared": prepared_observed,
                    "ablation_id": self.ablation_id,
                    "model_variant": self.model_variant,
                    "classifier_type": "mlp",
                    "diff_net_type": self.diff_net_type,
                    "feature_format": self.feature_format,
                }
            return logits

        assert self.model_variant == "diff_pgm_info_accum"
        assert self.smoother is not None
        assert self.accumulator is not None
        assert self.info_classifier is not None
        Y, lambda_smooth = self.smoother(R, return_lambda=True)
        U = self._build_pgm_evidence(R, Y)
        H_final, alpha = self.accumulator(U)
        if self.classifier_type == "temporal_evidence_attention":
            logits = self.info_classifier(H_final, U)
        else:
            logits = self.info_classifier(H_final)
        diagnostics = self._pgm_diagnostics(R, Y)

        if return_debug:
            return {
                "logits": logits,
                "R": R,
                "Y": Y,
                "U": U,
                "H_final": H_final,
                "alpha": alpha,
                "lambda_smooth": lambda_smooth,
                "frame_lambda_smooth": frame_lambda_smooth,
                "R_tokens": R_tokens,
                "pgm_diagnostics": diagnostics,
                "frame_pgm_diagnostics": frame_diagnostics,
                "Z": prepared if self.uses_frame_pgm else None,
                "X_prepared": prepared_observed,
                "pgm_type": self.pgm_type,
                "frame_pgm_type": self.frame_pgm_type,
                "ablation_id": self.ablation_id,
                "model_variant": self.model_variant,
                "classifier_type": self.classifier_type,
                "diff_net_type": self.diff_net_type,
                "feature_format": self.feature_format,
                "accumulator_input_dim": self.accumulator_input_dim,
            }
        return logits

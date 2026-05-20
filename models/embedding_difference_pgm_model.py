"""Controlled CLIP-embedding DiffTraj-PGM model for Diving48 V2."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .classifier import (
    AttentionPooledInformationMatrixClassifier,
    MeanPooledInformationMatrixClassifier,
    MLPClassifier,
)
from .gaussian_pgm_smoother import GaussianPGMSmoother
from .information_matrix_accumulator import InformationMatrixAccumulator
from .pairwise_diff_net import PairwiseDiffNet


SUPPORTED_BACKBONES = {
    "precomputed_clip_vit_b16",
    "precomputed_resnet50",
    "precomputed_resnet50_layer4",
    # Future documentation-only options. Feature extraction is not implemented here.
    "dinov2_s14_frame",
    "clip_vit_b32_frame",
    "clip_vit_b16_frame",
    "resnet50_frame",
}


ABLATION_VARIANTS: dict[str, dict[str, str]] = {
    "E0": {
        "variant": "mean_pool_baseline",
        "purpose": "Basic frozen CLIP frame-embedding baseline with no explicit temporal modeling.",
    },
    "E1": {
        "variant": "diff_only",
        "purpose": "Test whether learned adjacent-frame difference embeddings help.",
    },
    "E2": {
        "variant": "diff_pgm",
        "purpose": "Test whether Gaussian-chain PGM smoothing improves temporal difference features.",
    },
    "E3": {
        "variant": "diff_pgm_info",
        "purpose": "Test whether information matrix accumulation improves over pooled smoothed temporal features.",
    },
    "E4": {
        "variant": "diff_pgm_info_attention",
        "purpose": "Test whether attention pooling over the information matrix improves over simple mean pooling.",
    },
}

VARIANT_TO_ABLATION = {value["variant"]: key for key, value in ABLATION_VARIANTS.items()}


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
    """Backbone -> PairwiseDiffNet -> PGM smoother -> information matrix -> classifier.

    The implemented input path assumes precomputed OpenAI CLIP ViT-B/16 frame
    embeddings with shape ``X: [B, T, 512]``.
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
        classifier_type: str = "attention_pool",
        classifier_hidden: int = 256,
        classifier_dropout: float = 0.2,
        attention_heads: int = 4,
        backbone_name: str = "precomputed_clip_vit_b16",
        feature_format: str = "vector",
        spatial_tokens: int = 49,
        spatial_token_dim: int = 64,
        model_variant: str = "diff_pgm_info_attention",
        ablation_id: str = "E4",
    ) -> None:
        super().__init__()
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
        if backbone_name not in {"precomputed_clip_vit_b16", "precomputed_resnet50", "precomputed_resnet50_layer4"}:
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
        if classifier_type not in {"mlp", "attention_pool"}:
            raise ValueError(
                f"classifier.type must be 'mlp' or 'attention_pool', got {classifier_type!r}"
            )

        self.input_dim = input_dim
        self.D = input_dim
        self.d_y = d_y
        self.K = K
        self.d_h = d_h
        self.num_classes = num_classes
        self.pgm_type = pgm_type
        self.classifier_type = classifier_type
        self.backbone_name = backbone_name
        self.feature_format = feature_format
        self.spatial_tokens = spatial_tokens
        self.spatial_token_dim = spatial_token_dim
        self.model_variant = model_variant
        self.ablation_id = ablation_id
        self.pairwise_input_dim = spatial_token_dim if feature_format == "spatial_map" else input_dim

        self.spatial_projector = None
        if feature_format == "spatial_map":
            self.spatial_projector = nn.Sequential(
                nn.Linear(input_dim, spatial_token_dim),
                nn.GELU(),
                nn.LayerNorm(spatial_token_dim),
                nn.Dropout(diff_dropout),
            )

        self.pairwise_diff = PairwiseDiffNet(
            D=self.pairwise_input_dim,
            d_y=d_y,
            pair_hidden=pair_hidden,
            dropout=diff_dropout,
        )
        self.baseline_classifier = MLPClassifier(
            input_dim=self.pairwise_input_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden,
            dropout=classifier_dropout,
        )
        self.smoother = None
        if model_variant in {"diff_pgm", "diff_pgm_info", "diff_pgm_info_attention"}:
            self.smoother = GaussianPGMSmoother(
                lambda_init=lambda_smooth,
                learnable_lambda=pgm_type == "learnable_gaussian_chain",
            )
        self.diff_pool_classifier = MLPClassifier(
            input_dim=d_y,
            num_classes=num_classes,
            hidden_dim=classifier_hidden,
            dropout=classifier_dropout,
        )
        self.accumulator = InformationMatrixAccumulator(
            d_y=d_y,
            K=K,
            d_h=d_h,
            eta=eta,
            use_alpha=use_alpha,
            normalize_delta=True,
        )
        if classifier_type == "mlp":
            self.info_classifier = MeanPooledInformationMatrixClassifier(
                d_h=d_h,
                num_classes=num_classes,
                classifier_hidden=classifier_hidden,
                dropout=classifier_dropout,
            )
        else:
            self.info_classifier = AttentionPooledInformationMatrixClassifier(
                d_h=d_h,
                num_classes=num_classes,
                classifier_hidden=classifier_hidden,
                dropout=classifier_dropout,
                num_heads=attention_heads,
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EmbeddingDifferencePGMModel":
        """Build the controlled model from ``configs/default.yaml`` style config."""

        dataset_config = config.get("dataset", {})
        model_config = config.get("model", {})
        backbone_config = config.get("backbone", {})
        diff_config = config.get("diff_nn", {})
        pgm_config = config.get("pgm_smoother", {})
        info_config = config.get("information_matrix", {})
        classifier_config = config.get("classifier", {})

        diff_type = diff_config.get("type", "pairwise_diff_net")
        if diff_type != "pairwise_diff_net":
            raise ValueError(f'diff_nn.type must be "pairwise_diff_net", got {diff_type!r}')
        info_type = info_config.get("type", "accumulator")
        if info_type != "accumulator":
            raise ValueError(f'information_matrix.type must be "accumulator", got {info_type!r}')

        model_variant = str(model_config.get("variant", "diff_pgm_info_attention"))
        ablation_id = str(model_config.get("ablation_id", VARIANT_TO_ABLATION.get(model_variant, "E4")))
        if model_variant in {"mean_pool_baseline", "diff_only"}:
            pgm_type = "none"
        elif model_variant in {"diff_pgm", "diff_pgm_info", "diff_pgm_info_attention"}:
            pgm_type = str(pgm_config.get("type", "gaussian_chain"))
        else:
            pgm_type = str(pgm_config.get("type", "gaussian_chain"))

        if model_variant == "diff_pgm_info_attention":
            classifier_type = "attention_pool"
        else:
            classifier_type = "mlp"
        lambda_config = pgm_config.get("lambda_smooth", 1.0)
        lambda_smooth = 1.0 if lambda_config is None else float(lambda_config)

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
            backbone_name=str(backbone_config.get("name", "precomputed_clip_vit_b16")),
            feature_format=str(backbone_config.get("feature_format", "vector")),
            spatial_tokens=int(backbone_config.get("spatial_tokens", 49)),
            spatial_token_dim=int(backbone_config.get("spatial_token_dim", min(128, int(diff_config.get("d_y", 128))))),
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

    def _run_pairwise_diff(self, prepared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.feature_format == "vector":
            return self.pairwise_diff(prepared), None

        batch_size, num_frames, num_tokens, token_dim = prepared.shape
        token_sequences = prepared.permute(0, 2, 1, 3).reshape(batch_size * num_tokens, num_frames, token_dim)
        token_R = self.pairwise_diff(token_sequences)
        token_R = token_R.reshape(batch_size, num_tokens, num_frames - 1, self.d_y).permute(0, 2, 1, 3)
        return token_R.mean(dim=2), token_R

    def forward(self, X: torch.Tensor, return_debug: bool = False) -> torch.Tensor | dict[str, Any]:
        """Run DiffTraj-PGM on ``[B,T,D]`` vectors or ``[B,T,S,D]`` spatial tokens."""

        self._validate_input(X)
        prepared = self._prepare_frame_sequence(X)
        if self.model_variant == "mean_pool_baseline":
            pooled = prepared.mean(dim=1) if prepared.ndim == 3 else prepared.mean(dim=(1, 2))
            logits = self.baseline_classifier(pooled)
            if return_debug:
                return {
                    "logits": logits,
                    "R": None,
                    "Y": None,
                    "H_final": None,
                    "alpha": None,
                    "lambda_smooth": None,
                    "pooled": pooled,
                    "ablation_id": self.ablation_id,
                    "model_variant": self.model_variant,
                    "classifier_type": "mlp",
                    "feature_format": self.feature_format,
                }
            return logits

        R, R_tokens = self._run_pairwise_diff(prepared)
        if self.model_variant in {"diff_only", "diff_pgm"}:
            if self.smoother is None:
                Y = R
                lambda_smooth = None
            else:
                Y, lambda_smooth = self.smoother(R, return_lambda=True)
            pooled = Y.mean(dim=1)
            logits = self.diff_pool_classifier(pooled)
            if return_debug:
                return {
                    "logits": logits,
                    "R": R,
                    "Y": Y,
                    "H_final": None,
                    "alpha": None,
                    "lambda_smooth": lambda_smooth,
                    "pooled": pooled,
                    "R_tokens": R_tokens,
                    "ablation_id": self.ablation_id,
                    "model_variant": self.model_variant,
                    "classifier_type": "mlp",
                    "feature_format": self.feature_format,
                }
            return logits

        if self.smoother is None:
            Y = R
            lambda_smooth = None
        else:
            Y, lambda_smooth = self.smoother(R, return_lambda=True)
        H_final, alpha = self.accumulator(Y)
        logits = self.info_classifier(H_final)

        if return_debug:
            return {
                "logits": logits,
                "R": R,
                "Y": Y,
                "H_final": H_final,
                "alpha": alpha,
                "lambda_smooth": lambda_smooth,
                "R_tokens": R_tokens,
                "pgm_type": self.pgm_type,
                "ablation_id": self.ablation_id,
                "model_variant": self.model_variant,
                "classifier_type": self.classifier_type,
                "feature_format": self.feature_format,
            }
        return logits

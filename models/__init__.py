"""Model components for controlled CLIP-embedding DiffTraj-PGM reasoning."""

from .classifier import (
    AttentionPooledInformationMatrixClassifier,
    InformationMatrixClassifier,
    MeanPooledInformationMatrixClassifier,
    MLPClassifier,
)
from .clip_frame_encoder import CLIPFrameEncoder
from .embedding_difference_pgm_model import CLIPMeanPoolBaseline, EmbeddingDifferencePGMModel
from .gaussian_pgm_smoother import GaussianPGMSmoother
from .information_matrix_accumulator import InformationMatrixAccumulator
from .pairwise_diff_net import PairwiseDiffNet, SimpleConcatPairwiseDiffNet
from .p_series_trajectory_matrix import (
    GaussianFramePGMSmoother,
    PSeriesTrajectoryMatrixModel,
    PSeriesTrajectoryModel,
    ProjectedPairwiseDiffNet,
    TemporalProjection,
    TrajectoryMatrixAttentionClassifier,
    TrajectoryMatrixClassifier,
)
from .resnet_frame_encoder import ResNetFrameEncoder

__all__ = [
    "CLIPFrameEncoder",
    "PairwiseDiffNet",
    "SimpleConcatPairwiseDiffNet",
    "TemporalProjection",
    "GaussianFramePGMSmoother",
    "ProjectedPairwiseDiffNet",
    "TrajectoryMatrixClassifier",
    "TrajectoryMatrixAttentionClassifier",
    "PSeriesTrajectoryMatrixModel",
    "PSeriesTrajectoryModel",
    "GaussianPGMSmoother",
    "InformationMatrixAccumulator",
    "InformationMatrixClassifier",
    "MeanPooledInformationMatrixClassifier",
    "AttentionPooledInformationMatrixClassifier",
    "MLPClassifier",
    "CLIPMeanPoolBaseline",
    "EmbeddingDifferencePGMModel",
    "ResNetFrameEncoder",
]

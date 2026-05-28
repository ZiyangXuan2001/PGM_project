"""Compatibility imports for the P-series trajectory-matrix model."""

from .p_series_trajectory_matrix import (
    GaussianFramePGMSmoother,
    PSeriesTrajectoryMatrixModel,
    PSeriesTrajectoryModel,
    ProjectedPairwiseDiffNet,
    TemporalProjection,
    TrajectoryMatrixClassifier,
)

__all__ = [
    "TemporalProjection",
    "GaussianFramePGMSmoother",
    "ProjectedPairwiseDiffNet",
    "TrajectoryMatrixClassifier",
    "PSeriesTrajectoryMatrixModel",
    "PSeriesTrajectoryModel",
]

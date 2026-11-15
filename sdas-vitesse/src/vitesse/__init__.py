"""Projet SDAS : estimation robuste de la vitesse v(x) sur un troncon."""
from vitesse.basis import LegendreBasis, MonomialBasis, PolynomialBasis
from vitesse.estimators import Estimator, OLSEstimator
from vitesse.evaluation import MonteCarlo, MonteCarloResult, cramer_rao_covariance
from vitesse.noise import (
    ContaminatedGaussianNoise,
    GaussianNoise,
    NoiseModel,
    VehicleEffectNoise,
)
from vitesse.profile import SpeedProfile
from vitesse.simulator import (
    ClusteredPositionSampler,
    Measurements,
    MeasurementSimulator,
    PositionSampler,
    UniformPositionSampler,
)

__all__ = [
    "ClusteredPositionSampler", "ContaminatedGaussianNoise", "Estimator",
    "GaussianNoise", "LegendreBasis", "MeasurementSimulator", "Measurements",
    "MonomialBasis", "MonteCarlo", "MonteCarloResult", "NoiseModel",
    "OLSEstimator", "PolynomialBasis", "PositionSampler", "SpeedProfile",
    "UniformPositionSampler", "VehicleEffectNoise", "cramer_rao_covariance",
]

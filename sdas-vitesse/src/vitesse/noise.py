"""Modeles de bruit. Chacun tire un vecteur de perturbations de taille n."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class NoiseModel(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int,
               vehicle_ids: np.ndarray | None = None) -> np.ndarray:
        """Tire n perturbations. `vehicle_ids` sert aux bruits correles par vehicule."""


class GaussianNoise(NoiseModel):
    """Cas simple : eps_k ~ N(0, sigma^2) i.i.d."""

    def __init__(self, sigma: float) -> None:
        if sigma < 0:
            raise ValueError("sigma doit etre >= 0")
        self.sigma = float(sigma)

    def sample(self, rng, n, vehicle_ids=None):
        return self.sigma * rng.standard_normal(n)


class ContaminatedGaussianNoise(NoiseModel):
    """Cas difficile (aberrants) : N(0, sigma^2) avec proba 1-p,
    N(0, outlier_sigma^2) avec proba p (arrets, bouchons, erreurs capteur)."""

    def __init__(self, sigma: float, outlier_rate: float, outlier_sigma: float) -> None:
        if not 0.0 <= outlier_rate <= 1.0:
            raise ValueError("outlier_rate doit etre dans [0, 1]")
        if sigma < 0 or outlier_sigma < 0:
            raise ValueError("les ecarts-types doivent etre >= 0")
        self.sigma = float(sigma)
        self.outlier_rate = float(outlier_rate)
        self.outlier_sigma = float(outlier_sigma)

    def sample(self, rng, n, vehicle_ids=None):
        is_outlier = rng.random(n) < self.outlier_rate
        scale = np.where(is_outlier, self.outlier_sigma, self.sigma)
        return scale * rng.standard_normal(n)


class VehicleEffectNoise(NoiseModel):
    """Cas difficile (correlation) : eps = b_vehicule + e_k, b ~ N(0, sigma_v^2) commun
    a tous les envois d'un meme vehicule (style de conduite), e_k ~ N(0, sigma^2)."""

    def __init__(self, sigma: float, sigma_vehicle: float) -> None:
        if sigma < 0 or sigma_vehicle < 0:
            raise ValueError("les ecarts-types doivent etre >= 0")
        self.sigma = float(sigma)
        self.sigma_vehicle = float(sigma_vehicle)

    def sample(self, rng, n, vehicle_ids=None):
        if vehicle_ids is None:
            raise ValueError("VehicleEffectNoise necessite vehicle_ids")
        ids = np.asarray(vehicle_ids)
        _, inverse = np.unique(ids, return_inverse=True)
        offsets = self.sigma_vehicle * rng.standard_normal(inverse.max() + 1)
        return offsets[inverse] + self.sigma * rng.standard_normal(n)

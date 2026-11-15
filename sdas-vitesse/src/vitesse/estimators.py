"""Estimateurs de v(x). Chacun renvoie un SpeedProfile estime."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from vitesse.basis import MonomialBasis, PolynomialBasis
from vitesse.profile import SpeedProfile
from vitesse.simulator import Measurements


class Estimator(ABC):
    def __init__(self, degree: int, length: float,
                 basis: PolynomialBasis | None = None) -> None:
        if degree < 0:
            raise ValueError("degree doit etre >= 0")
        if length <= 0:
            raise ValueError("length doit etre > 0")
        self.degree = degree
        self.length = length
        self.basis = basis or MonomialBasis()

    def design_matrix(self, x_m: np.ndarray) -> np.ndarray:
        xi = 2.0 * np.asarray(x_m, dtype=float) / self.length - 1.0
        return self.basis.design_matrix(xi, self.degree)

    def fit(self, data: Measurements) -> SpeedProfile:
        if data.n_points < self.degree + 1:
            raise ValueError("pas assez de points : il faut au moins d+1 mesures")
        coef = self._solve(self.design_matrix(data.x_m), data.v)
        return SpeedProfile(coef, self.length, self.basis)

    @abstractmethod
    def _solve(self, A: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Coefficients estimes a partir de la matrice de conception A et des vitesses v."""


class OLSEstimator(Estimator):
    """Moindres carres ordinaires (= maximum de vraisemblance si bruit gaussien i.i.d.)."""

    def _solve(self, A, v):
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return coef

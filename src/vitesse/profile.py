"""Profil de vitesse v(x) polynomial sur un troncon de longueur L."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vitesse.basis import MonomialBasis, PolynomialBasis


@dataclass(frozen=True)
class SpeedProfile:
    """v(x) = sum_j c_j phi_j(xi),  xi = 2x/L - 1 in [-1, 1],  x en metres."""

    coefficients: np.ndarray
    length: float
    basis: PolynomialBasis = field(default_factory=MonomialBasis)

    def __post_init__(self) -> None:
        coef = np.asarray(self.coefficients, dtype=float).ravel()
        if coef.size == 0:
            raise ValueError("il faut au moins un coefficient")
        if self.length <= 0:
            raise ValueError("la longueur du troncon doit etre > 0")
        object.__setattr__(self, "coefficients", coef)

    @property
    def degree(self) -> int:
        return self.coefficients.size - 1

    def normalise(self, x_m: np.ndarray) -> np.ndarray:
        """Position [0, L] (m) -> [-1, 1]."""
        return 2.0 * np.asarray(x_m, dtype=float) / self.length - 1.0

    def __call__(self, x_m: np.ndarray) -> np.ndarray:
        A = self.basis.design_matrix(self.normalise(x_m), self.degree)
        return A @ self.coefficients

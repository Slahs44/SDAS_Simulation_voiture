"""Bases polynomiales : construisent la matrice de conception A (N x d+1)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.polynomial import legendre


class PolynomialBasis(ABC):
    """Base de polynomes evaluee sur des positions normalisees x dans [-1, 1]."""

    @abstractmethod
    def design_matrix(self, x: np.ndarray, degree: int) -> np.ndarray:
        """Matrice A telle que A[k, j] = phi_j(x_k), j = 0..degree."""

    @staticmethod
    def _check(x: np.ndarray, degree: int) -> np.ndarray:
        if degree < 0:
            raise ValueError("le degre doit etre >= 0")
        return np.asarray(x, dtype=float).ravel()


class MonomialBasis(PolynomialBasis):
    """Base canonique : 1, x, x^2, ..."""

    def design_matrix(self, x: np.ndarray, degree: int) -> np.ndarray:
        x = self._check(x, degree)
        return np.vander(x, degree + 1, increasing=True)


class LegendreBasis(PolynomialBasis):
    """Base de Legendre : mieux conditionnee que la base canonique sur [-1, 1]."""

    def design_matrix(self, x: np.ndarray, degree: int) -> np.ndarray:
        x = self._check(x, degree)
        return legendre.legvander(x, degree)

"""Evaluation : borne de Cramer-Rao et experiences Monte-Carlo."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vitesse.estimators import Estimator
from vitesse.simulator import MeasurementSimulator


def cramer_rao_covariance(A: np.ndarray, sigma: float) -> np.ndarray:
    """Borne de Cramer-Rao (bruit gaussien i.i.d., sigma connu) : sigma^2 (A^T A)^-1."""
    return sigma**2 * np.linalg.inv(A.T @ A)


@dataclass(frozen=True)
class MonteCarloResult:
    grid_m: np.ndarray          # (G,)   positions d'evaluation
    truth: np.ndarray           # (G,)   v(x) vraie
    estimates: np.ndarray       # (M, G) v(x) estimee a chaque tirage
    coefficients: np.ndarray    # (M, d_est+1)
    true_coefficients: np.ndarray

    @property
    def errors(self) -> np.ndarray:
        return self.estimates - self.truth

    @property
    def bias_profile(self) -> np.ndarray:
        return self.errors.mean(axis=0)

    @property
    def rmse_profile(self) -> np.ndarray:
        """RMSE de v_hat(x) en chaque position."""
        return np.sqrt((self.errors**2).mean(axis=0))

    @property
    def rmse_global(self) -> float:
        return float(np.sqrt((self.errors**2).mean()))

    @property
    def coefficient_bias(self) -> np.ndarray | None:
        """Biais sur les coefficients (None si degre/base differents du modele vrai)."""
        if self.coefficients.shape[1] != self.true_coefficients.size:
            return None
        return self.coefficients.mean(axis=0) - self.true_coefficients

    @property
    def coefficient_covariance(self) -> np.ndarray:
        return np.atleast_2d(np.cov(self.coefficients.T))


class MonteCarlo:
    """Repete M fois : tirage du bruit (et eventuellement des positions) puis estimation."""

    def __init__(self, simulator: MeasurementSimulator, estimator: Estimator) -> None:
        self.simulator = simulator
        self.estimator = estimator

    def run(self, n_runs: int, grid_m: np.ndarray, redraw_positions: bool = False) -> MonteCarloResult:
        sim = self.simulator
        x_m, ids = sim.draw_geometry()
        coefs, values = [], []
        for _ in range(n_runs):
            if redraw_positions:
                x_m, ids = sim.draw_geometry()
            fitted = self.estimator.fit(sim.measure(x_m, ids))
            coefs.append(fitted.coefficients)
            values.append(fitted(grid_m))
        return MonteCarloResult(
            grid_m=grid_m, truth=sim.profile(grid_m),
            estimates=np.array(values), coefficients=np.array(coefs),
            true_coefficients=sim.profile.coefficients,
        )

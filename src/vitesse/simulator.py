"""Simulation des donnees : chaque vehicule envoie des couples (position, vitesse)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from vitesse.noise import NoiseModel
from vitesse.profile import SpeedProfile


@dataclass(frozen=True)
class Measurements:
    """Points (x_k, v_k) recus, avec le vehicule emetteur de chacun."""

    x_m: np.ndarray
    v: np.ndarray
    vehicle_ids: np.ndarray

    def __post_init__(self) -> None:
        if not (self.x_m.shape == self.v.shape == self.vehicle_ids.shape):
            raise ValueError("x_m, v et vehicle_ids doivent avoir la meme taille")

    @property
    def n_points(self) -> int:
        return self.x_m.size

    @property
    def n_vehicles(self) -> int:
        return np.unique(self.vehicle_ids).size


class PositionSampler(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int, length: float) -> np.ndarray:
        """n positions dans [0, length]."""


class UniformPositionSampler(PositionSampler):
    def sample(self, rng, n, length):
        return rng.uniform(0.0, length, n)


class ClusteredPositionSampler(PositionSampler):
    """Cas difficile : tous les points dans [start, end] (fractions de L) -> trou de couverture."""

    def __init__(self, start: float, end: float) -> None:
        if not 0.0 <= start < end <= 1.0:
            raise ValueError("il faut 0 <= start < end <= 1")
        self.start, self.end = start, end

    def sample(self, rng, n, length):
        return rng.uniform(self.start * length, self.end * length, n)


class MeasurementSimulator:
    """Genere n_points mesures issues de n_vehicles vehicules (>= 1 envoi chacun)."""

    def __init__(self, profile: SpeedProfile, noise: NoiseModel,
                 n_vehicles: int = 70, n_points: int = 100,
                 position_sampler: PositionSampler | None = None,
                 rng: np.random.Generator | None = None) -> None:
        if n_vehicles < 1:
            raise ValueError("n_vehicles doit etre >= 1")
        if n_points < n_vehicles:
            raise ValueError("n_points doit etre >= n_vehicles")
        self.profile = profile
        self.noise = noise
        self.n_vehicles = n_vehicles
        self.n_points = n_points
        self.position_sampler = position_sampler or UniformPositionSampler()
        self.rng = rng if rng is not None else np.random.default_rng()

    def draw_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions et identifiants vehicules (sans bruit)."""
        extra = self.rng.integers(0, self.n_vehicles, self.n_points - self.n_vehicles)
        vehicle_ids = np.concatenate([np.arange(self.n_vehicles), extra])
        x_m = self.position_sampler.sample(self.rng, self.n_points, self.profile.length)
        return x_m, vehicle_ids

    def measure(self, x_m: np.ndarray, vehicle_ids: np.ndarray) -> Measurements:
        """v_k = v(x_k) + eps_k aux positions donnees (nouveau bruit a chaque appel)."""
        eps = self.noise.sample(self.rng, x_m.size, vehicle_ids)
        return Measurements(x_m=x_m, v=self.profile(x_m) + eps, vehicle_ids=vehicle_ids)

    def simulate(self) -> Measurements:
        x_m, vehicle_ids = self.draw_geometry()
        return self.measure(x_m, vehicle_ids)

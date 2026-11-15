"""Utilitaires partages par les scripts main_*.py (arguments, figures)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from vitesse import GaussianNoise, MeasurementSimulator, SpeedProfile

LENGTH = 2000.0                                    # longueur du troncon [m]
TRUE_COEFS = np.array([25.0, 3.0, -8.0])           # d = 2, x normalise dans [-1, 1]
SIGMA = 2.0                                        # bruit gaussien [m/s]
N_VEHICLES, N_POINTS = 70, 100
FIG_DIR = Path(__file__).resolve().parents[2] / "figures"


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true", help="afficher les figures (sinon : sauvegarde seule)")
    args = p.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    return args


def true_profile() -> SpeedProfile:
    return SpeedProfile(TRUE_COEFS, LENGTH)


def make_simulator(seed: int, noise=None, **kwargs) -> MeasurementSimulator:
    return MeasurementSimulator(true_profile(), noise or GaussianNoise(SIGMA),
                                n_vehicles=kwargs.pop("n_vehicles", N_VEHICLES),
                                n_points=kwargs.pop("n_points", N_POINTS),
                                rng=np.random.default_rng(seed), **kwargs)


def finish(fig, name: str, show: bool) -> None:
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"-> figure : {path}")
    if show:
        plt.show()
    plt.close(fig)

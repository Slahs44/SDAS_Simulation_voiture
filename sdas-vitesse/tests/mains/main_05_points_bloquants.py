"""Main 5 - Les cas qui posent question (a discuter avec le professeur) :
   (a) mauvais degre (sous/sur-ajustement), (b) valeurs aberrantes, (c) effet vehicule
   (bruit correle), (d) trou de couverture des positions."""
import matplotlib.pyplot as plt
import numpy as np
from _common import LENGTH, SIGMA, finish, make_simulator, parse_args, true_profile

from vitesse import (ClusteredPositionSampler, ContaminatedGaussianNoise, GaussianNoise,
                     MonteCarlo, OLSEstimator, VehicleEffectNoise)

GRID = np.linspace(0, LENGTH, 200)


def run(seed, noise, degree=2, n_runs=1000, **sim_kwargs):
    sim = make_simulator(seed, noise, **sim_kwargs)
    return MonteCarlo(sim, OLSEstimator(degree, LENGTH)).run(n_runs, GRID, redraw_positions=True)


def main() -> None:
    args = parse_args(__doc__)
    gauss = GaussianNoise(SIGMA)
    cases = {
        "reference (d=2, gaussien)": run(args.seed, gauss),
        "(a) sous-ajuste d=1": run(args.seed, gauss, degree=1),
        "(a) sur-ajuste d=5": run(args.seed, gauss, degree=5),
        "(b) 10 % aberrants x10": run(args.seed, ContaminatedGaussianNoise(SIGMA, 0.10, 10 * SIGMA)),
        "(c) effet vehicule sigma_v=3": run(args.seed, VehicleEffectNoise(SIGMA, 3.0)),
        "(d) points sur 0-50 % du troncon": run(
            args.seed, gauss, position_sampler=ClusteredPositionSampler(0.0, 0.5)),
    }
    print(f"{'cas':36s} RMSE globale   biais max")
    for name, r in cases.items():
        print(f"{name:36s} {r.rmse_global:8.3f}     {np.abs(r.bias_profile).max():8.3f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, r in cases.items():
        ax.plot(GRID, r.rmse_profile, label=name)
    ax.set(xlabel="position x [m]", ylabel="RMSE sur v(x) [m/s]", yscale="log",
           title="Precision de l'estimation de v(x) selon le scenario")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    finish(fig, "05_points_bloquants", args.show)


if __name__ == "__main__":
    main()

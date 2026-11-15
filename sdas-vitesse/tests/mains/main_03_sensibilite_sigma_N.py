"""Main 3 - Sensibilite : RMSE en fonction du bruit sigma et du nombre de points N."""
import matplotlib.pyplot as plt
import numpy as np
from _common import LENGTH, N_VEHICLES, finish, make_simulator, parse_args

from vitesse import GaussianNoise, MonteCarlo, OLSEstimator


def rmse(seed, sigma, n_points, n_vehicles=N_VEHICLES, n_runs=500):
    sim = make_simulator(seed, GaussianNoise(sigma), n_points=n_points,
                         n_vehicles=min(n_vehicles, n_points))
    return MonteCarlo(sim, OLSEstimator(2, LENGTH)).run(
        n_runs, np.linspace(0, LENGTH, 100), redraw_positions=True).rmse_global


def main() -> None:
    args = parse_args(__doc__)
    sigmas = np.array([0.5, 1, 2, 4, 8])
    ns = np.array([20, 50, 100, 200, 500, 1000])

    r_sigma = [rmse(args.seed, s, 100) for s in sigmas]
    r_n = [rmse(args.seed, 2.0, n) for n in ns]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    axs[0].loglog(sigmas, r_sigma, "o-")
    axs[0].set(xlabel="sigma [m/s]", ylabel="RMSE sur v(x) [m/s]", title="N = 100 : RMSE ~ sigma")
    axs[1].loglog(ns, r_n, "o-", label="empirique")
    axs[1].loglog(ns, r_n[2] * np.sqrt(ns[2] / ns), "k--", label="pente -1/2")
    axs[1].set(xlabel="N points", ylabel="RMSE sur v(x) [m/s]", title="sigma = 2 : RMSE ~ 1/sqrt(N)")
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
    axs[1].legend()
    finish(fig, "03_sensibilite", args.show)


if __name__ == "__main__":
    main()

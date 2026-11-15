"""Main 1 - Un tirage : donnees simulees (d=2, bruit gaussien) + estimation MCO + trace de v(x)."""
import matplotlib.pyplot as plt
import numpy as np
from _common import LENGTH, N_POINTS, N_VEHICLES, SIGMA, finish, make_simulator, parse_args, true_profile

from vitesse import OLSEstimator, cramer_rao_covariance


def main() -> None:
    args = parse_args(__doc__)
    sim = make_simulator(args.seed)
    data = sim.simulate()
    est = OLSEstimator(degree=2, length=LENGTH)
    fitted = est.fit(data)

    A = est.design_matrix(data.x_m)
    cov = cramer_rao_covariance(A, SIGMA)
    grid = np.linspace(0, LENGTH, 400)
    Ag = est.design_matrix(grid)
    std = np.sqrt(np.einsum("ij,jk,ik->i", Ag, cov, Ag))

    print("coefficients vrais   :", true_profile().coefficients)
    print("coefficients estimes :", np.round(fitted.coefficients, 3))
    print("ecart-type theorique :", np.round(np.sqrt(np.diag(cov)), 3))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(data.x_m, data.v, s=14, c="tab:gray", alpha=0.7,
               label=f"mesures ({N_POINTS} pts, {N_VEHICLES} vehicules)")
    ax.plot(grid, true_profile()(grid), "r-", lw=2, label="v(x) vraie")
    ax.plot(grid, fitted(grid), "b--", lw=2, label="v(x) estimee (MCO)")
    ax.fill_between(grid, fitted(grid) - 1.96 * std, fitted(grid) + 1.96 * std,
                    color="tab:blue", alpha=0.15, label="IC 95 % (theorique)")
    ax.set(xlabel="position x [m]", ylabel="vitesse [m/s]",
           title=f"d = 2, bruit gaussien sigma = {SIGMA} m/s")
    ax.grid(alpha=0.3)
    ax.legend()
    finish(fig, "01_un_tirage", args.show)


if __name__ == "__main__":
    main()

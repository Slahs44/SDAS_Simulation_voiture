"""Main 2 - Monte-Carlo : biais, covariance et comparaison a la borne de Cramer-Rao."""
import matplotlib.pyplot as plt
import numpy as np
from _common import LENGTH, SIGMA, finish, make_simulator, parse_args

from vitesse import MonteCarlo, OLSEstimator, cramer_rao_covariance


def main() -> None:
    args = parse_args(__doc__)
    sim = make_simulator(args.seed)
    est = OLSEstimator(2, LENGTH)
    grid = np.linspace(0, LENGTH, 200)

    # Geometrie (positions) fixee : on la tire une fois pour calculer la CRB sur la meme
    probe = make_simulator(args.seed)
    x_m, _ = probe.draw_geometry()
    crb = cramer_rao_covariance(est.design_matrix(x_m), SIGMA)

    res = MonteCarlo(sim, est).run(5000, grid)
    print("biais empirique sur les coefficients :", np.round(res.coefficient_bias, 4))
    print("ecart-type empirique :", np.round(np.sqrt(np.diag(res.coefficient_covariance)), 4))
    print("ecart-type CRB       :", np.round(np.sqrt(np.diag(crb)), 4))
    print(f"RMSE globale sur v(x) : {res.rmse_global:.3f} m/s")

    Ag = est.design_matrix(grid)
    crb_std = np.sqrt(np.einsum("ij,jk,ik->i", Ag, crb, Ag))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, res.rmse_profile, "b-", lw=2, label="RMSE empirique")
    ax.plot(grid, crb_std, "r--", lw=2, label="CRB (theorique)")
    ax.set(xlabel="position x [m]", ylabel="erreur sur v(x) [m/s]",
           title="Precision de v(x) selon la position")
    ax.grid(alpha=0.3)
    ax.legend()
    finish(fig, "02_monte_carlo", args.show)


if __name__ == "__main__":
    main()

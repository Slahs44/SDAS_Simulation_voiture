"""Main 4 - Conditionnement de A^T A : base canonique vs base de Legendre, selon le degre d."""
import matplotlib.pyplot as plt
import numpy as np
from _common import LENGTH, finish, make_simulator, parse_args

from vitesse import LegendreBasis, MonomialBasis, OLSEstimator


def main() -> None:
    args = parse_args(__doc__)
    x_m, _ = make_simulator(args.seed).draw_geometry()
    degrees = np.arange(1, 11)
    conds = {name: [] for name in ("canonique", "Legendre")}
    for d in degrees:
        for name, basis in (("canonique", MonomialBasis()), ("Legendre", LegendreBasis())):
            A = OLSEstimator(int(d), LENGTH, basis).design_matrix(x_m)
            conds[name].append(np.linalg.cond(A.T @ A))

    for name, c in conds.items():
        print(f"cond(A^T A) base {name:9s}:", [f"{v:.1e}" for v in c])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, c in conds.items():
        ax.semilogy(degrees, c, "o-", label=name)
    ax.set(xlabel="degre d", ylabel="cond(A^T A)", title="Conditionnement selon la base")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    finish(fig, "04_conditionnement", args.show)


if __name__ == "__main__":
    main()

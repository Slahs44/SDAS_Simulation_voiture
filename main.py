"""Simulation 1 : v(x) polynomial de degre d=2, bruit gaussien, estimation MCO."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# --- Parametres du modele ---------------------------------------------------
L = 1000.0                      # longueur du troncon (m)
d = 2                           # degre du polynome (connu)
coef_true = np.array([-7.2e-5, 7.2e-2, 72.0])   # a2, a1, a0 (np.polyval : degre decroissant), v en km/h, x en m
N = 100                         # nombre de points (x, v) recus
n_veh = 70                      # nombre de vehicules distincts (<= N)
tau = 5.0                       # ecart-type de l'effet vehicule b_i (km/h)
sigma = 5.0                     # ecart-type du bruit de mesure (km/h)

def v_true(x):
    return np.polyval(coef_true, x)

def simulate(N, sigma, rng, n_veh=n_veh, tau=tau):
    # chaque vehicule envoie >= 1 point ; les N - n_veh points restants vont a des vehicules tires au hasard
    veh = np.concatenate([np.arange(n_veh), rng.integers(0, n_veh, N - n_veh)])
    x = rng.uniform(0, L, N)                         # position de chaque point
    b = rng.normal(0, tau, n_veh)                    # effet vehicule (conducteur plus ou moins rapide)
    y = v_true(x) + b[veh] + rng.normal(0, sigma, N) # v_i(x) = v(x) + b_i + bruit gaussien
    o = np.argsort(x)
    return x[o], y[o], veh[o]

def fit_ols(x, y, d):
    # on normalise x dans [0,1] pour un meilleur conditionnement
    return np.polyfit(x / L, y, d)

def predict(c, x):
    return np.polyval(c, x / L)

# --- Une realisation --------------------------------------------------------
x, y, veh = simulate(N, sigma, rng)
c_hat = fit_ols(x, y, d)
xg = np.linspace(0, L, 400)
rmse = np.sqrt(np.mean((predict(c_hat, xg) - v_true(xg)) ** 2))
print(f"RMSE sur v(x) (1 realisation, N={N}, sigma={sigma}) : {rmse:.3f} km/h")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(x, y, s=12, alpha=.6, label="mesures bruitees")
ax[0].plot(xg, v_true(xg), "k-", lw=2, label="v(x) vraie")
ax[0].plot(xg, predict(c_hat, xg), "r--", lw=2, label="v(x) estimee (d=2)")
ax[0].set(xlabel="position x (m)", ylabel="vitesse (km/h)",
          title=f"Une realisation (N={N} points, {len(np.unique(veh))} vehicules)")
ax[0].legend()

# --- Monte Carlo : erreur vs N pour plusieurs sigma ---------------------------
Ns = [20, 50, 100, 200, 500, 1000]
for s in [1.5, 5.0, 12.0]:
    err = []
    for n in Ns:
        e = []
        for _ in range(300):
            xs, ys, _ = simulate(n, s, rng, n_veh=min(n_veh, n))
            e.append(np.sqrt(np.mean((predict(fit_ols(xs, ys, d), xg) - v_true(xg)) ** 2)))
        err.append(np.mean(e))
    ax[1].loglog(Ns, err, "o-", label=f"sigma={s} km/h")
ax[1].set(xlabel="N (nb de mesures)", ylabel="RMSE moyenne sur v(x) (km/h)",
          title="Monte Carlo (300 tirages) : decroissance en 1/sqrt(N)")
ax[1].legend(); ax[1].grid(True, which="both", alpha=.3)
plt.tight_layout()
plt.savefig("simu_gauss.png", dpi=150)
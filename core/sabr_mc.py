# =============================================================
# Module 5 - SABR Monte Carlo (professor's std_MC loop, vectorised over paths)
#   Notation mapping: professor's `initial_volatility` (sigma) = our alpha (vol level),
#                     professor's `vol_of_vol` (alpha)         = our nu.
#   Correlation convention (std_MC): Z1 drives the vol, the price shock is rho Z1 + sqrt(1-rho^2) Z2.
#   The professor's script omits r: here the log-price has drift (r - sigma^2/2) dt and the payoff
#   is discounted by e^{-rT}.
#
#   beta = 1 (lognormal, equities): X = ln(S/S0)
#       X_{t+dt} = X_t + (r - s_t^2/2) dt + s_t sqrt(dt) (rho Z1 + sqrt(1-rho^2) Z2)
#   beta < 1: simulated on the forward F_T (driftless under the T-forward measure), S_T = F_T,
#       F_{t+dt} = F_t + s_t (F_t^+)^beta sqrt(dt) (rho Z1 + sqrt(1-rho^2) Z2), absorbed at 0.
#
#   Schemes for the vol s_t:
#     euler : s_{t+dt} = s_t + nu s_t sqrt(dt) Z1                         (professor's version)
#     full  : s_t^+ = max(s_t, 0) used in drift and diffusion;
#             s_{t+dt} = s_t + nu s_t^+ sqrt(dt) Z1                       (full truncation)
#     log   : s_{t+dt} = s_t exp(nu sqrt(dt) Z1 - nu^2 dt / 2)            (exact, always > 0)
# =============================================================
import numpy as np
import pandas as pd

from core.black_scholes import bs_price, black76_price
from core.sabr import sabr_vol

SCHEMES = {"euler": "Euler-Maruyama", "full": "Full truncation", "log": "Log transform"}
MAX_WORK = 5e7          # cap on simulations x steps (Streamlit Cloud)


def simulate(S0, K, T, r, alpha, nu, rho, beta=1.0, n_paths=10000, n_steps=100,
             scheme="log", seed=42, phi=1, n_store=50, n_checkpoints=40):
    """Monte Carlo price of a European option under SABR.
    Returns a dict with price, se, ci95, mean_ST, forward, ST (array), payoffs, stored paths
    (S and vol, first n_store paths) and a convergence table."""
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme {scheme!r}")
    if n_paths < 2 or n_steps < 1:
        raise ValueError("Need at least 2 simulations and 1 time step.")
    if not -1 < rho < 1:
        raise ValueError("rho must be strictly between -1 and 1.")
    if min(S0, K, T, alpha) <= 0 or nu < 0:
        raise ValueError("S0, K, T, alpha must be > 0 and nu >= 0.")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c = np.sqrt(1 - rho**2)
    lognormal = beta >= 1.0
    X = np.zeros(n_paths)                              # log-price (beta = 1)
    F = np.full(n_paths, S0 * np.exp(r * T))           # forward (beta < 1)
    v = np.full(n_paths, float(alpha))
    m = min(n_store, n_paths)
    S_paths = np.zeros((m, n_steps + 1))
    v_paths = np.zeros((m, n_steps + 1))
    S_paths[:, 0], v_paths[:, 0] = S0, alpha
    for step in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        dW = rho * z1 + c * z2                         # correlated asset shock
        vp = v if scheme == "euler" else np.maximum(v, 0.0)
        if lognormal:
            X = X + (r - 0.5 * vp**2) * dt + vp * dW * sqrt_dt
        else:
            Fp = np.maximum(F, 0.0)
            F = np.maximum(F + vp * Fp**beta * dW * sqrt_dt, 0.0)
        if scheme == "log":
            v = v * np.exp(nu * z1 * sqrt_dt - 0.5 * nu**2 * dt)
        elif scheme == "full":
            v = v + nu * vp * z1 * sqrt_dt
        else:
            v = v + nu * v * z1 * sqrt_dt
        tau_left = T - (step + 1) * dt
        S_paths[:, step + 1] = (S0 * np.exp(X[:m]) if lognormal else F[:m] * np.exp(-r * tau_left))
        v_paths[:, step + 1] = v[:m]
    ST = S0 * np.exp(X) if lognormal else F
    disc = np.exp(-r * T)
    pay = disc * np.maximum(phi * (ST - K), 0.0)
    price = pay.mean()
    se = pay.std(ddof=1) / np.sqrt(n_paths)
    # convergence: running estimate at log-spaced checkpoints
    cps = np.unique(np.geomspace(min(100, n_paths), n_paths, n_checkpoints).astype(int))
    csum = np.cumsum(pay)
    conv = pd.DataFrame({"Simulations": cps, "Price": csum[cps - 1] / cps})
    return {"price": price, "se": se, "ci95": (price - 1.96 * se, price + 1.96 * se),
            "mean_ST": ST.mean(), "forward": S0 * np.exp(r * T), "ST": ST, "payoffs": pay,
            "S_paths": S_paths, "v_paths": v_paths, "convergence": conv,
            "neg_vol_share": float(np.mean(v < 0)), "scheme": scheme}


def benchmarks(S0, K, T, r, alpha, nu, rho, beta=1.0, phi=1):
    """Closed-form comparisons: Black price with Hagan vol, and BS with sigma = alpha (beta = 1)."""
    F = S0 * np.exp(r * T)
    hv = float(sabr_vol(K, F, T, alpha, beta, rho, nu))
    return {"hagan_vol": hv, "black_hagan": float(black76_price(F, K, hv, r, T, phi)),
            "bs_alpha": float(bs_price(S0, K, alpha * (F ** (beta - 1) if beta < 1 else 1.0),
                                       r, 0.0, T, phi))}


def compare_schemes(S0, K, T, r, alpha, nu, rho, beta=1.0, n_paths=10000, n_steps=100, seed=42, phi=1):
    """Run the 3 schemes with the same seed and return a comparison table."""
    rows = []
    for key, name in SCHEMES.items():
        res = simulate(S0, K, T, r, alpha, nu, rho, beta, n_paths, n_steps, key, seed, phi,
                       n_store=0, n_checkpoints=2)
        rows.append({"Scheme": name, "Price": res["price"], "Std. error": res["se"],
                     "95% CI low": res["ci95"][0], "95% CI high": res["ci95"][1],
                     "E[S_T]": res["mean_ST"], "S0 e^{rT}": res["forward"],
                     "Share of paths with vol < 0": res["neg_vol_share"]})
    return pd.DataFrame(rows)

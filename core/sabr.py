# =============================================================
# Module 4 - SABR (Hagan 2002, written as in Alos & Garcia 2021) - HW8 pipeline
#   dF = alpha F^beta dW ,  d alpha = nu alpha dZ ,  dW dZ = rho dt
#   Notation: alpha = vol level, nu = vol-of-vol.
#   (Professor's MC code / thesis: sigma0 = our alpha, their alpha = our nu.)
#   Forward of a non-dividend stock: F = S e^{rT}.
# =============================================================
import numpy as np
import pandas as pd

from core.black_scholes import black76_price, bs_delta, implied_vol, phi_of

# HW8 bounds: alpha, beta, rho, nu
BOUNDS = [(0.001, 5), (0, 1), (-0.999, 0.999), (0.001, 5)]


def forward(S, r, T):
    """F = S e^{rT} (non-dividend stock)."""
    return S * np.exp(r * np.asarray(T, dtype=float))


def sabr_vol(K, F, T, alpha, beta, rho, nu):
    # Hagan (2002) lognormal implied volatility (HW8, unchanged)
    K = np.asarray(K, dtype=float)
    FK = F * K
    logFK = np.log(F / K)
    z = (nu / alpha) * FK**((1 - beta) / 2) * logFK
    x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
    z_over_x = np.where(np.abs(z) < 1e-7, 1.0, z / np.where(np.abs(z) < 1e-7, 1.0, x))   # z/x -> 1 at the money
    denom = FK**((1 - beta) / 2) * (1 + (1 - beta)**2 / 24 * logFK**2 + (1 - beta)**4 / 1920 * logFK**4)
    correction = 1 + ((1 - beta)**2 / 24 * alpha**2 / FK**(1 - beta)
                      + rho * beta * nu * alpha / (4 * FK**((1 - beta) / 2))
                      + (2 - 3 * rho**2) / 24 * nu**2) * T
    return alpha / denom * z_over_x * correction


def SABR_calibration_thesis(K, F, tau, alpha, rho, sigma0):
    """Alos & Garcia formula (beta = 1). Their alpha = our nu, their sigma0 = our alpha."""
    z = (alpha * np.log(F / K)) / sigma0
    x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
    return sigma0 * (z / x) * (1 + 0.125 * tau * (2*alpha*rho*sigma0 + alpha**2 * ((2/3) - rho**2)))


def sabr_price(K, S, T, r, alpha, beta, rho, nu, phi=1):
    """Black-Scholes price with the Hagan vol (= Black-76 on F = S e^{rT}, discounted)."""
    F = forward(S, r, T)
    vol = sabr_vol(K, F, T, alpha, beta, rho, nu)
    return black76_price(F, K, vol, r, T, phi), vol


def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    # Differential Evolution - Pablo Rodriguez-Mier (pablormier.github.io, 2017), unchanged (HW8 section 6)
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def calibrate_slice(K_mkt, iv_mkt, F, T, beta=1.0, popsize=20, its=100, bounds=BOUNDS):
    """Calibrate one maturity with DE. beta=None -> beta is calibrated too (4 parameters);
    otherwise beta is fixed (3 parameters alpha, rho, nu).
    Generator: yields (iteration, params dict, sum of squared errors) after each DE iteration."""
    K_mkt, iv_mkt = np.asarray(K_mkt, float), np.asarray(iv_mkt, float)
    if beta is None:
        b = bounds

        def unpack(p):
            return p[0], p[1], p[2], p[3]
    else:
        b = [bounds[0], bounds[2], bounds[3]]

        def unpack(p):
            return p[0], beta, p[1], p[2]

    def objective(p):
        with np.errstate(all="ignore"):
            err = np.sum((sabr_vol(K_mkt, F, T, *unpack(p)) - iv_mkt)**2)
        return err if np.isfinite(err) else 1e10

    for i, (best, err) in enumerate(de(objective, b, popsize=popsize, its=its)):
        a, be, rh, nu = unpack(best)
        yield i, {"alpha": a, "beta": be, "rho": rh, "nu": nu}, err


def calibrate(df, beta=1.0, popsize=20, its=100, seed=42, progress=None):
    """Calibrate every maturity of a clean DataFrame (columns T, F, strike, iv[, expiry]).
    np.random.seed(seed) once, as in HW8. progress(fraction, text) is called after each iteration.
    Returns a DataFrame: Expiry, T, F, n, alpha, beta, rho, nu, RMSE."""
    np.random.seed(seed)
    groups = list(df.groupby("T"))
    total = max(len(groups) * its, 1)
    done = 0
    results = []
    for T, g in groups:
        K_mkt, iv_mkt, F = g["strike"].values, g["iv"].values, g["F"].iloc[0]
        best, err = None, None
        for i, best, err in calibrate_slice(K_mkt, iv_mkt, F, T, beta, popsize, its):
            done += 1
            if progress is not None:
                progress(done / total, f"Maturity T={T:.4f}: iteration {i + 1}/{its}")
        results.append({"Expiry": g["expiry"].iloc[0] if "expiry" in g else "", "T": T, "F": F,
                        "n": len(K_mkt), **best, "RMSE": np.sqrt(err / len(K_mkt))})
    return pd.DataFrame(results)


# -------------------------------------------------------------
# Data loading / cleaning
# -------------------------------------------------------------
COURSE_COLUMNS = {"contractSymbol", "expirationDate", "Call_Put", "strike", "bid", "ask",
                  "volume", "impliedVolatility"}


def is_course_format(raw):
    return COURSE_COLUMNS.issubset(set(raw.columns))


def course_tickers(raw):
    return sorted(raw["contractSymbol"].astype(str).str.extract(r"^([A-Z]+)")[0].dropna().unique())


def clean_course_file(raw, ticker, S0, r, val_date):
    """HW8 cleaning of Final_Database_Options_.xlsx: last snapshot block of the ticker, calls,
    liquid quotes, 0.8 <= K/F <= 1.2, 5% < IV < 200%, T > 7 days."""
    raw = raw.copy()
    raw["ticker"] = raw["contractSymbol"].astype(str).str.extract(r"^([A-Z]+)")[0]
    raw["expirationDate"] = pd.to_datetime(raw["expirationDate"])
    new_block = (raw["ticker"] != raw["ticker"].shift()) | (raw["expirationDate"] < raw["expirationDate"].shift())
    raw["snapshot"] = new_block.cumsum()
    df = raw[raw["ticker"] == ticker]
    df = df[df["snapshot"] == df["snapshot"].max()].copy()
    df["T"] = (df["expirationDate"] - pd.Timestamp(val_date)).dt.days / 365
    df["F"] = S0 * np.exp(r * df["T"])
    df = df[df["T"] > 7 / 365]
    df = df[df["Call_Put"] == "C"]
    df = df[(df["bid"] > 0) & (df["ask"] > df["bid"]) & (df["volume"] > 0)]
    df = df[(df["strike"] / df["F"] >= 0.8) & (df["strike"] / df["F"] <= 1.2)]
    df = df[(df["impliedVolatility"] > 0.05) & (df["impliedVolatility"] < 2)]
    df = df.sort_values(["T", "strike"])
    out = pd.DataFrame({"expiry": df["expirationDate"].dt.date.astype(str), "T": df["T"],
                        "F": df["F"], "strike": df["strike"], "iv": df["impliedVolatility"]})
    return out.reset_index(drop=True)


def _find(cols, *names):
    low = {c.lower().strip(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def clean_simple_file(raw, S0, r, val_date=None, apply_filters=True):
    """Simple format: strike, T (years) or expiry date, impliedVolatility (or price -> invert BS).
    Optional column type/Call_Put (call by default for prices). Same filters as HW8 (no liquidity filter)."""
    cols = list(raw.columns)
    c_k = _find(cols, "strike", "k")
    c_t = _find(cols, "t", "maturity", "t (years)", "tau")
    c_e = _find(cols, "expiry", "expirationdate", "expiration", "expiry date")
    c_iv = _find(cols, "impliedvolatility", "iv", "implied_vol", "implied volatility", "vol")
    c_p = _find(cols, "price", "market price", "mid", "lastprice")
    c_type = _find(cols, "type", "call_put", "cp")
    if c_k is None or (c_t is None and c_e is None) or (c_iv is None and c_p is None):
        raise ValueError("Simple format needs columns: strike, T (or expiry), impliedVolatility (or price).")
    df = pd.DataFrame({"strike": pd.to_numeric(raw[c_k], errors="coerce")})
    if c_t is not None:
        df["T"] = pd.to_numeric(raw[c_t], errors="coerce")
        df["expiry"] = df["T"].round(4).astype(str)
    else:
        if val_date is None:
            raise ValueError("A valuation date is needed to convert expiry dates into T.")
        exp = pd.to_datetime(raw[c_e], errors="coerce")
        df["T"] = (exp - pd.Timestamp(val_date)).dt.days / 365
        df["expiry"] = exp.dt.date.astype(str)
    df["F"] = S0 * np.exp(r * df["T"])
    if c_iv is not None:
        df["iv"] = pd.to_numeric(raw[c_iv], errors="coerce")
    else:
        prices = pd.to_numeric(raw[c_p], errors="coerce")
        kinds = raw[c_type].astype(str) if c_type is not None else pd.Series(["call"] * len(raw))
        df["iv"] = [implied_vol(p, S0, k, r, 0.0, t, phi_of(ty))
                    if np.isfinite(p) and t > 0 else np.nan
                    for p, k, t, ty in zip(prices, df["strike"], df["T"], kinds)]
    df = df.dropna(subset=["strike", "T", "iv"])
    df = df[df["T"] > 0]
    if apply_filters:
        df = df[df["T"] > 7 / 365]
        df = df[(df["strike"] / df["F"] >= 0.8) & (df["strike"] / df["F"] <= 1.2)]
        df = df[(df["iv"] > 0.05) & (df["iv"] < 2)]
    return df.sort_values(["T", "strike"]).reset_index(drop=True)[["expiry", "T", "F", "strike", "iv"]]


# -------------------------------------------------------------
# Smile grid, pricing table, mispriced options
# -------------------------------------------------------------
def smile_grid(S_values, K_grid, T, r, p):
    """sigma_SABR(K) for each spot S (F = S e^{rT}). Returns a long DataFrame."""
    rows = []
    for S in S_values:
        F = float(forward(S, r, T))
        vols = sabr_vol(np.asarray(K_grid, float), F, T, p["alpha"], p["beta"], p["rho"], p["nu"])
        for K, v in zip(K_grid, np.atleast_1d(vols)):
            rows.append({"S": S, "F": F, "K": K, "K/F": K / F, "sigma_SABR": float(v)})
    return pd.DataFrame(rows)


def pricing_table(K_list, S, T, r, p, phi=1):
    """K, F, T, sigma_Hagan, SABR price (BS with Hagan vol), BS price with flat ATM vol."""
    K = np.asarray(K_list, float)
    F = float(forward(S, r, T))
    sig_h = sabr_vol(K, F, T, p["alpha"], p["beta"], p["rho"], p["nu"])
    sig_atm = float(sabr_vol(F, F, T, p["alpha"], p["beta"], p["rho"], p["nu"]))
    return pd.DataFrame({"K": K, "F": F, "T": T, "sigma_Hagan": sig_h,
                         "SABR price": black76_price(F, K, sig_h, r, T, phi),
                         "sigma_ATM (flat)": sig_atm,
                         "BS price (flat ATM vol)": black76_price(F, K, sig_atm, r, T, phi)})


def mispricing_strategy(market_price, K, T, S, r, p, phi=1, tol=0.0, contracts=1, multiplier=100):
    """Model-based (relative value) arbitrage vs the SABR price: buy the cheap option / sell the
    rich one, delta-hedge with the stock, finance the rest with bonds. NOT riskless."""
    model, vol = sabr_price(K, S, T, r, p["alpha"], p["beta"], p["rho"], p["nu"], phi)
    model, vol = float(model), float(vol)
    gap = market_price - model
    units = contracts * multiplier
    res = {"model_price": model, "sigma_hagan": vol, "gap": gap, "units": units,
           "market_iv": implied_vol(market_price, S, K, r, 0.0, T, phi)}
    if abs(gap) <= tol:
        res.update(decision="fair", steps=pd.DataFrame())
        return res
    s_opt = -1 if gap > 0 else 1                       # rich -> SELL (-1), cheap -> BUY (+1)
    delta = float(bs_delta(S, K, vol, r, 0.0, T, phi))
    n_opt = s_opt * units
    n_sh = -n_opt * delta                              # delta-neutral stock position
    cash = -n_opt * market_price - n_sh * S            # cash left after option + stock trades
    name = "call" if phi == 1 else "put"
    steps = pd.DataFrame([
        {"Step": 1, "Instrument": f"{name.title()} K={K:g} T={T:g}",
         "Action": "BUY" if n_opt > 0 else "SELL", "Quantity": abs(n_opt), "Cash flow at t=0": -n_opt * market_price},
        {"Step": 2, "Instrument": "Stock (delta hedge)", "Action": "BUY" if n_sh > 0 else "SELL",
         "Quantity": abs(n_sh), "Cash flow at t=0": -n_sh * S},
        {"Step": 3, "Instrument": "Risk-free bond (financing)",
         "Action": "LEND (buy bond)" if cash > 0 else "BORROW (sell bond)", "Quantity": abs(cash),
         "Cash flow at t=0": -cash},
    ])
    res.update(decision="overpriced" if gap > 0 else "underpriced", steps=steps,
               delta=delta, shares=n_sh, bond=cash, expected_profit=abs(gap) * units - tol * units)
    return res

# =============================================================
# Module 1 - Portfolio of European options (valuation + payoff scenarios)
# Convention: Long = +1, Short = -1 (HW "Module B").
# Position value = sign x contracts x multiplier x price per option.
# All positions are on ONE underlying with spot S (shared market input).
# =============================================================
import numpy as np
import pandas as pd

from core.black_scholes import bs_price, bs_delta, bs_gamma, payoff, phi_of

COLUMNS = ["Ticker", "Type", "Position", "K", "T (years)", "Contracts",
           "Multiplier", "Sigma", "Market price"]

# Course example (HW "Module B"): long 1 call 105, short 2 puts 95, short 1 call 110
DEFAULT_POSITIONS = pd.DataFrame([
    ["AAPL", "call", "long", 105.0, 0.5, 1, 100, 0.25, np.nan],
    ["AAPL", "put", "short", 95.0, 0.5, 2, 100, 0.25, np.nan],
    ["AAPL", "call", "short", 110.0, 0.5, 1, 100, 0.25, np.nan],
], columns=COLUMNS).astype({"K": float, "T (years)": float, "Contracts": float, "Multiplier": float,
                            "Sigma": float, "Market price": float})


def sign_of(position):
    """'long' -> +1, 'short' -> -1."""
    p = str(position).strip().lower()
    if p in ("long", "buy", "+1", "1"):
        return 1
    if p in ("short", "sell", "-1"):
        return -1
    raise ValueError(f"Unknown position {position!r} (use long or short)")


def validate_positions(df):
    """Drop empty rows and check every field. Returns a clean copy or raises ValueError."""
    df = df.copy()
    df = df.dropna(how="all")
    df = df[~df[["Type", "K"]].isna().all(axis=1)]
    if df.empty:
        raise ValueError("The portfolio is empty: add at least one position.")
    for i, row in df.iterrows():
        n = f"Row {i + 1}"
        phi_of(row["Type"])
        sign_of(row["Position"])
        for col, strict in (("K", True), ("T (years)", True), ("Sigma", True),
                            ("Contracts", False), ("Multiplier", True)):
            v = row[col]
            if v is None or pd.isna(v):
                raise ValueError(f"{n}: '{col}' is missing.")
            if (v <= 0) if strict else (v < 0):
                raise ValueError(f"{n}: '{col}' must be {'> 0' if strict else '>= 0'}.")
        mp = row["Market price"]
        if mp is not None and not pd.isna(mp) and mp < 0:
            raise ValueError(f"{n}: market price must be >= 0.")
    return df.reset_index(drop=True)


def value_portfolio(df, S, r, q=0.0):
    """Value each position and the whole portfolio.
    Returns (per-position DataFrame, totals dict)."""
    df = validate_positions(df)
    rows = []
    for _, p in df.iterrows():
        phi, sgn = phi_of(p["Type"]), sign_of(p["Position"])
        K, T, sig = float(p["K"]), float(p["T (years)"]), float(p["Sigma"])
        units = sgn * float(p["Contracts"]) * float(p["Multiplier"])     # signed number of options x multiplier
        bs = bs_price(S, K, sig, r, q, T, phi)
        mp = p["Market price"]
        used = bs if (mp is None or pd.isna(mp)) else float(mp)
        intrinsic = float(payoff(S, K, phi))
        delta = bs_delta(S, K, sig, r, q, T, phi)
        gamma = bs_gamma(S, K, sig, r, q, T, phi)
        rows.append({
            "Ticker": p["Ticker"], "Type": str(p["Type"]).lower(), "Position": str(p["Position"]).lower(),
            "K": K, "T": T, "Contracts": float(p["Contracts"]), "Multiplier": float(p["Multiplier"]),
            "Sigma": sig, "BS price": bs, "Price used": used,
            "Price source": "BS" if (mp is None or pd.isna(mp)) else "Market",
            "Intrinsic / option": intrinsic, "Time value / option": used - intrinsic,
            "Delta / option": delta, "Gamma / option": gamma,
            "Market value": units * used, "Intrinsic value": units * intrinsic,
            "Time value": units * (used - intrinsic),
            "Delta (shares)": units * delta, "Gamma (shares)": units * gamma,
        })
    res = pd.DataFrame(rows)
    totals = {
        "Market value": res["Market value"].sum(),
        "Intrinsic value": res["Intrinsic value"].sum(),
        "Time value": res["Time value"].sum(),
        "Net delta (shares)": res["Delta (shares)"].sum(),
        "Net gamma (shares)": res["Gamma (shares)"].sum(),
    }
    return res, totals


def payoff_at_maturity(df, ST):
    """Portfolio payoff at maturity for each S_T (HW Module B), in money (x contracts x multiplier)."""
    df = validate_positions(df)
    ST = np.asarray(ST, dtype=float)
    total = np.zeros_like(ST)
    for _, p in df.iterrows():
        units = sign_of(p["Position"]) * float(p["Contracts"]) * float(p["Multiplier"])
        total = total + units * payoff(ST, float(p["K"]), phi_of(p["Type"]))
    return total


def value_curve(df, S_grid, r, q=0.0):
    """Current Black-Scholes value of the portfolio for each spot in S_grid."""
    df = validate_positions(df)
    S_grid = np.asarray(S_grid, dtype=float)
    total = np.zeros_like(S_grid)
    for _, p in df.iterrows():
        units = sign_of(p["Position"]) * float(p["Contracts"]) * float(p["Multiplier"])
        total = total + units * bs_price(S_grid, float(p["K"]), float(p["Sigma"]), r, q,
                                         float(p["T (years)"]), phi_of(p["Type"]))
    return total


def scenario_table(df, scenarios, premium_paid=None):
    """Payoff at maturity for user-chosen S_T values. If premium_paid (net cost of the
    portfolio today) is given, a P&L column = payoff - premium is added (no discounting)."""
    sc = np.asarray(scenarios, dtype=float)
    out = pd.DataFrame({"S_T": sc, "Portfolio payoff": payoff_at_maturity(df, sc)})
    if premium_paid is not None:
        out["P&L vs value today"] = out["Portfolio payoff"] - premium_paid
    return out

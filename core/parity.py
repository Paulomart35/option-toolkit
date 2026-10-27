# =============================================================
# Module 2 - Put-call parity arbitrage with synthetic options (stock + bonds)
#   Parity:          C - P = S e^{-qT} - K e^{-rT}
#   Synthetic call = long put + long e^{-qT} shares + borrow PV(K)  (short zero-coupon bond paying K)
#   Synthetic put  = long call + short e^{-qT} shares + lend PV(K)  (long zero-coupon bond paying K)
# The call view and the put view are the SAME arbitrage (same four trades).
# Positive quantity = BUY, negative = SELL (HW7 action()).
# =============================================================
import numpy as np
import pandas as pd

from core.black_scholes import payoff


def action(qty):
    """Translate a signed quantity into BUY / SELL / NOTHING (HW7)."""
    return "BUY" if qty > 1e-9 else ("SELL" if qty < -1e-9 else "NOTHING")


def parity_analysis(S, K, T, r, C, P, q=0.0, tol=0.0):
    """Compare market call/put prices with put-call parity.
    tol = transaction cost / tolerance per share: an arbitrage is reported only if |gap| > tol.
    Returns a dict with both sides of parity, synthetics, the gap and the decision."""
    if min(S, K, T) <= 0:
        raise ValueError("S, K and T must be > 0.")
    if C < 0 or P < 0:
        raise ValueError("Option prices must be >= 0.")
    pv_K = K * np.exp(-r * T)
    stock_leg = np.exp(-q * T)                  # shares held today so that 1 share is owned at T (dividends reinvested)
    S_adj = S * stock_leg
    synthetic_call = P + S_adj - pv_K
    synthetic_put = C - S_adj + pv_K
    gap = C - synthetic_call                    # = (C - P) - (S e^{-qT} - K e^{-rT}) = synthetic_put - P
    if gap > tol:
        decision = "call_overpriced"            # call rich / put cheap
    elif gap < -tol:
        decision = "call_underpriced"           # call cheap / put rich
    else:
        decision = "none"
    return {
        "S": S, "K": K, "T": T, "r": r, "q": q, "C": C, "P": P, "tol": tol,
        "pv_K": pv_K, "stock_leg": stock_leg, "S_adj": S_adj,
        "lhs": C - P, "rhs": S_adj - pv_K, "gap": gap,
        "synthetic_call": synthetic_call, "synthetic_put": synthetic_put,
        "decision": decision,
        "profit_per_share": max(abs(gap) - tol, 0.0) if decision != "none" else 0.0,
        "bounds": check_bounds(S, K, T, r, C, P, q),
    }


def check_bounds(S, K, T, r, C, P, q=0.0):
    """No-arbitrage bounds for European options. Returns a DataFrame (one row per bound)."""
    S_adj, pv_K = S * np.exp(-q * T), K * np.exp(-r * T)
    rows = [
        ("Call lower bound", "C ≥ max(S e^{-qT} − K e^{-rT}, 0)", C, max(S_adj - pv_K, 0.0), C >= max(S_adj - pv_K, 0.0) - 1e-12,
         "Call too cheap: BUY the call, SELL e^{-qT} shares, LEND PV(K)."),
        ("Call upper bound", "C ≤ S e^{-qT}", C, S_adj, C <= S_adj + 1e-12,
         "Call above the stock: SELL the call, BUY e^{-qT} shares."),
        ("Put lower bound", "P ≥ max(K e^{-rT} − S e^{-qT}, 0)", P, max(pv_K - S_adj, 0.0), P >= max(pv_K - S_adj, 0.0) - 1e-12,
         "Put too cheap: BUY the put, BUY e^{-qT} shares, BORROW PV(K)."),
        ("Put upper bound", "P ≤ K e^{-rT}", P, pv_K, P <= pv_K + 1e-12,
         "Put above PV(K): SELL the put, LEND PV(K)."),
    ]
    return pd.DataFrame(rows, columns=["Bound", "Condition", "Price", "Limit", "OK", "If violated"])


def build_strategy(res, contracts=1, multiplier=100):
    """Step-by-step arbitrage strategy (scaled by contracts x multiplier).
    Returns (steps DataFrame, payoff-at-T DataFrame, summary dict). Cash flows: + = received, - = paid."""
    units = contracts * multiplier
    if res["decision"] == "none":
        return pd.DataFrame(), pd.DataFrame(), {"units": units, "profit_today": 0.0, "profit_fv": 0.0}
    s = 1 if res["decision"] == "call_overpriced" else -1     # +1: sell call / buy synthetic call
    n_sh = res["stock_leg"] * units
    legs = [
        ("Call option (K={:g})".format(res["K"]), -s * units, s * res["C"] * units),
        ("Put option (K={:g})".format(res["K"]), s * units, -s * res["P"] * units),
        ("Stock (e^{-qT} share per option)", s * n_sh, -s * res["S_adj"] * units),
        ("Zero-coupon bond paying K at T", -s * units, s * res["pv_K"] * units),
    ]
    steps = []
    for i, (inst, qty, cf) in enumerate(legs, 1):
        if inst.startswith("Zero"):
            verb = "BORROW PV(K) (sell the bond)" if qty < 0 else "LEND PV(K) (buy the bond)"
        else:
            verb = action(qty)
        steps.append({"Step": i, "Instrument": inst, "Action": verb, "Quantity": abs(qty),
                      "Cash flow at t=0": cf})
    steps = pd.DataFrame(steps)
    gross = steps["Cash flow at t=0"].sum()                    # = |gap| x units
    cost = res["tol"] * units
    profit_today = gross - cost
    K = res["K"]
    payoff_rows = []
    for case, ST in (("S_T > K", K * 1.2), ("S_T ≤ K", K * 0.8)):
        call = -s * units * payoff(ST, K, 1)
        put = s * units * payoff(ST, K, -1)
        stock = s * units * ST                                  # e^{-qT} shares grow to 1 share with reinvested dividends
        bond = -s * units * K
        payoff_rows.append({"Case": case, "Example S_T": ST, "Call leg": float(call),
                            "Put leg": float(put), "Stock leg": stock, "Bond leg": bond,
                            "Total": float(call + put + stock + bond)})
    payoff_tab = pd.DataFrame(payoff_rows)
    summary = {"units": units, "gross_today": gross, "cost": cost, "profit_today": profit_today,
               "profit_fv": profit_today * np.exp(res["r"] * res["T"]),
               "view_call": "SELL the call, BUY the synthetic call (put + stock + borrow PV(K))" if s == 1
               else "BUY the call, SELL the synthetic call (sell put, short stock, lend PV(K))",
               "view_put": "BUY the put, SELL the synthetic put (sell call, buy stock, borrow PV(K))" if s == 1
               else "SELL the put, BUY the synthetic put (buy call, short stock, lend PV(K))"}
    return steps, payoff_tab, summary


def leg_payoffs(res, ST, contracts=1, multiplier=100):
    """Payoff at T of each leg and of the total, for a grid of S_T (for the chart)."""
    ST = np.asarray(ST, dtype=float)
    units = contracts * multiplier
    s = 1 if res["decision"] == "call_overpriced" else -1
    K = res["K"]
    df = pd.DataFrame({"S_T": ST,
                       "Call leg": -s * units * payoff(ST, K, 1),
                       "Put leg": s * units * payoff(ST, K, -1),
                       "Stock leg": s * units * ST,
                       "Bond leg": -s * units * K * np.ones_like(ST)})
    df["Total (at T)"] = df[["Call leg", "Put leg", "Stock leg", "Bond leg"]].sum(axis=1)
    return df

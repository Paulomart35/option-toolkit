# =============================================================
# Module 3 - Delta and Delta-Gamma hedging (HW7 conventions)
#  - We are SHORT N written options.
#  - Delta hedge uses the underlying only.
#  - Delta-Gamma hedge uses one hedging option (for Gamma) + the underlying (for Delta):
#        n_h = N * Gamma_w / Gamma_h ,   n_s = N * Delta_w - n_h * Delta_h
#  - Portfolio delta = -N*Delta_w + n_h*Delta_h + n_s ; gamma = -N*Gamma_w + n_h*Gamma_h
#  - Positive quantity = BUY, negative quantity = SELL.
# =============================================================
import numpy as np
import pandas as pd

from core.black_scholes import bs_price, bs_delta, bs_gamma, phi_of

DT_DAY = 1.0 / 252.0          # one trading day
GAMMA_MIN = 1e-10


def action(q):
    """Translate a signed quantity into BUY / SELL / NOTHING (HW7)."""
    return "BUY" if q > 1e-9 else ("SELL" if q < -1e-9 else "NOTHING")


def _greeks(opt, S, tau, r, sigma, q=0.0):
    phi = phi_of(opt["kind"])
    return (bs_price(S, opt["K"], sigma, r, q, tau, phi),
            bs_delta(S, opt["K"], sigma, r, q, tau, phi),
            bs_gamma(S, opt["K"], sigma, r, q, tau, phi))


def check_hedge_option(S, r, sigma, hedge, q=0.0):
    """HW7 validation: the hedging option must have a non-zero Gamma."""
    _, _, g = _greeks(hedge, S, hedge["T"], r, sigma, q)
    if g < GAMMA_MIN:
        raise ValueError("Gamma of the hedging option is ~0 (too deep ITM/OTM): choose another K/T.")


def compute_hedge(p):
    """HW7 compute_hedge. p = dict(S, r, sigma, N, written={kind,K,T}, strategy, hedge, n_override, q)."""
    S, r, sig, N, q = p["S"], p["r"], p["sigma"], p["N"], p.get("q", 0.0)
    w = p["written"]
    _, d_w, g_w = _greeks(w, S, w["T"], r, sig, q)
    n_h, d_h, g_h = 0.0, 0.0, 0.0
    if p["strategy"] == "delta-gamma":
        h = p["hedge"]
        check_hedge_option(S, r, sig, h, q)
        _, d_h, g_h = _greeks(h, S, h["T"], r, sig, q)
        n_h = N * g_w / g_h if p.get("n_override") is None else p["n_override"]
    n_s = N * d_w - n_h * d_h
    return dict(d_w=d_w, g_w=g_w, d_h=d_h, g_h=g_h, n_h=n_h, n_s=n_s,
                port_delta=-N * d_w + n_h * d_h + n_s,
                port_gamma=-N * g_w + n_h * g_h)


def report_lines(p, h):
    """HW7 print_report as a list of lines."""
    w, hg = p["written"], p.get("hedge") if p["strategy"] == "delta-gamma" else None
    lines = [f"Written option: {p['N']:g} x {w['kind'].upper()} (K={w['K']:g}, T={w['T']:g}) -> SOLD (already written)",
             f"  Delta = {h['d_w']:.4f} | Gamma = {h['g_w']:.6f}"]
    if hg:
        lines += [f"Hedging option: {hg['kind'].upper()} (K={hg['K']:g}, T={hg['T']:g})",
                  f"  Delta = {h['d_h']:.4f} | Gamma = {h['g_h']:.6f}",
                  f"  -> {action(h['n_h'])} {abs(h['n_h']):.4f} hedging options"]
    lines += [f"Underlying: -> {action(h['n_s'])} {abs(h['n_s']):.4f} shares",
              f"Residual portfolio Delta = {h['port_delta']:.6f} | Gamma = {h['port_gamma']:.6f}"]
    return lines


def run_scenarios(p, h, n_points=61, move=0.30):
    """HW7 run_scenarios: revalue the STATIC hedged portfolio after an instantaneous spot move."""
    S0, r, sig, N, q = p["S"], p["r"], p["sigma"], p["N"], p.get("q", 0.0)
    w = p["written"]
    hg = p.get("hedge") if p["strategy"] == "delta-gamma" else None
    S_grid = np.linspace(S0 * (1 - move), S0 * (1 + move), n_points)
    pw = phi_of(w["kind"])

    def value(S):
        v = -N * bs_price(S, w["K"], sig, r, q, w["T"], pw) + h["n_s"] * S
        if hg:
            v = v + h["n_h"] * bs_price(S, hg["K"], sig, r, q, hg["T"], phi_of(hg["kind"]))
        return v

    def unhedged(S):
        return -N * bs_price(S, w["K"], sig, r, q, w["T"], pw)

    V0, U0 = value(S0), unhedged(S0)
    res = pd.DataFrame({"S": S_grid})
    res["Written price"] = bs_price(S_grid, w["K"], sig, r, q, w["T"], pw)
    res["Delta (written)"] = bs_delta(S_grid, w["K"], sig, r, q, w["T"], pw)
    res["Gamma (written)"] = bs_gamma(S_grid, w["K"], sig, r, q, w["T"], pw)
    res["Hedge options"] = h["n_h"]
    res["Hedge shares"] = h["n_s"]
    res["PF value"] = value(S_grid)
    ph = phi_of(hg["kind"]) if hg else 1
    res["PF delta"] = (-N * res["Delta (written)"] + h["n_s"]
                       + (h["n_h"] * bs_delta(S_grid, hg["K"], sig, r, q, hg["T"], ph) if hg else 0))
    res["PF gamma"] = (-N * res["Gamma (written)"]
                       + (h["n_h"] * bs_gamma(S_grid, hg["K"], sig, r, q, hg["T"], ph) if hg else 0))
    res["P&L hedged"] = res["PF value"] - V0
    res["P&L unhedged"] = unhedged(S_grid) - U0
    return res


# -------------------------------------------------------------
# Dynamic rebalancing engine (daily)
# -------------------------------------------------------------
def rebalance_engine(spots, r, sigma, N, written, strategy, hedge=None, dt=DT_DAY, q=0.0):
    """Run a daily-rebalanced hedge along one or many spot paths.

    spots    : array (n_days+1,) or (n_paths, n_days+1); spots[..., 0] = today.
    strategy : 'unhedged' | 'delta' | 'delta-gamma'
    At day t: tau_t = T - t*dt for each option. Holdings are rebalanced at the close of each day.
    Cash (bank account): premium received at t=0 minus hedge purchases; grows by e^{r dt} each day.
    Portfolio value V_t = -N*C_w + n_h*H + n_s*S + cash (V_0 = 0). Daily P&L = V_t - V_{t-1}
    computed with the holdings of the previous day, BEFORE rebalancing.

    Returns dict of arrays with shape (n_paths, n_days+1): S, n_s, n_h, trade_s, trade_h, cash,
    value, pnl, cum_pnl, written_price, hedge_price, delta_w, gamma_w, delta_h, gamma_h, pf_delta, pf_gamma.
    """
    S = np.atleast_2d(np.asarray(spots, dtype=float))
    n_paths, n_t = S.shape
    if np.any(S <= 0):
        raise ValueError("Spot prices must be > 0.")
    use_h = strategy == "delta-gamma"
    if use_h and hedge is None:
        raise ValueError("Delta-gamma hedging needs a hedging option.")
    out = {k: np.zeros((n_paths, n_t)) for k in
           ("n_s", "n_h", "trade_s", "trade_h", "cash", "value", "pnl", "cum_pnl", "written_price",
            "hedge_price", "delta_w", "gamma_w", "delta_h", "gamma_h", "pf_delta", "pf_gamma", "tau_w")}
    out["S"] = S
    n_s_prev = np.zeros(n_paths)
    n_h_prev = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    v_prev = np.zeros(n_paths)
    for t in range(n_t):
        St = S[:, t]
        tau_w = max(written["T"] - t * dt, 0.0)
        c_w, d_w, g_w = _greeks(written, St, tau_w, r, sigma, q)
        if use_h:
            tau_h = max(hedge["T"] - t * dt, 0.0)
            c_h, d_h, g_h = _greeks(hedge, St, tau_h, r, sigma, q)
        else:
            c_h = d_h = g_h = np.zeros(n_paths)
        c_w, d_w, g_w, c_h, d_h, g_h = (np.broadcast_to(np.asarray(x, dtype=float), (n_paths,))
                                        for x in (c_w, d_w, g_w, c_h, d_h, g_h))
        if t == 0:
            cash = N * c_w                                   # premium received for the written options
        else:
            cash = cash * np.exp(r * dt)                     # bank account accrues interest
        # value with yesterday's holdings (before today's trades)
        value = -N * c_w + n_h_prev * c_h + n_s_prev * St + cash
        pnl = value - v_prev if t > 0 else np.zeros(n_paths)
        # new hedge quantities
        if strategy == "unhedged":
            n_h_new, n_s_new = np.zeros(n_paths), np.zeros(n_paths)
        else:
            if use_h:
                n_h_new = np.where(g_h > GAMMA_MIN, N * g_w / np.where(g_h > GAMMA_MIN, g_h, 1.0), 0.0)
            else:
                n_h_new = np.zeros(n_paths)
            n_s_new = N * d_w - n_h_new * d_h
        tr_s, tr_h = n_s_new - n_s_prev, n_h_new - n_h_prev
        cash = cash - tr_s * St - tr_h * c_h                 # self-financing: trades paid from the bank account
        value_after = -N * c_w + n_h_new * c_h + n_s_new * St + cash
        for k, v in (("n_s", n_s_new), ("n_h", n_h_new), ("trade_s", tr_s), ("trade_h", tr_h),
                     ("cash", cash), ("value", value_after), ("pnl", pnl), ("written_price", c_w),
                     ("hedge_price", c_h), ("delta_w", d_w), ("gamma_w", g_w), ("delta_h", d_h),
                     ("gamma_h", g_h), ("pf_delta", -N * d_w + n_h_new * d_h + n_s_new),
                     ("pf_gamma", -N * g_w + n_h_new * g_h)):
            out[k][:, t] = v
        out["tau_w"][:, t] = tau_w
        n_s_prev, n_h_prev, v_prev = n_s_new, n_h_new, value_after
    out["cum_pnl"] = np.cumsum(out["pnl"], axis=1)
    return out


def daily_report(eng, path=0, dates=None, hedge_kind="call"):
    """One row per day for one path + plain-English instructions."""
    n_t = eng["S"].shape[1]
    rows = []
    for t in range(n_t):
        ts, th = eng["trade_s"][path, t], eng["trade_h"][path, t]
        instr = []
        if abs(th) > 1e-9:
            instr.append(f"{action(th)} {abs(th):.4f} hedge {hedge_kind}s")
        if abs(ts) > 1e-9:
            instr.append(f"{action(ts)} {abs(ts):.4f} shares")
        label = f"Day {t}" + (f" ({dates[t]})" if dates is not None else "")
        rows.append({
            "Day": t, "Date": dates[t] if dates is not None else "",
            "Spot": eng["S"][path, t], "tau (y)": eng["tau_w"][path, t],
            "Written price": eng["written_price"][path, t],
            "Delta_w": eng["delta_w"][path, t], "Gamma_w": eng["gamma_w"][path, t],
            "Hedge shares": eng["n_s"][path, t], "Hedge options": eng["n_h"][path, t],
            "Trade shares": ts, "Trade options": th, "Cash": eng["cash"][path, t],
            "PF value": eng["value"][path, t], "Daily P&L": eng["pnl"][path, t],
            "Cum. P&L": eng["cum_pnl"][path, t],
            "Instruction": f"{label}: " + (", ".join(instr) if instr else "no trade"),
        })
    return pd.DataFrame(rows)


def simulate_gbm(S0, mu, sigma, days, n_paths=1, seed=42, dt=DT_DAY, q=0.0):
    """Spot Price Simulator (Hull 2021 GBM, daily steps):
    S_{t+1} = S_t exp((mu - q - sigma^2/2) dt + sigma sqrt(dt) Z). Returns (n_paths, days+1)."""
    if not 1 <= days <= 5:
        raise ValueError("The simulator forecasts 1 to 5 days.")
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, days))
    steps = (mu - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(steps, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def pnl_distribution(final_pnls):
    """Summary statistics of final P&L by strategy: dict name -> array."""
    rows = []
    for name, x in final_pnls.items():
        x = np.asarray(x, dtype=float)
        rows.append({"Strategy": name, "Mean": x.mean(), "Std": x.std(ddof=1) if x.size > 1 else 0.0,
                     "5%": np.percentile(x, 5), "95%": np.percentile(x, 95),
                     "Min": x.min(), "Max": x.max()})
    return pd.DataFrame(rows)

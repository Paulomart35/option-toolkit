# Module 3 - Delta and Delta-Gamma Hedging (HW7 conventions) + daily rebalancing + Spot Price Simulator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core import hedging as hg
from core.market_data import MarketDataError
from core.reports import make_pdf
from ui.common import (AMBER, GREEN, INK, RED, SLATE, VIOLET, action_color, cached_closes, downloads, line,
                       money, page_header, pdf_report, show_fig, spot_input, style_fig, table)

page_header("Delta & delta-gamma hedging", "We are short N written options. Compute the hedge today, then "
            "rebalance it every day on historical or simulated spot prices.", "hedging")

with st.expander("Market and written option", expanded=True, icon=":material/tune:"):
    S, ticker, src = spot_input("hdg", default=100.0, ticker="AAPL")
    c1, c2 = st.columns(2)
    r = c1.number_input("r (decimal, e.g. 0.03)", value=0.03, step=0.005, format="%.4f", key="hdg_r")
    sigma = c2.number_input("σ volatility (decimal)", value=0.20, min_value=0.0001, step=0.01, format="%.4f", key="hdg_sig")
    w_kind = c1.selectbox("Written option type", ["call", "put"], key="hdg_wk")
    w_K = c2.number_input("Written strike K ($)", value=100.0, min_value=0.01, step=1.0, key="hdg_wK")
    w_T = c1.number_input("Written maturity T (years)", value=1.0, min_value=0.01, step=0.25, format="%.4f", key="hdg_wT")
    N = c2.number_input("N written options (sold)", value=10.0, min_value=0.0001, step=1.0, key="hdg_N")

with st.expander("Hedging strategy", expanded=True, icon=":material/shield:"):
    strategy = st.radio("Hedge with", ["delta-gamma", "delta"], horizontal=True, key="hdg_strat",
                        format_func=lambda s: {"delta": "Delta (underlying only)",
                                               "delta-gamma": "Delta-gamma (hedging option + underlying)"}[s])
    st.caption("Hedging option (used by the delta-gamma hedge and in the strategy comparisons)")
    c1, c2 = st.columns(2)
    h_kind = c1.selectbox("Hedging option type", ["call", "put"], key="hdg_hk")
    h_K = c2.number_input("Hedging strike K ($)", value=110.0, min_value=0.01, step=1.0, key="hdg_hK")
    h_T = c1.number_input("Hedging maturity T (years)", value=2.0, min_value=0.01, step=0.25, format="%.4f", key="hdg_hT")
    override = c2.number_input("Number of hedging options (0 = computed)", value=0.0, step=1.0, key="hdg_ovr",
                               help="HW7 option: force a quantity; Gamma is then not fully neutralised.")

written = {"kind": w_kind, "K": w_K, "T": w_T}
hedge = {"kind": h_kind, "K": h_K, "T": h_T}
params = dict(S=S, r=r, sigma=sigma, N=N, written=written, strategy=strategy, hedge=hedge,
              n_override=override if override != 0 else None)

hedge_ok = True
try:
    hg.check_hedge_option(S, r, sigma, hedge)
except ValueError as e:
    hedge_ok = False
    if strategy == "delta-gamma":
        st.error(str(e))
        st.stop()
try:
    h = hg.compute_hedge(params)
except Exception as e:
    st.error(f"Could not compute the hedge: {e}")
    st.stop()

STRATS = ["unhedged", "delta"] + (["delta-gamma"] if hedge_ok else [])
pdf_blocks = {}
tab1, tab2, tab3 = st.tabs(["Static hedge (today)", "Historical rebalancing", "Forecast simulator"])

# ---------------- Static hedge (HW7 cells 3-5) ----------------
with tab1:
    c1, c2 = st.columns(2)
    c1.metric("Delta of written option", f"{h['d_w']:.4f}")
    c2.metric("Gamma of written option", f"{h['g_w']:.6f}")
    if strategy == "delta-gamma":
        c1.metric("Delta of hedging option", f"{h['d_h']:.4f}")
        c2.metric("Gamma of hedging option", f"{h['g_h']:.6f}")
    rep = [{"Instrument": f"Written {w_kind} K={w_K:g} T={w_T:g}", "Action": "SELL (already written)",
            "Quantity": N, "Delta": h["d_w"], "Gamma": h["g_w"]}]
    if strategy == "delta-gamma":
        rep.append({"Instrument": f"Hedging {h_kind} K={h_K:g} T={h_T:g}", "Action": hg.action(h["n_h"]),
                    "Quantity": abs(h["n_h"]), "Delta": h["d_h"], "Gamma": h["g_h"]})
    rep.append({"Instrument": "Underlying shares", "Action": hg.action(h["n_s"]), "Quantity": abs(h["n_s"]),
                "Delta": 1.0, "Gamma": 0.0})
    rep = pd.DataFrame(rep)
    st.markdown("**Hedging report**")
    st.dataframe(rep.style.format({"Quantity": "{:.4f}", "Delta": "{:.4f}", "Gamma": "{:.6f}"})
                 .map(action_color, subset=["Action"]), width="stretch", hide_index=True)
    st.caption(f"Residual portfolio delta = {h['port_delta']:.6f}, gamma = {h['port_gamma']:.6f}")
    if strategy == "delta-gamma" and abs(h["port_gamma"]) > 1e-6:
        st.warning("Manual quantity: Gamma is not fully neutralised.")

    move = st.slider("Instant spot move range (±%)", 5, 50, 30, key="hdg_move") / 100
    sc = hg.run_scenarios(params, h, move=move)
    worst_h, worst_u = sc["P&L hedged"].abs().max(), sc["P&L unhedged"].abs().max()
    c1, c2 = st.columns(2)
    c1.metric("Max |P&L| unhedged", money(worst_u))
    c2.metric("Max |P&L| hedged", money(worst_h),
              delta=f"{100 * (1 - worst_h / worst_u):.1f}% risk reduction" if worst_u > 0 else None)
    fig = make_subplots(rows=2, cols=2, subplot_titles=("P&L vs spot", "Hedged portfolio value",
                                                        "Portfolio delta", "Portfolio gamma"),
                        vertical_spacing=0.14, horizontal_spacing=0.1)
    fig.add_trace(line(sc["S"], sc["P&L unhedged"], "Unhedged", RED, "dash"), 1, 1)
    fig.add_trace(line(sc["S"], sc["P&L hedged"], "Hedged", GREEN, width=3), 1, 1)
    fig.add_trace(line(sc["S"], sc["PF value"], "PF value", INK), 1, 2)
    fig.add_trace(line(sc["S"], sc["PF delta"], "PF delta", VIOLET), 2, 1)
    fig.add_trace(line(sc["S"], sc["PF gamma"], "PF gamma", AMBER), 2, 2)
    for rr in (1, 2):
        for cc in (1, 2):
            fig.add_vline(S, line_dash="dot", line_color="grey", row=rr, col=cc)
    show_fig(style_fig(fig, None, None, None, height=620))
    matrix = sc.iloc[::6].reset_index(drop=True)
    st.markdown("**Scenario P&L grid** (static hedge, instant move)")
    st.dataframe(matrix.style.format("{:+.4f}").format({"S": "{:.2f}"})
                 .background_gradient(subset=["P&L hedged", "P&L unhedged"], cmap="RdYlGn"),
                 width="stretch", hide_index=True)
    downloads(sc, "hedge_scenarios", "hdg_sc", sheets={"report": rep, "scenarios": sc})

    def _static_fig():
        f, ax = plt.subplots(2, 2, figsize=(11.69, 8.27))
        f.suptitle(f"{strategy.upper()} hedging of {N:g} written {w_kind}s")
        ax[0, 0].plot(sc["S"], sc["P&L unhedged"], "r--", label="Unhedged")
        ax[0, 0].plot(sc["S"], sc["P&L hedged"], "g-", lw=2, label="Hedged"); ax[0, 0].set_title("P&L vs spot")
        ax[0, 1].plot(sc["S"], sc["PF value"], "b-"); ax[0, 1].set_title("Hedged portfolio value")
        ax[1, 0].plot(sc["S"], sc["PF delta"], color="purple"); ax[1, 0].set_title("Portfolio delta")
        ax[1, 1].plot(sc["S"], sc["PF gamma"], color="darkorange"); ax[1, 1].set_title("Portfolio gamma")
        for a in ax.ravel():
            a.axvline(S, color="grey", ls=":"); a.axhline(0, color="black", lw=0.6); a.grid(alpha=0.3)
        ax[0, 0].legend(); f.tight_layout()
        return f

    pdf_blocks["static"] = lambda: [("title", "Static hedge (t = 0)"), ("text", "\n".join(hg.report_lines(params, h))),
                                    ("table", ("Hedging report", rep)), ("figure", _static_fig()),
                                    ("table", ("Scenario P&L grid", matrix))]


def engine_runs(spots):
    out = {}
    for s_ in STRATS:
        out[s_] = hg.rebalance_engine(spots, r, sigma, N, written, s_, hedge if s_ == "delta-gamma" else None)
    return out


def cum_pnl_fig(runs, x, title, path=0):
    colors = {"unhedged": RED, "delta": AMBER, "delta-gamma": GREEN}
    fig = go.Figure([line(x, runs[s_]["cum_pnl"][path], s_.title(), colors[s_]) for s_ in runs])
    fig.add_hline(0, line_color="black", line_width=0.6)
    return style_fig(fig, title, "Day", "Cumulative P&L ($)")


def report_block(runs, dates=None, path=0, key="x"):
    eng = runs[strategy]
    rep_d = hg.daily_report(eng, path, dates, hedge_kind=h_kind)
    st.markdown(f"**Daily adjustment report: {strategy} hedge**")
    for txt in rep_d["Instruction"]:
        st.markdown(f"- {txt}")
    cols = ["Day", "Date", "Spot", "tau (y)", "Written price", "Delta_w", "Hedge shares", "Hedge options",
            "Trade shares", "Trade options", "Cash", "PF value", "Daily P&L", "Cum. P&L"]
    if dates is None:
        cols.remove("Date")
    if strategy == "delta":
        cols = [c for c in cols if c not in ("Hedge options", "Trade options")]
    st.dataframe(rep_d[cols].style.format({c: "{:,.4f}" for c in cols if c not in ("Day", "Date")})
                 .map(lambda v: f"color: {GREEN}" if v > 1e-9 else (f"color: {RED}" if v < -1e-9 else ""),
                      subset=[c for c in ("Trade shares", "Trade options", "Daily P&L") if c in cols]),
                 width="stretch", hide_index=True)
    downloads(rep_d, f"daily_report_{key}", f"hdg_{key}_rep")
    return rep_d


def _pdf_path_fig(x, runs, title, path=0):
    f, ax = plt.subplots(2, 1, figsize=(8.27, 8))
    ax[0].plot(x, runs[strategy]["S"][path], "b-o"); ax[0].set_title("Spot path"); ax[0].grid(alpha=0.3)
    for s_, sty in zip(runs, ["r--", "y-", "g-"]):
        ax[1].plot(x, runs[s_]["cum_pnl"][path], sty, label=s_)
    ax[1].axhline(0, color="black", lw=0.6); ax[1].set_title(title); ax[1].legend(); ax[1].grid(alpha=0.3)
    f.tight_layout()
    return f


# ---------------- Historical mode ----------------
with tab2:
    st.caption("Replay the hedge on real daily closes. Day 0 = first close: the hedge is set up there and "
               "rebalanced at every following close (τ = T − t/252).")
    mode = st.radio("Spot path", ["Type or paste closes", "Fetch from Yahoo Finance"], horizontal=True, key="hdg_hmode")
    dates, closes = None, None
    if mode == "Fetch from Yahoo Finance":
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        n_days = c1.slider("Number of trading days (≤ 30)", 2, 30, 10, key="hdg_ndays")
        if c2.button("Fetch closes", key="hdg_fetch_hist", icon=":material/cloud_download:"):
            try:
                st.session_state["hdg_hist"] = cached_closes(ticker, n_days)
            except MarketDataError as e:
                st.warning(f"{e} Switch to 'Type or paste closes'.")
        if st.session_state.get("hdg_hist"):
            dates, closes = st.session_state["hdg_hist"]
            st.caption(f"Yahoo Finance closes for {ticker.upper()}: {dates[0]} to {dates[-1]}")
        else:
            st.info("Click 'Fetch closes', or type the closes manually.")
    else:
        txt = st.text_area("Daily closes, oldest first (comma or newline separated, 2 to 31 values)",
                           value="100, 101.2, 99.8, 100.5, 102.1, 101.4, 100.9, 102.6, 103.0, 102.2",
                           key="hdg_closes")
        try:
            closes = [float(x) for x in txt.replace("\n", ",").replace(";", ",").split(",") if x.strip()]
            if not 2 <= len(closes) <= 31 or min(closes) <= 0:
                raise ValueError
        except ValueError:
            st.error("Enter between 2 and 31 positive numbers separated by commas.")
            closes = None
    if closes:
        hist_runs = engine_runs(np.array(closes))
        x_h = np.arange(len(closes))
        c1, c2 = st.columns(2)
        final = {s_: hist_runs[s_]["cum_pnl"][0, -1] for s_ in hist_runs}
        c1.metric(f"Final P&L, {strategy}", money(final[strategy]))
        c2.metric("Final P&L, unhedged", money(final["unhedged"]))
        fig = go.Figure([line(x_h, closes, "Close", INK)])
        show_fig(style_fig(fig, "Spot path", "Day", "$", height=280))
        show_fig(cum_pnl_fig(hist_runs, x_h, "Cumulative P&L by strategy"))
        hist_rep = report_block(hist_runs, dates, 0, "hist")
        pdf_blocks["hist"] = lambda: [("title", "Historical rebalancing"),
                                      ("text", "Spot path: " + ", ".join(f"{c:.2f}" for c in closes)),
                                      ("text", "Final P&L: " + "; ".join(f"{k} = {v:,.2f}" for k, v in final.items())),
                                      ("figure", _pdf_path_fig(x_h, hist_runs, "Cumulative P&L (historical)")),
                                      ("table", ("Daily adjustment report (historical)", hist_rep.drop(columns=["Instruction"]))),
                                      ("text", "\n".join(hist_rep["Instruction"]))]

# ---------------- Forecast mode (Spot Price Simulator) ----------------
with tab3:
    st.caption("Simulate daily spots with a geometric Brownian motion (Hull 2021), dt = 1/252, starting from S "
               "above, then rebalance the hedge every day on each path.")
    c1, c2 = st.columns(2)
    days = c1.slider("Days to simulate", 1, 5, 5, key="hdg_days")
    mu = c2.number_input("Drift μ (decimal, default r)", value=float(r), step=0.01, format="%.4f", key="hdg_mu")
    n_paths = c1.number_input("Number of paths (1-1000)", value=200, min_value=1, max_value=1000, step=50, key="hdg_np")
    seed = c2.number_input("Random seed", value=42, min_value=0, step=1, key="hdg_seed")
    try:
        spots = hg.simulate_gbm(S, mu, sigma, days, int(n_paths), int(seed))
        fc_runs = engine_runs(spots)
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        st.stop()
    x = np.arange(days + 1)
    fig = go.Figure()
    for i in range(min(int(n_paths), 100)):
        fig.add_trace(go.Scatter(x=x, y=spots[i], mode="lines", line=dict(color=SLATE, width=1), opacity=0.25,
                                 showlegend=False, hoverinfo="skip"))
    for qq, nm, col in ((5, "5th percentile", RED), (50, "Median", INK), (95, "95th percentile", GREEN)):
        fig.add_trace(line(x, np.percentile(spots, qq, axis=0), nm, col, width=3))
    show_fig(style_fig(fig, f"Simulated spot paths ({int(n_paths)} paths, fan chart)", "Day", "$"))

    finals = {s_: fc_runs[s_]["cum_pnl"][:, -1] for s_ in fc_runs}
    stats = hg.pnl_distribution(finals)
    st.markdown(f"**P&L distribution after {days} day(s)**")
    table(stats, "{:+,.2f}")
    if int(n_paths) > 1:
        fig = go.Figure()
        colors = {"unhedged": RED, "delta": AMBER, "delta-gamma": GREEN}
        for s_, v in finals.items():
            fig.add_trace(go.Histogram(x=v, name=s_.title(), marker_color=colors[s_], opacity=0.6, nbinsx=40))
        fig.update_layout(barmode="overlay")
        show_fig(style_fig(fig, "Final P&L by strategy", "P&L ($)", "Number of paths"))
    downloads(stats, "forecast_pnl_stats", "hdg_fc_stats",
              sheets={"stats": stats, "spots": pd.DataFrame(spots, columns=[f"Day {i}" for i in x]),
                      **{f"final_pnl_{k}": pd.DataFrame({"P&L": v}) for k, v in finals.items()}})
    path = st.number_input("Path for the detailed report", value=1, min_value=1, max_value=int(n_paths), step=1,
                           key="hdg_path") - 1
    show_fig(cum_pnl_fig(fc_runs, x, f"Cumulative P&L, path {path + 1}", path))
    fc_rep = report_block(fc_runs, None, path, "fc")

    def _fc_fig():
        f, ax = plt.subplots(2, 1, figsize=(8.27, 9))
        for i in range(min(int(n_paths), 100)):
            ax[0].plot(x, spots[i], color="grey", alpha=0.2, lw=0.8)
        for qq, sty in ((5, "r-"), (50, "b-"), (95, "g-")):
            ax[0].plot(x, np.percentile(spots, qq, axis=0), sty, lw=2, label=f"{qq}%")
        ax[0].set_title("Simulated spot paths"); ax[0].legend(); ax[0].grid(alpha=0.3)
        for s_, v in finals.items():
            ax[1].hist(v, bins=40, alpha=0.5, label=s_)
        ax[1].set_title("Final P&L by strategy"); ax[1].legend(); ax[1].grid(alpha=0.3)
        f.tight_layout()
        return f

    pdf_blocks["fc"] = lambda: [("title", "Forecast (Spot Price Simulator)"),
                                ("text", f"GBM: S0 = {S:.4f}, mu = {mu:.4f}, sigma = {sigma:.4f}, days = {days}, "
                                         f"paths = {int(n_paths)}, seed = {int(seed)}, dt = 1/252"),
                                ("table", ("P&L distribution", stats)), ("figure", _fc_fig()),
                                ("figure", _pdf_path_fig(x, fc_runs, f"Cumulative P&L, path {path + 1}", path)),
                                ("table", (f"Daily adjustment report (path {path + 1})", fc_rep.drop(columns=["Instruction"]))),
                                ("text", "\n".join(fc_rep["Instruction"]))]

st.divider()


def build_pdf():
    blocks = [("title", "Hedging inputs"),
              ("text", f"S = {S:.4f} ({src}); r = {r:.4f}; sigma = {sigma:.4f}; short {N:g} {w_kind}s K = {w_K:g}, T = {w_T:g}"),
              ("text", f"Strategy: {strategy}; hedging option: {h_kind} K = {h_K:g}, T = {h_T:g}")]
    for k in ("static", "hist", "fc"):
        if k in pdf_blocks:
            blocks += pdf_blocks[k]()
    return make_pdf("Option Toolkit - Hedging report", blocks)


pdf_report(build_pdf, "hedging_report.pdf", "hdg_pdf")

# Module 4 - SABR Calculator (Hagan 2002 as in Alos & Garcia 2021; HW8 pipeline)
import hashlib
import io
import pathlib
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import sabr
from core.black_scholes import black76_price, phi_of
from core.parity import parity_analysis, build_strategy
from core.reports import make_pdf
from ui.common import (GREEN, INK, RED, SERIES, action_color, downloads, line, money, page_header,
                       pdf_report, show_fig, spot_input, style_fig, table)

SAMPLE = pathlib.Path(__file__).resolve().parents[1] / "sample_data" / "sabr_sample.csv"

page_header("SABR calculator", "Hagan (2002) implied volatility, calibration per maturity with Differential "
            "Evolution, smiles, pricing and mispriced options. α = vol level, ν = vol-of-vol.", "sabr")


@st.cache_data(show_spinner="Reading the file...")
def read_table(data, name):
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


with st.expander("Market inputs", expanded=True, icon=":material/tune:"):
    S, ticker, src = spot_input("sabr", default=100.0, ticker="TSLA")
    c1, c2 = st.columns(2)
    r = c1.number_input("r (decimal, e.g. 0.05)", value=0.05, step=0.005, format="%.5f", key="sabr_r")
    val_date = c2.date_input("Valuation date", value=date.today(), key="sabr_date")
    st.caption("Forward of a non-dividend stock: F = S·e^{rT}.")

tab_calc, tab_cal, tab_smile, tab_grid, tab_price, tab_mis, tab_3d = st.tabs(
    ["Hagan calculator", "Calibration", "Smile", "S/F grid", "Pricing", "Mispriced option", "3D surface"])

# ---------------- Hagan calculator (manual parameters) ----------------
with tab_calc:
    st.caption("Implied volatility from Hagan's formula for parameters you type (default: HW8 check values).")
    c1, c2 = st.columns(2)
    use_F = c1.radio("Underlying input", ["Forward F", "Spot S (F = S e^{rT})"], key="hc_mode")
    T_c = c2.number_input("T (years)", value=0.5, min_value=0.001, step=0.25, format="%.4f", key="hc_T")
    if use_F == "Forward F":
        F_c = c1.number_input("F ($)", value=410.0, min_value=0.01, key="hc_F")
    else:
        F_c = float(sabr.forward(S, r, T_c))
        c1.metric("F = S e^{rT}", f"{F_c:.4f}")
    a_c = c2.number_input("α (vol level)", value=0.55, min_value=0.0001, step=0.01, format="%.4f", key="hc_a")
    b_c = c1.number_input("β (elasticity, 1 for equities)", value=1.0, min_value=0.0, max_value=1.0, step=0.1, key="hc_b")
    rho_c = c2.number_input("ρ (correlation)", value=-0.4, min_value=-0.999, max_value=0.999, step=0.05, key="hc_rho")
    nu_c = c1.number_input("ν (vol-of-vol)", value=1.2, min_value=0.0, step=0.05, format="%.4f", key="hc_nu")
    ks = c2.text_input("Strikes K (comma-separated)", value="300, 350, 410, 450, 500", key="hc_K")
    try:
        Kc = np.array([float(x) for x in ks.split(",") if x.strip()])
        if Kc.size == 0 or Kc.min() <= 0:
            raise ValueError
        vols = np.atleast_1d(sabr.sabr_vol(Kc, F_c, T_c, a_c, b_c, rho_c, nu_c))
        disc_r = r
        calc_tab = pd.DataFrame({"K": Kc, "F": F_c, "T": T_c, "σ Hagan": vols,
                                 "Call (Black-76)": black76_price(F_c, Kc, vols, disc_r, T_c, 1),
                                 "Put (Black-76)": black76_price(F_c, Kc, vols, disc_r, T_c, -1)})
        table(calc_tab, "{:,.6f}")
        downloads(calc_tab, "hagan_calculator", "hc_dl")
        kk = np.linspace(Kc.min() * 0.9, Kc.max() * 1.1, 120)
        fig = go.Figure([line(kk, sabr.sabr_vol(kk, F_c, T_c, a_c, b_c, rho_c, nu_c), "Hagan σ(K)", INK),
                         go.Scatter(x=Kc, y=vols, mode="markers", name="Your strikes", marker=dict(color=RED, size=9))])
        fig.add_vline(F_c, line_dash="dot", line_color="grey")
        show_fig(style_fig(fig, "Hagan implied volatility smile", "Strike K", "σ"))
    except ValueError:
        st.error("Strikes must be positive numbers separated by commas.")

# ---------------- Data + calibration ----------------
with tab_cal:
    source = st.radio("Data", ["Sample data (generated, S = 100, r = 5%)", "Upload CSV / Excel"], key="sabr_src")
    df, data_label = None, ""
    try:
        if source.startswith("Sample"):
            raw = pd.read_csv(SAMPLE)
            df = sabr.clean_simple_file(raw, S, r, val_date)
            data_label = "sample_data/sabr_sample.csv (Hagan vols for 3 maturities + noise)"
            if abs(S - 100) > 1e-9 or abs(r - 0.05) > 1e-9:
                st.info("The sample was generated with S = 100 and r = 0.05: use these values for a clean fit.")
        else:
            up = st.file_uploader("Course file (Final_Database_Options_.xlsx) or simple file: strike, T (or expiry), "
                                  "impliedVolatility (or price)", type=["csv", "xlsx", "xls"], key="sabr_up")
            if up is not None:
                raw = read_table(up.getvalue(), up.name)
                if sabr.is_course_format(raw):
                    tks = sabr.course_tickers(raw)
                    tk = st.selectbox("Stock in the file", tks, index=tks.index("TSLA") if "TSLA" in tks else 0,
                                      key="sabr_tk")
                    st.caption("HW8 used the last snapshot (17 Nov 2025) with TSLA 408.92, AAPL 267.46, MSFT 507.49 "
                               "and r = 5.985%: set S, r and the valuation date above accordingly.")
                    df = sabr.clean_course_file(raw, tk, S, r, val_date)
                    data_label = f"{up.name}, {tk}, course format (HW8 cleaning)"
                else:
                    df = sabr.clean_simple_file(raw, S, r, val_date)
                    data_label = f"{up.name}, simple format"
            else:
                st.info("Upload a file, or switch to the sample data.")
    except Exception as e:
        st.error(f"Could not read the data: {e}")
        df = None

    calib = None
    if df is not None:
        if df.empty:
            st.error("No option left after cleaning (check S, r, the valuation date and the columns).")
        else:
            st.caption(f"{len(df)} options kept, {df['T'].nunique()} maturities. Source: {data_label}")
            with st.expander("Cleaned data", icon=":material/table_rows:"):
                table(df)
            c1, c2 = st.columns(2)
            free_beta = c1.toggle("Calibrate β (off: β = 1)", value=False, key="sabr_fb")
            seed = c2.number_input("Seed", value=42, step=1, key="sabr_seed")
            pop = c1.number_input("DE population", value=15, min_value=5, max_value=100, step=5, key="sabr_pop")
            its = c2.number_input("DE iterations", value=60, min_value=5, max_value=1000, step=10, key="sabr_its")
            st.caption("HW8 bounds: α ∈ [0.001, 5], β ∈ [0, 1], ρ ∈ [−0.999, 0.999], ν ∈ [0.001, 5]; "
                       "mutation 0.8, crossover 0.7. Increase population/iterations for a finer fit.")
            sig = hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()
                              + repr((free_beta, seed, pop, its)).encode()).hexdigest()
            if st.session_state.get("sabr_calib_sig") != sig:
                bar = st.progress(0.0, text="Calibrating...")
                try:
                    out = sabr.calibrate(df, beta=None if free_beta else 1.0, popsize=int(pop), its=int(its),
                                         seed=int(seed), progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
                    st.session_state["sabr_calib"] = out
                    st.session_state["sabr_calib_sig"] = sig
                finally:
                    bar.empty()
            calib = st.session_state.get("sabr_calib")
            st.markdown("**Calibrated parameters per maturity**")
            table(calib, "{:,.5f}")
            st.caption(f"Average RMSE: {100 * calib['RMSE'].mean():.2f} vol points; "
                       f"max: {100 * calib['RMSE'].max():.2f}.")
            downloads(calib, "sabr_parameters", "sabr_par", sheets={"parameters": calib, "data": df})

# parameters used by the other tabs
if calib is not None and not calib.empty:
    labels = [f"T = {t:.4f} y ({e})" for t, e in zip(calib["T"], calib["Expiry"])]
    sel = st.selectbox("Maturity used for the S/F grid, pricing and mispricing tabs", range(len(labels)),
                       format_func=lambda i: labels[i], key="sabr_sel")
    row = calib.iloc[sel]
    p = {k: float(row[k]) for k in ("alpha", "beta", "rho", "nu")}
    T_sel = float(row["T"])
else:
    p = {"alpha": a_c, "beta": b_c, "rho": rho_c, "nu": nu_c}
    T_sel = T_c
    st.caption("No calibration available: the other tabs use the Hagan calculator parameters.")
F_sel = float(sabr.forward(S, r, T_sel))

# ---------------- Smile ----------------
smile_fig_data = []
with tab_smile:
    if calib is None or df is None:
        st.info("Run a calibration first (Calibration tab).")
    else:
        fig = go.Figure()
        for i, (_, row) in enumerate(calib.iterrows()):
            g = df[np.isclose(df["T"], row["T"])]
            col = SERIES[i % len(SERIES)]
            kk = np.linspace(g["strike"].min(), g["strike"].max(), 100)
            vv = sabr.sabr_vol(kk, row["F"], row["T"], row["alpha"], row["beta"], row["rho"], row["nu"])
            fig.add_trace(go.Scatter(x=g["strike"], y=g["iv"], mode="markers", name=f"Market T={row['T']:.3f}",
                                     marker=dict(color=col, size=7)))
            fig.add_trace(line(kk, vv, f"SABR T={row['T']:.3f}", col))
            smile_fig_data.append((g["strike"].values, g["iv"].values, kk, vv, row["T"]))
        show_fig(style_fig(fig, "Market implied vol (dots) vs calibrated SABR (lines)", "Strike K", "σ", height=460))

# ---------------- S / F grid ----------------
with tab_grid:
    st.caption(f"σ_SABR(K) for several spots S, with F = S·e^{{rT}} and the parameters of T = {T_sel:.4f} y.")
    c1, c2 = st.columns(2)
    s_lo = c1.number_input("Lowest spot S", value=round(0.9 * S, 2), min_value=0.01, key="grid_lo")
    s_hi = c2.number_input("Highest spot S", value=round(1.1 * S, 2), min_value=0.01, key="grid_hi")
    n_s = c1.slider("Number of spots", 2, 9, 5, key="grid_n")
    n_k = c2.slider("Number of strikes", 5, 41, 17, key="grid_nk")
    S_vals = np.linspace(min(s_lo, s_hi), max(s_lo, s_hi), n_s)
    K_grid = np.round(np.linspace(0.8 * F_sel, 1.2 * F_sel, n_k), 2)
    grid = sabr.smile_grid(S_vals, K_grid, T_sel, r, p)
    fig = go.Figure()
    for i, (s_v, g) in enumerate(grid.groupby("S")):
        fig.add_trace(line(g["K"], g["sigma_SABR"], f"S={s_v:.2f}, F={g['F'].iloc[0]:.2f}", SERIES[i % len(SERIES)]))
    show_fig(style_fig(fig, "SABR smile across spot / forward", "Strike K", "σ_SABR", height=440))
    pivot = grid.pivot_table(index="K", columns="S", values="sigma_SABR")
    pivot.columns = [f"S={s_v:.2f} (F={sabr.forward(s_v, r, T_sel):.2f})" for s_v in pivot.columns]
    pivot = pivot.reset_index()
    table(pivot, "{:.5f}")
    downloads(grid, "sabr_smile_grid", "grid_dl", sheets={"grid_long": grid, "grid_table": pivot})

# ---------------- Pricing ----------------
with tab_price:
    st.caption(f"Black-Scholes with Hagan's vol (= Black-76 on F, discounted) vs Black-Scholes with the flat ATM "
               f"vol. T = {T_sel:.4f} y, F = {F_sel:.4f}.")
    kind = st.radio("Option type", ["call", "put"], horizontal=True, key="pr_kind")
    default_k = ", ".join(f"{k:g}" for k in np.round(np.linspace(0.85 * F_sel, 1.15 * F_sel, 7), 2))
    kt = st.text_input("Strikes K (comma-separated)", value=default_k, key="pr_K")
    try:
        Kp = [float(x) for x in kt.split(",") if x.strip()]
        if not Kp or min(Kp) <= 0:
            raise ValueError
        price_tab = sabr.pricing_table(Kp, S, T_sel, r, p, phi_of(kind))
        table(price_tab, "{:,.4f}")
        params_tab = pd.DataFrame([{"T": T_sel, **p}])
        downloads(price_tab, "sabr_pricing", "pr_dl", sheets={"pricing": price_tab, "parameters": params_tab})
        fig = go.Figure([line(price_tab["K"], price_tab["SABR price"], "SABR (Hagan vol)", INK),
                         line(price_tab["K"], price_tab["BS price (flat ATM vol)"], "BS flat ATM vol", "#C98B12", "dash")])
        show_fig(style_fig(fig, f"{kind.title()} prices", "Strike K", "Price ($)"))
    except ValueError:
        st.error("Strikes must be positive numbers separated by commas.")
        price_tab = pd.DataFrame()

# ---------------- Mispriced option ----------------
mis = None
with tab_mis:
    st.warning("Model-based (relative value) arbitrage: it profits only if SABR is right and the hedge is "
               "rebalanced. It is NOT riskless. The riskless parity arbitrage is in the Arbitrage calculator.",
               icon=":material/info:")
    c1, c2 = st.columns(2)
    m_kind = c1.selectbox("Option type", ["call", "put"], key="mis_kind")
    m_K = c2.number_input("Strike K ($)", value=round(F_sel, 2), min_value=0.01, key="mis_K")
    m_model = float(sabr.sabr_price(m_K, S, T_sel, r, p["alpha"], p["beta"], p["rho"], p["nu"], phi_of(m_kind))[0])
    m_px = c1.number_input("Market price ($)", value=round(m_model * 1.05, 4), min_value=0.0, format="%.4f",
                           key="mis_px", help="Default: 5% above the SABR price, to show a strategy.")
    m_tol = c2.number_input("Tolerance per option ($)", value=0.05, min_value=0.0, format="%.4f", key="mis_tol")
    m_n = c1.number_input("Contracts", value=1, min_value=1, step=1, key="mis_n")
    m_m = c2.number_input("Multiplier", value=100, min_value=1, step=1, key="mis_m")
    other = st.number_input(f"Optional: market price of the {'put' if m_kind == 'call' else 'call'} with the same "
                            "K and T (0 = not given) for the parity check", value=0.0, min_value=0.0, format="%.4f",
                            key="mis_other")
    st.caption(f"Maturity T = {T_sel:.4f} y (selected calibrated maturity); F = {F_sel:.4f}.")
    try:
        mis = sabr.mispricing_strategy(m_px, m_K, T_sel, S, r, p, phi_of(m_kind), m_tol, m_n, m_m)
        c1, c2 = st.columns(2)
        c1.metric("SABR price", f"{mis['model_price']:.4f}", help=f"Hagan σ = {mis['sigma_hagan']:.4%}")
        c2.metric("Market − SABR", f"{mis['gap']:+.4f}",
                  help=f"Market implied vol = {mis['market_iv']:.4%}" if np.isfinite(mis["market_iv"]) else None)
        if mis["decision"] == "fair":
            st.info(f"Within tolerance: no trade (|gap| ≤ {m_tol:.4f}).")
        else:
            st.success(f"The {m_kind} is {mis['decision'].upper()} vs SABR. Expected model profit "
                       f"{money(mis['expected_profit'])} if the market converges to the SABR price.")
            st.dataframe(mis["steps"].style.format({"Quantity": "{:,.4f}", "Cash flow at t=0": "{:+,.2f}"})
                         .map(action_color, subset=["Action"]), width="stretch", hide_index=True)
            st.caption(f"Delta (Hagan vol) = {mis['delta']:.4f}. Net cash at t = 0 is zero: the bond finances the "
                       "trades. Rebalance the delta hedge over time (see the Hedging module).")
            downloads(mis["steps"], "sabr_mispricing", "mis_dl")
        if other > 0:
            C_, P_ = (m_px, other) if m_kind == "call" else (other, m_px)
            par = parity_analysis(S, m_K, T_sel, r, C_, P_, tol=m_tol)
            if par["decision"] == "none":
                st.info(f"Parity check: consistent (gap {par['gap']:+.4f}).")
            else:
                stp, _, sm = build_strategy(par, m_n, m_m)
                st.success(f"Parity check: riskless arbitrage, gap {par['gap']:+.4f} per share. {sm['view_call']}. "
                           f"Locked-in profit {money(sm['profit_today'])}.")
                st.dataframe(stp.style.format({"Quantity": "{:,.4f}", "Cash flow at t=0": "{:+,.2f}"})
                             .map(action_color, subset=["Action"]), width="stretch", hide_index=True)
    except Exception as e:
        st.error(f"Could not analyse this option: {e}")

# ---------------- 3D surface ----------------
with tab_3d:
    if calib is None or df is None or df["T"].nunique() < 2:
        st.info("The surface needs a calibration with at least 2 maturities.")
    else:
        d3 = df.merge(calib[["T", "alpha", "beta", "rho", "nu"]], on="T", how="left")
        d3["iv_sabr"] = sabr.sabr_vol(d3["strike"].values, d3["F"].values, d3["T"].values, d3["alpha"].values,
                                      d3["beta"].values, d3["rho"].values, d3["nu"].values)
        Ks = (d3["strike"] - d3["strike"].min()) / max(d3["strike"].max() - d3["strike"].min(), 1e-12)
        tri = mtri.Triangulation(Ks, d3["T"] / d3["T"].max())
        i, j, k = tri.triangles.T
        fig = go.Figure([
            go.Mesh3d(x=d3["strike"], y=d3["T"], z=d3["iv"], i=i, j=j, k=k, color="#2E6FB0", opacity=0.45,
                      name="Market implied vol", showlegend=True),
            go.Mesh3d(x=d3["strike"], y=d3["T"], z=d3["iv_sabr"], i=i, j=j, k=k, color=RED, opacity=0.35,
                      name="Calibrated SABR", showlegend=True)])
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0), title="Implied volatility surface",
                          scene=dict(xaxis_title="Strike K", yaxis_title="T (years)", zaxis_title="σ"),
                          legend=dict(orientation="h", y=-0.05))
        show_fig(fig)


def build_pdf():
    blocks = [("title", "SABR calculator"),
              ("text", f"S = {S:.4f} ({src}); r = {r:.5f}; valuation date {val_date}"),
              ("text", f"Parameters used for pricing (T = {T_sel:.4f}): alpha = {p['alpha']:.5f}, beta = {p['beta']:.4f}, "
                       f"rho = {p['rho']:.5f}, nu = {p['nu']:.5f}; F = {F_sel:.4f}")]
    if calib is not None:
        blocks.append(("table", ("Calibrated parameters per maturity", calib)))
    if smile_fig_data:
        f, ax = plt.subplots(figsize=(8.27, 6))
        for idx, (km, ivm, kk, vv, t) in enumerate(smile_fig_data):
            c = f"C{idx}"
            ax.plot(km, ivm, "o", color=c, ms=4, label=f"Market T={t:.3f}")
            ax.plot(kk, vv, "-", color=c, label=f"SABR T={t:.3f}")
        ax.set_xlabel("Strike K"); ax.set_ylabel("Implied vol"); ax.set_title("Volatility smile: market vs SABR")
        ax.grid(alpha=0.3); ax.legend(fontsize=7); f.tight_layout()
        blocks.append(("figure", f))
    if not price_tab.empty:
        blocks.append(("table", ("Pricing (K, F, T, sigma Hagan, SABR price, BS price)", price_tab)))
    blocks.append(("table", ("Smile grid across S and F", pivot)))
    if mis is not None and mis["decision"] != "fair":
        blocks += [("text", f"Mispriced {m_kind} K = {m_K:g}: market {m_px:.4f} vs SABR {mis['model_price']:.4f} "
                            f"-> {mis['decision']} (model-based relative value arbitrage, not riskless)"),
                   ("table", ("Relative value strategy", mis["steps"]))]
    return make_pdf("Option Toolkit - SABR report", blocks)


st.divider()
pdf_report(build_pdf, "sabr_report.pdf", "sabr_pdf")

# Module 5 - SABR Monte Carlo (professor's std_MC, vectorised; Euler / Full truncation / Log transform)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import sabr_mc as mc
from core.black_scholes import phi_of
from core.reports import make_pdf
from ui.common import (AMBER, GREEN, INK, RED, SLATE, downloads, line, page_header, pdf_report, show_fig,
                       spot_input, style_fig, table)

page_header("SABR Monte Carlo", "Simulate the SABR equations to get the final price of the underlying and "
            "the present value of the average option payoff.", "sabr_mc")

EQS = {
    "euler": [r"\sigma_{t+\Delta t} = \sigma_t + \nu\,\sigma_t\sqrt{\Delta t}\,Z_1",
              r"X_{t+\Delta t} = X_t + \left(r - \tfrac{1}{2}\sigma_t^2\right)\Delta t + \sigma_t\sqrt{\Delta t}\,"
              r"\left(\rho Z_1 + \sqrt{1-\rho^2}\,Z_2\right)"],
    "full": [r"\sigma_t^{+} = \max(\sigma_t, 0),\qquad \sigma_{t+\Delta t} = \sigma_t + \nu\,\sigma_t^{+}\sqrt{\Delta t}\,Z_1",
             r"X_{t+\Delta t} = X_t + \left(r - \tfrac{1}{2}(\sigma_t^{+})^2\right)\Delta t + \sigma_t^{+}\sqrt{\Delta t}\,"
             r"\left(\rho Z_1 + \sqrt{1-\rho^2}\,Z_2\right)"],
    "log": [r"\sigma_{t+\Delta t} = \sigma_t\,\exp\!\left(\nu\sqrt{\Delta t}\,Z_1 - \tfrac{1}{2}\nu^2\Delta t\right)",
            r"X_{t+\Delta t} = X_t + \left(r - \tfrac{1}{2}\sigma_t^2\right)\Delta t + \sigma_t\sqrt{\Delta t}\,"
            r"\left(\rho Z_1 + \sqrt{1-\rho^2}\,Z_2\right)"],
}
EQ_BETA = (r"F_{t+\Delta t} = \max\!\left(F_t + \sigma_t^{(+)}\,(F_t^{+})^{\beta}\sqrt{\Delta t}\,"
           r"\left(\rho Z_1 + \sqrt{1-\rho^2}\,Z_2\right),\,0\right),\quad F_0 = S_0 e^{rT},\ S_T = F_T")

with st.expander("Inputs", expanded=True, icon=":material/tune:"):
    S0, ticker, src = spot_input("mc", default=100.0, ticker="AAPL", label="Initial price S₀ ($)")
    c1, c2 = st.columns(2)
    K = c1.number_input("Strike K ($)", value=100.0, min_value=0.01, step=1.0, key="mc_K")
    T = c2.number_input("T (years)", value=1.0, min_value=0.01, step=0.25, format="%.4f", key="mc_T")
    r = c1.number_input("r (decimal, e.g. 0.05)", value=0.05, step=0.005, format="%.4f", key="mc_r")
    kind = c2.selectbox("Option type", ["call", "put"], key="mc_kind")
    alpha = c1.number_input("α initial vol (prof.: initial_volatility)", value=0.20, min_value=0.0001, step=0.01,
                            format="%.4f", key="mc_a")
    nu = c2.number_input("ν vol-of-vol (prof.: vol_of_vol)", value=0.50, min_value=0.0, step=0.05, format="%.4f", key="mc_nu")
    rho = c1.number_input("ρ correlation", value=-0.40, min_value=-0.999, max_value=0.999, step=0.05, key="mc_rho")
    beta = c2.number_input("β (1 for equities)", value=1.0, min_value=0.0, max_value=1.0, step=0.1, key="mc_beta")
    n_sims = c1.number_input("Number of simulations", value=20000, min_value=100, max_value=2_000_000, step=10000,
                             key="mc_n")
    n_steps = c2.number_input("Time steps", value=100, min_value=1, max_value=5000, step=50, key="mc_steps")
    seed = c1.number_input("Random seed", value=42, min_value=0, step=1, key="mc_seed")
    scheme = c2.selectbox("Scheme", list(mc.SCHEMES), format_func=lambda s: mc.SCHEMES[s], index=2, key="mc_scheme")
    if beta < 1 and alpha < 0.5:
        st.caption("With β < 1, α is in price units: σ_lognormal ≈ α·F^(β−1). Increase α accordingly.")

st.markdown(f"**{mc.SCHEMES[scheme]} update equations** (X = ln S/S₀, Z₁, Z₂ independent N(0,1))")
for eq in EQS[scheme] if beta >= 1 else EQS[scheme][:1] + [EQ_BETA]:
    st.latex(eq)
st.caption("The professor's std_MC omits r: here the drift r is added and the payoff is discounted by e^{−rT}.")

work = int(n_sims) * int(n_steps)
if work > mc.MAX_WORK:
    st.warning(f"Simulations × steps = {work:,.0f} exceeds the cloud limit of {mc.MAX_WORK:,.0f}. "
               "Reduce the number of simulations or time steps.")
    st.stop()


@st.cache_data(show_spinner="Simulating paths...", max_entries=20)
def run(S0, K, T, r, alpha, nu, rho, beta, n, steps, scheme, seed, phi):
    return mc.simulate(S0, K, T, r, alpha, nu, rho, beta, n, steps, scheme, seed, phi)


@st.cache_data(show_spinner="Comparing the three schemes...", max_entries=20)
def run_compare(S0, K, T, r, alpha, nu, rho, beta, n, steps, seed, phi):
    return mc.compare_schemes(S0, K, T, r, alpha, nu, rho, beta, n, steps, seed, phi)


phi = phi_of(kind)
try:
    res = run(S0, K, T, r, alpha, nu, rho, beta, int(n_sims), int(n_steps), scheme, int(seed), phi)
    bench = mc.benchmarks(S0, K, T, r, alpha, nu, rho, beta, phi)
except Exception as e:
    st.error(f"Simulation failed: {e}")
    st.stop()

c1, c2 = st.columns(2)
c1.metric("Mean final price E[S_T]", f"{res['mean_ST']:.4f}",
          delta=f"{res['mean_ST'] - res['forward']:+.4f} vs S₀e^(rT) = {res['forward']:.4f}", delta_color="off")
c2.metric(f"{kind.title()} price (MC)", f"{res['price']:.4f}", help="Discounted average payoff")
c1.metric("Standard error", f"{res['se']:.4f}")
c2.metric("95% confidence interval", f"[{res['ci95'][0]:.4f}, {res['ci95'][1]:.4f}]")
c1.metric("Black price with Hagan vol", f"{bench['black_hagan']:.4f}", help=f"Hagan σ = {bench['hagan_vol']:.4%}")
c2.metric("BS price with σ = α", f"{bench['bs_alpha']:.4f}")
z = (res["price"] - bench["black_hagan"]) / res["se"] if res["se"] > 0 else 0.0
if abs(z) <= 3:
    st.success(f"MC price is within {abs(z):.1f} standard errors of the Hagan/Black price.")
else:
    st.info(f"MC price differs from the Hagan/Black price by {abs(z):.1f} standard errors (Hagan's formula is an "
            "approximation; discretisation also adds bias: try more steps).")
if res["neg_vol_share"] > 0:
    st.warning(f"{res['neg_vol_share']:.2%} of paths end with a negative volatility (Euler artefact).")

t1, t2, t3, t4, t5 = st.tabs(["Paths", "Histogram of S_T", "Convergence", "Scheme comparison", "Data"])
tgrid = np.linspace(0, T, int(n_steps) + 1)
with t1:
    fig = go.Figure()
    for i in range(res["S_paths"].shape[0]):
        fig.add_trace(go.Scatter(x=tgrid, y=res["S_paths"][i], mode="lines", line=dict(width=1), opacity=0.6,
                                 showlegend=False, hoverinfo="skip"))
    fig.add_hline(K, line_dash="dot", line_color=RED, annotation_text="K")
    show_fig(style_fig(fig, f"Sample paths of S (first {res['S_paths'].shape[0]})", "t (years)", "S"))
    fig = go.Figure()
    for i in range(res["v_paths"].shape[0]):
        fig.add_trace(go.Scatter(x=tgrid, y=res["v_paths"][i], mode="lines", line=dict(width=1), opacity=0.6,
                                 showlegend=False, hoverinfo="skip"))
    fig.add_hline(0, line_color="black", line_width=0.6)
    show_fig(style_fig(fig, "Sample paths of the volatility σ", "t (years)", "σ"))
with t2:
    fig = go.Figure(go.Histogram(x=res["ST"][:200000], nbinsx=80, marker_color=INK, opacity=0.8))
    fig.add_vline(K, line_dash="dot", line_color=RED, annotation_text="K")
    fig.add_vline(res["mean_ST"], line_color=GREEN, annotation_text="E[S_T]")
    show_fig(style_fig(fig, "Distribution of the final price S_T", "S_T", "Number of paths"))
with t3:
    conv = res["convergence"]
    fig = go.Figure([line(conv["Simulations"], conv["Price"], "MC estimate", INK)])
    fig.add_hline(bench["black_hagan"], line_dash="dash", line_color=AMBER, annotation_text="Hagan/Black")
    fig.add_hrect(res["ci95"][0], res["ci95"][1], fillcolor=GREEN, opacity=0.12, line_width=0)
    fig.update_xaxes(type="log")
    show_fig(style_fig(fig, "Convergence of the price vs number of simulations", "Simulations (log scale)", "Price"))
with t4:
    if st.toggle("Run the three schemes with the same inputs", value=True, key="mc_cmp"):
        cmp_tab = run_compare(S0, K, T, r, alpha, nu, rho, beta, int(n_sims), int(n_steps), int(seed), phi)
        table(cmp_tab, "{:,.4f}")
        downloads(cmp_tab, "sabr_mc_schemes", "mc_cmp_dl")
    else:
        cmp_tab = pd.DataFrame()
with t5:
    summary = pd.DataFrame([{"Scheme": mc.SCHEMES[scheme], "Simulations": int(n_sims), "Steps": int(n_steps),
                             "Seed": int(seed), "E[S_T]": res["mean_ST"], "S0 e^{rT}": res["forward"],
                             "Price": res["price"], "SE": res["se"], "CI low": res["ci95"][0],
                             "CI high": res["ci95"][1], "Hagan vol": bench["hagan_vol"],
                             "Black (Hagan)": bench["black_hagan"], "BS (sigma=alpha)": bench["bs_alpha"]}])
    table(summary)
    downloads(summary, "sabr_mc_summary", "mc_sum_dl",
              sheets={"summary": summary, "convergence": res["convergence"],
                      "S_T_sample": pd.DataFrame({"S_T": res["ST"][:100000]})})


def build_pdf():
    f, ax = plt.subplots(2, 2, figsize=(11.69, 8.27))
    for i in range(res["S_paths"].shape[0]):
        ax[0, 0].plot(tgrid, res["S_paths"][i], lw=0.7)
        ax[0, 1].plot(tgrid, res["v_paths"][i], lw=0.7)
    ax[0, 0].set_title("Sample paths of S"); ax[0, 1].set_title("Sample paths of sigma")
    ax[1, 0].hist(res["ST"], bins=80, color="steelblue"); ax[1, 0].axvline(K, color="r", ls=":")
    ax[1, 0].set_title("Histogram of S_T")
    conv = res["convergence"]
    ax[1, 1].semilogx(conv["Simulations"], conv["Price"], "b-")
    ax[1, 1].axhline(bench["black_hagan"], color="orange", ls="--", label="Hagan/Black")
    ax[1, 1].set_title("Convergence"); ax[1, 1].legend()
    for a in ax.ravel():
        a.grid(alpha=0.3)
    f.tight_layout()
    blocks = [("title", "SABR Monte Carlo"),
              ("text", f"S0 = {S0:.4f} ({src}); K = {K:g}; T = {T:g}; r = {r:.4f}; {kind}; alpha = {alpha:.4f}; "
                       f"nu = {nu:.4f}; rho = {rho:.4f}; beta = {beta:.2f}"),
              ("text", f"Scheme: {mc.SCHEMES[scheme]}; simulations = {int(n_sims)}; steps = {int(n_steps)}; seed = {int(seed)}"),
              ("text", "Update equations: " + " ; ".join(EQS[scheme])),
              ("table", ("Results", summary.T.reset_index().rename(columns={"index": "Item", 0: "Value"}))),
              ("figure", f)]
    if not cmp_tab.empty:
        blocks.append(("table", ("Scheme comparison", cmp_tab)))
    return make_pdf("Option Toolkit - SABR Monte Carlo report", blocks)


st.divider()
pdf_report(build_pdf, "sabr_monte_carlo_report.pdf", "mc_pdf")

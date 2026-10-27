# Module 2 - Arbitrage Calculator (put-call parity, synthetic options with stock + bonds)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import parity
from core.black_scholes import days_to_years
from core.reports import make_pdf, line_figure
from ui.common import (AMBER, GREEN, INK, RED, VIOLET, action_color, downloads, line, money,
                       page_header, pdf_report, show_fig, spot_input, style_fig, table)

page_header("Arbitrage calculator", "Put-call parity check with synthetic options built from the stock "
            "and zero-coupon bonds, and the riskless strategy step by step.", "arbitrage")

with st.expander("Inputs", expanded=True, icon=":material/tune:"):
    S, ticker, src = spot_input("arb", default=100.0, ticker="AAPL")
    c1, c2 = st.columns(2)
    K = c1.number_input("Strike K ($)", value=100.0, min_value=0.01, step=1.0, key="arb_K")
    T_mode = c2.radio("Maturity in", ["years", "days"], horizontal=True, key="arb_Tmode")
    if T_mode == "years":
        T = c1.number_input("T (years)", value=1.0, min_value=0.0001, step=0.25, format="%.4f", key="arb_T")
    else:
        T = float(days_to_years(c1.number_input("T (calendar days, /365)", value=365, min_value=1, step=1, key="arb_Td")))
    r = c2.number_input("r (decimal, e.g. 0.05)", value=0.05, step=0.005, format="%.4f", key="arb_r")
    q = c1.number_input("q dividend yield (decimal)", value=0.0, step=0.005, format="%.4f", key="arb_q")
    C = c2.number_input("Market call price C ($)", value=12.00, min_value=0.0, step=0.1, format="%.4f", key="arb_C")
    P = c1.number_input("Market put price P ($)", value=5.5735, min_value=0.0, step=0.1, format="%.4f", key="arb_P")
    tol = c2.number_input("Tolerance / cost per share ($)", value=0.01, min_value=0.0, step=0.01, format="%.4f",
                          key="arb_tol", help="An arbitrage is reported only if the mispricing exceeds this amount.")
    contracts = c1.number_input("Number of contracts", value=1, min_value=1, step=1, key="arb_n")
    mult = c2.number_input("Multiplier (shares / contract)", value=100, min_value=1, step=1, key="arb_m")

try:
    res = parity.parity_analysis(S, K, T, r, C, P, q=q, tol=tol)
    steps, pay_tab, summ = parity.build_strategy(res, contracts, mult)
except Exception as e:
    st.error(f"Please check the inputs: {e}")
    st.stop()

st.subheader("Put-call parity")
st.latex(r"C - P = S e^{-qT} - K e^{-rT}")
c1, c2 = st.columns(2)
c1.metric("Left side  C − P", f"{res['lhs']:.4f}")
c2.metric("Right side  S e^{−qT} − K e^{−rT}", f"{res['rhs']:.4f}")
c1.metric("Gap (per share)", f"{res['gap']:+.4f}", help="C − synthetic call = synthetic put − P")
c2.metric("PV(K) = K e^{−rT}", f"{res['pv_K']:.4f}")

with st.expander("Synthetic options (composition)", expanded=False, icon=":material/construction:"):
    st.latex(r"C_{syn} = P + S e^{-qT} - K e^{-rT} = " + f"{res['synthetic_call']:.4f}")
    st.markdown(f"Long 1 put + long {res['stock_leg']:.4f} share (e^{{−qT}}) + borrow PV(K) = {res['pv_K']:.4f} "
                "(sell a zero-coupon bond paying K at T).")
    st.latex(r"P_{syn} = C - S e^{-qT} + K e^{-rT} = " + f"{res['synthetic_put']:.4f}")
    st.markdown(f"Long 1 call + short {res['stock_leg']:.4f} share + lend PV(K) (buy the zero-coupon bond).")
    st.caption(f"Market call {C:.4f} vs synthetic {res['synthetic_call']:.4f}; market put {P:.4f} vs synthetic "
               f"{res['synthetic_put']:.4f}.")

st.subheader("No-arbitrage bounds")
b = res["bounds"].copy()
b["Status"] = np.where(b["OK"], "OK", "VIOLATED")
st.dataframe(b[["Bound", "Condition", "Price", "Limit", "Status"]].style
             .format({"Price": "{:.4f}", "Limit": "{:.4f}"})
             .map(lambda v: f"color: {GREEN if v == 'OK' else RED}; font-weight: 600", subset=["Status"]),
             width="stretch", hide_index=True)
for _, row in b[~b["OK"]].iterrows():
    st.warning(f"{row['Bound']} violated. {row['If violated']}")

st.subheader("Verdict")
if res["decision"] == "none":
    st.info(f"No arbitrage: |gap| = {abs(res['gap']):.4f} ≤ tolerance {tol:.4f}. Prices are consistent with parity.")
else:
    rich = "call is OVERPRICED (put cheap)" if res["decision"] == "call_overpriced" else "call is UNDERPRICED (put rich)"
    st.success(f"Arbitrage opportunity: the {rich} by {abs(res['gap']):.4f} per share.")
    st.markdown(f"- **Call view:** {summ['view_call']}\n- **Put view:** {summ['view_put']}\n\n"
                "Both views are the same four trades.")
    c1, c2 = st.columns(2)
    c1.metric("Locked-in profit today", money(summ["profit_today"]),
              help=f"{summ['units']:g} options × |gap| − costs")
    c2.metric("Value at maturity", money(summ["profit_fv"]), help="profit × e^{rT}")

    st.markdown("**Step-by-step strategy (t = 0)**")
    st.dataframe(steps.style.format({"Quantity": "{:,.4f}", "Cash flow at t=0": "{:+,.2f}"})
                 .map(action_color, subset=["Action"]), width="stretch", hide_index=True)
    st.caption(f"Net cash received today: {money(summ['gross_today'])}"
               + (f", minus costs {money(summ['cost'])}" if summ["cost"] else "")
               + ". Positive cash flow = received, negative = paid.")
    st.markdown("**Payoff at maturity: the position closes at zero**")
    table(pay_tab, "{:+,.2f}")
    downloads(steps, "arbitrage_strategy", "arb_steps", sheets={"strategy": steps, "payoff_at_T": pay_tab,
                                                                  "bounds": res["bounds"]})

    grid = np.linspace(0.5 * K, 1.5 * K, 101)
    legs = parity.leg_payoffs(res, grid, contracts, mult)
    fig = go.Figure([line(grid, legs["Call leg"], "Call leg", RED), line(grid, legs["Put leg"], "Put leg", GREEN),
                     line(grid, legs["Stock leg"], "Stock leg", AMBER), line(grid, legs["Bond leg"], "Bond leg", VIOLET),
                     line(grid, legs["Total (at T)"], "Total at T (= 0)", INK, width=4)])
    fig.add_vline(K, line_dash="dot", line_color="grey")
    show_fig(style_fig(fig, "Payoff at T of each leg (profit was locked in at t = 0)", "S_T ($)", "$"))


def build_pdf():
    blocks = [("title", "Arbitrage calculator (put-call parity)"),
              ("text", f"S = {S:.4f} ({src}), K = {K:.4f}, T = {T:.4f} y, r = {r:.4f}, q = {q:.4f}"),
              ("text", f"Market C = {C:.4f}, P = {P:.4f}, tolerance = {tol:.4f}, contracts = {contracts}, multiplier = {mult}"),
              ("text", f"C - P = {res['lhs']:.4f};  S e^-qT - K e^-rT = {res['rhs']:.4f};  gap = {res['gap']:+.4f}"),
              ("text", f"Synthetic call = {res['synthetic_call']:.4f};  synthetic put = {res['synthetic_put']:.4f};  PV(K) = {res['pv_K']:.4f}"),
              ("table", ("No-arbitrage bounds", res["bounds"].drop(columns=["If violated"])))]
    if res["decision"] == "none":
        blocks.append(("text", "Verdict: no arbitrage."))
    else:
        blocks += [("text", f"Verdict: {res['decision']}. {summ['view_call']}"),
                   ("text", f"Locked-in profit today = {summ['profit_today']:,.2f}; at maturity = {summ['profit_fv']:,.2f}"),
                   ("table", ("Strategy", steps)), ("table", ("Payoff at T", pay_tab))]
        g = np.linspace(0.5 * K, 1.5 * K, 101)
        lg = parity.leg_payoffs(res, g, contracts, mult)
        blocks.append(("figure", line_figure(g, {c: (lg[c], s) for c, s in
                                                 zip(["Call leg", "Put leg", "Stock leg", "Bond leg", "Total (at T)"],
                                                     ["r-", "g-", "y-", "m-", "k-"])},
                                             "Payoff at T of each leg", "S_T", "$", vline=K)))
    return make_pdf("Option Toolkit - Arbitrage report", blocks)


pdf_report(build_pdf, "arbitrage_report.pdf", "arb_pdf")

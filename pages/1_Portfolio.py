# Module 1 - Portfolio Evaluation
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import portfolio as pf
from core.reports import make_pdf, line_figure
from ui.common import (GREEN, INK, RED, AMBER, downloads, line, money, page_header, pdf_report,
                       show_fig, spot_input, style_fig, table)

page_header("Portfolio evaluation", "Market value, intrinsic value and Greeks of a portfolio of European "
            "options on one stock (Black-Scholes-Merton).", "portfolio")

with st.expander("Market inputs", expanded=True, icon=":material/tune:"):
    S, ticker, src = spot_input("pf", default=100.0, ticker="AAPL")
    c1, c2 = st.columns(2)
    r = c1.number_input("r (decimal, e.g. 0.05)", value=0.05, step=0.005, format="%.4f", key="pf_r")
    q = c2.number_input("q dividend yield (decimal)", value=0.0, step=0.005, format="%.4f", key="pf_q")

st.subheader("Positions")
st.caption("One row per option position. Long = bought, short = sold. T in years. Leave *Market price* "
           "empty to value the option with Black-Scholes. Add or delete rows with the table toolbar.")
positions = st.data_editor(
    pf.DEFAULT_POSITIONS, num_rows="dynamic", width="stretch", hide_index=True, key="pf_editor",
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Type": st.column_config.SelectboxColumn("Type", options=["call", "put"], required=True),
        "Position": st.column_config.SelectboxColumn("Position", options=["long", "short"], required=True),
        "K": st.column_config.NumberColumn("K ($)", min_value=0.01, format="%.2f", required=True),
        "T (years)": st.column_config.NumberColumn("T (years)", min_value=0.0001, format="%.4f", required=True),
        "Contracts": st.column_config.NumberColumn("Contracts", min_value=0, step=1, required=True),
        "Multiplier": st.column_config.NumberColumn("Multiplier", min_value=1, step=1, default=100,
                                                    help="Shares per contract (100 for US equity options)"),
        "Sigma": st.column_config.NumberColumn("σ (decimal)", min_value=0.0001, format="%.4f", required=True),
        "Market price": st.column_config.NumberColumn("Market price / option ($)", min_value=0.0, format="%.4f"),
    })

try:
    res, tot = pf.value_portfolio(positions, S, r, q)
except Exception as e:
    st.error(f"Please check the positions table: {e}")
    st.stop()

if positions["Ticker"].dropna().astype(str).str.upper().nunique() > 1:
    st.info("Several tickers are listed: all positions are valued with the single spot S above.")

c1, c2 = st.columns(2)
c1.metric("Portfolio market value", money(tot["Market value"]))
c2.metric("Intrinsic value", money(tot["Intrinsic value"]))
c1.metric("Time value", money(tot["Time value"]))
c2.metric("Net delta (shares)", f"{tot['Net delta (shares)']:,.2f}")
c1.metric("Net gamma (shares per $1)", f"{tot['Net gamma (shares)']:,.4f}")
c2.metric("Spot used", f"${S:,.2f}", help=src)

t1, t2, t3, t4 = st.tabs(["Per position", "Payoff at maturity", "Value today vs S", "Scenarios"])
with t1:
    show_cols = ["Type", "Position", "K", "T", "Contracts", "BS price", "Price used", "Intrinsic / option",
                 "Time value / option", "Delta / option", "Gamma / option", "Market value",
                 "Intrinsic value", "Delta (shares)"]
    table(res[show_cols])
    downloads(res, "portfolio_positions", "pf_pos")

lo, hi = 0.5 * S, 1.5 * S
grid = np.linspace(lo, hi, 201)
pay = pf.payoff_at_maturity(positions, grid)
val = pf.value_curve(positions, grid, r, q)
with t2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=np.where(pay >= 0, pay, np.nan), name="Payoff ≥ 0", mode="lines",
                             line=dict(color=GREEN, width=3)))
    fig.add_trace(go.Scatter(x=grid, y=np.where(pay < 0, pay, np.nan), name="Payoff < 0", mode="lines",
                             line=dict(color=RED, width=3)))
    fig.add_trace(line(grid, pay - tot["Market value"], "Net P&L (payoff − value today)", INK, "dot"))
    fig.add_vline(S, line_dash="dot", line_color="grey")
    fig.add_hline(0, line_color="black", line_width=0.6)
    show_fig(style_fig(fig, "Portfolio payoff at maturity", "S_T ($)", "$"))
with t3:
    fig = go.Figure([line(grid, val, "Black-Scholes value today", INK),
                     line(grid, pay, "Payoff at maturity", AMBER, "dash")])
    fig.add_vline(S, line_dash="dot", line_color="grey")
    show_fig(style_fig(fig, "Portfolio value vs spot", "S ($)", "$"))
with t4:
    default_sc = ", ".join(f"{x:g}" for x in np.round(S * np.array([0.8, 0.9, 1.0, 1.05, 1.1, 1.2]), 2))
    txt = st.text_input("S_T scenarios (comma-separated)", value=default_sc, key="pf_sc")
    try:
        sc = [float(x) for x in txt.replace(";", ",").split(",") if x.strip()]
        if not sc or min(sc) < 0:
            raise ValueError
        sc_tab = pf.scenario_table(positions, sc, premium_paid=tot["Market value"])
        table(sc_tab, "{:,.2f}")
        downloads(sc_tab, "portfolio_scenarios", "pf_sc_dl")
    except ValueError:
        st.error("Scenarios must be non-negative numbers separated by commas, e.g. 80, 100, 120.")
        sc_tab = pd.DataFrame()


def build_pdf():
    blocks = [("title", "Portfolio evaluation"),
              ("text", f"Spot S = {S:.4f} ({src}); r = {r:.4f}; q = {q:.4f}"),
              ("text", "Totals: " + "; ".join(f"{k} = {v:,.4f}" for k, v in tot.items())),
              ("table", ("Positions (input)", positions)),
              ("table", ("Valuation per position", res[show_cols])),
              ("figure", line_figure(grid, {"Payoff at maturity": (pay, "g-"),
                                            "BS value today": (val, "b--")},
                                     "Portfolio payoff and value", "S", "$", vline=S))]
    if not sc_tab.empty:
        blocks.append(("table", ("Scenarios", sc_tab)))
    return make_pdf("Option Toolkit - Portfolio evaluation report", blocks)


pdf_report(build_pdf, "portfolio_report.pdf", "pf_pdf")

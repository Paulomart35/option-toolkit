import streamlit as st

st.title("Option Toolkit")
st.markdown("Five calculators for European equity options, built on the methods of the "
            "*Forwards, Futures & Options* course (Prof. Mauricio Vergara, Fall 2026). "
            "Every page has a **Help** button with the theory, the assumptions and the formulas.")

MODULES = [
    ("pages/1_Portfolio.py", "Portfolio evaluation",
     "Market value, intrinsic and time value, delta and gamma of a portfolio of calls and puts, with payoff scenarios."),
    ("pages/2_Arbitrage.py", "Arbitrage calculator",
     "Checks put-call parity, builds synthetic options with stock and bonds and writes the riskless trade step by step."),
    ("pages/3_Hedging.py", "Delta & delta-gamma hedging",
     "Hedge for written options, then daily rebalancing on historical closes or on simulated spots (up to 5 days)."),
    ("pages/4_SABR.py", "SABR calculator",
     "Hagan implied volatility, calibration on uploaded option data, smiles, pricing and mispriced-option strategies."),
    ("pages/5_SABR_Monte_Carlo.py", "SABR Monte Carlo",
     "Simulates the SABR equations (Euler, full truncation or log transform) to price an option."),
]
nav_pages = st.session_state.get("_nav_pages", {})      # st.Page objects registered by app.py
for path, title, text in MODULES:
    with st.container(border=True):
        if path in nav_pages:
            st.page_link(nav_pages[path], label=title)
        else:
            st.markdown(f"**{title}**")
        st.caption(text)

st.caption("Spot prices can be fetched from Yahoo Finance or typed by hand. Tables export to CSV/Excel "
           "and each module produces a printable PDF report.")

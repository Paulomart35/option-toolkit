# Option Toolkit - Final exam app (Forwards, Futures, Options - Prof. Mauricio Vergara, Pace, Fall 2026)
# Run: streamlit run app.py
import streamlit as st

st.set_page_config(page_title="Option Toolkit", page_icon=":material/candlestick_chart:",
                   layout="centered", initial_sidebar_state="auto")

SPEC = [
    ("pages/0_Home.py", "Home", ":material/home:"),
    ("pages/1_Portfolio.py", "Portfolio evaluation", ":material/account_balance_wallet:"),
    ("pages/2_Arbitrage.py", "Arbitrage calculator", ":material/balance:"),
    ("pages/3_Hedging.py", "Delta & delta-gamma hedging", ":material/shield:"),
    ("pages/4_SABR.py", "SABR calculator", ":material/ssid_chart:"),
    ("pages/5_SABR_Monte_Carlo.py", "SABR Monte Carlo", ":material/casino:"),
]
pages = {path: st.Page(path, title=title, icon=icon, default=(i == 0)) for i, (path, title, icon) in enumerate(SPEC)}
st.session_state["_nav_pages"] = pages          # lets the home page link to the modules
nav = st.navigation(list(pages.values()), position="sidebar")
st.sidebar.caption("Option Toolkit by Paul Martin, FFO Fall 2026")
nav.run()

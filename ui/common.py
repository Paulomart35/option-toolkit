# Shared UI pieces: spot input (live / manual), download buttons, PDF button, chart styling.
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from core.market_data import MarketDataError, fetch_closes, fetch_spot
from core.reports import to_csv_bytes, to_excel_bytes

# Palette: green = profit / BUY, red = loss / SELL
INK = "#1F3A5F"
GREEN = "#1B7F4C"
RED = "#B03A2E"
AMBER = "#C98B12"
VIOLET = "#6B4E9B"
SLATE = "#5B6B7B"
SERIES = [INK, AMBER, VIOLET, GREEN, RED, SLATE, "#2E86AB", "#8C5E3C"]


@st.cache_data(ttl=300, show_spinner=False)
def cached_spot(ticker):
    return fetch_spot(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_closes(ticker, n):
    return fetch_closes(ticker, n)


def _fetch_cb(key):
    tk = st.session_state.get(f"{key}_ticker", "")
    try:
        price, src = cached_spot(tk)
        st.session_state[f"{key}_S"] = round(price, 4)
        st.session_state[f"{key}_src"] = src
        st.session_state[f"{key}_err"] = None
    except MarketDataError as e:
        st.session_state[f"{key}_err"] = str(e)


def _manual_cb(key):
    st.session_state[f"{key}_src"] = "Manual, " + datetime.now().strftime("%Y-%m-%d %H:%M")


def spot_input(key, default=100.0, ticker="AAPL", label="Spot price S ($)"):
    """Ticker + 'Fetch live price' + manual number input (pre-filled by the fetch, always editable).
    Returns (S, ticker, source text)."""
    st.session_state.setdefault(f"{key}_S", float(default))
    st.session_state.setdefault(f"{key}_src", "Manual (default value)")
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    with c1:
        tk = st.text_input("Ticker (Yahoo Finance)", value=ticker, key=f"{key}_ticker")
    with c2:
        st.button("Fetch live price", key=f"{key}_fetch", on_click=_fetch_cb, args=(key,),
                  icon=":material/cloud_download:", width="stretch")
    S = st.number_input(label, min_value=0.0001, step=1.0, format="%.4f", key=f"{key}_S",
                        on_change=_manual_cb, args=(key,))
    err = st.session_state.get(f"{key}_err")
    if err:
        st.warning(err)
    st.caption(f"Source: {st.session_state[f'{key}_src']}")
    return float(S), tk, st.session_state[f"{key}_src"]


def downloads(df, name, key, sheets=None):
    """CSV + Excel download buttons for a results table."""
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download CSV", to_csv_bytes(df), f"{name}.csv", "text/csv",
                           key=f"{key}_csv", icon=":material/download:", width="stretch")
    with c2:
        st.download_button("Download Excel", to_excel_bytes(sheets or {name: df}), f"{name}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key=f"{key}_xlsx", icon=":material/table:", width="stretch")


def pdf_report(builder, file_name, key, label="Download PDF report"):
    """Two-step PDF: build on click (costly), then offer the download."""
    if st.button("Build PDF report", key=f"{key}_build", icon=":material/picture_as_pdf:"):
        try:
            with st.spinner("Building the PDF..."):
                st.session_state[f"{key}_pdf"] = builder()
        except Exception as e:  # never show a traceback
            st.error(f"Could not build the PDF report: {e}")
    if st.session_state.get(f"{key}_pdf"):
        st.download_button(label, st.session_state[f"{key}_pdf"], file_name, "application/pdf",
                           key=f"{key}_dl", icon=":material/download:", type="primary")


def style_fig(fig, title=None, x=None, y=None, height=380):
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
                      xaxis_title=x, yaxis_title=y, template="plotly_white",
                      font=dict(color="#26313D"), hovermode="x unified")
    return fig


def show_fig(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def table(df, fmt="{:,.4f}", height=None):
    kw = {"width": "stretch", "hide_index": True}
    if height:
        kw["height"] = height
    num = df.select_dtypes("number").columns
    st.dataframe(df.style.format({c: fmt for c in num}), **kw)


def money(x):
    return f"${x:,.2f}" if x >= 0 else f"−${-x:,.2f}"


def action_color(text):
    t = str(text).upper()
    if t.startswith(("BUY", "LEND")):
        return f"color: {GREEN}; font-weight: 600"
    if t.startswith(("SELL", "BORROW")):
        return f"color: {RED}; font-weight: 600"
    return ""


def page_header(title, subtitle, help_key):
    from help.render import help_button
    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    with c1:
        st.title(title)
    with c2:
        help_button(help_key)
    st.caption(subtitle)


def line(x, y, name, color, dash=None, width=2):
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, dash=dash, width=width))

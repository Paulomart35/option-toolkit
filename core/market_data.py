# =============================================================
# Online market data (Yahoo Finance via yfinance) with a hard timeout.
# Never raises anything but MarketDataError, so pages can fall back to manual input.
# (Caching with st.cache_data is done in the UI layer: core/ stays Streamlit-free.)
# =============================================================
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from datetime import datetime

TIMEOUT = 8  # seconds

_pool = ThreadPoolExecutor(max_workers=2)


class MarketDataError(Exception):
    pass


def _history(ticker, period):
    import yfinance as yf
    return yf.Ticker(ticker).history(period=period, timeout=TIMEOUT, auto_adjust=False)


def _run(fn, *args):
    try:
        return _pool.submit(fn, *args).result(timeout=TIMEOUT + 2)
    except FutTimeout:
        raise MarketDataError("Yahoo Finance did not answer in time (timeout). Enter the value manually.")
    except MarketDataError:
        raise
    except Exception as e:  # network, parsing, yfinance internals
        raise MarketDataError(f"Could not fetch data from Yahoo Finance ({type(e).__name__}). "
                              "Enter the value manually.")


def fetch_spot(ticker):
    """Last close/last price. Returns (price, source text)."""
    t = (ticker or "").strip().upper()
    if not t:
        raise MarketDataError("Enter a ticker first (e.g. AAPL).")
    hist = _run(_history, t, "5d")
    if hist is None or hist.empty or "Close" not in hist:
        raise MarketDataError(f"No price found for '{t}'. Check the ticker or enter the price manually.")
    close = hist["Close"].dropna()
    if close.empty:
        raise MarketDataError(f"No price found for '{t}'.")
    try:
        day = close.index[-1].strftime("%Y-%m-%d")
    except AttributeError:
        day = "latest session"
    fetched = datetime.now().strftime("%Y-%m-%d %H:%M")
    return float(close.iloc[-1]), f"Yahoo Finance ({t}), last price of {day}, fetched {fetched}"


def fetch_closes(ticker, n=10):
    """Last n daily closes (n <= 30). Returns (list of dates as str, list of closes)."""
    t = (ticker or "").strip().upper()
    if not t:
        raise MarketDataError("Enter a ticker first (e.g. AAPL).")
    n = int(max(2, min(n, 30)))
    hist = _run(_history, t, "3mo")
    if hist is None or hist.empty or "Close" not in hist:
        raise MarketDataError(f"No history found for '{t}'.")
    close = hist["Close"].dropna().iloc[-n:]
    return [d.strftime("%Y-%m-%d") for d in close.index], [float(x) for x in close.values]

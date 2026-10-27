# Option Toolkit

Final exam app for *Forwards, Futures & Options* (Prof. Mauricio Vergara, Pace University, Fall 2026), by
Paul Martin. A mobile-friendly Streamlit web app with four calculators and a SABR Monte Carlo module, built
on the course methods (Black-Scholes-Merton with q, HW7 hedging conventions, Hagan 2002 SABR with
Differential Evolution as in HW8, the professor's `std_MC` Monte Carlo loop).

## How to open the app

**Online, nothing to install:** open **APP_LINK_HERE** on a computer or a phone.
No account or sign-in is needed.

- If the page says *"This app has gone to sleep"*, click **"Yes, get this app back up!"** and wait about
  30 seconds. Free Streamlit apps sleep after a few days without visitors.
- Use the menu on the left to switch modules (on a phone, tap **>** at the top left).
- Every module opens with the course examples already filled in, so results show immediately.
  The **Help** button at the top of each module explains the theory, assumptions and formulas.
- To add it to a phone's home screen: open the link, then use **Share → Add to Home Screen** (iPhone)
  or **⋮ → Add to Home screen** (Android).

**Suggested tour (about 5 minutes)**

1. *Arbitrage calculator*: the default prices (call 12.00, put 5.5735) show a parity violation, the
   step-by-step strategy and the payoff table that closes at zero.
2. *Delta & delta-gamma hedging*: the static hedge reproduces HW7; the *Forecast simulator* tab runs the
   5-day Spot Price Simulator and compares unhedged, delta and delta-gamma P&L.
3. *SABR calculator*: the *Calibration* tab calibrates on the bundled sample data. To use the course file,
   choose *Upload CSV / Excel*, select `Final_Database_Options_.xlsx`, then set S, r and the valuation
   date as in HW8 (TSLA 408.92, r = 0.05985, 17 Nov 2025).
4. *SABR Monte Carlo*: choose the scheme and the number of simulations; the price is compared with the
   Hagan/Black price.

Live prices come from Yahoo Finance. If Yahoo does not answer, a message appears and the spot price can be
typed by hand; every page keeps working.

**On your own computer instead (Python 3.11 or newer):** download the repository (green **Code** button →
**Download ZIP**), unzip it, open a terminal in that folder, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in the browser at http://localhost:8501.

## Modules

| Page | What it does |
|---|---|
| Portfolio evaluation | Editable table of call/put positions (long/short, contracts, multiplier). Black-Scholes price or observed market price, intrinsic and time value, delta and gamma per position and in total. Payoff at maturity, value curve vs S, user scenarios. |
| Arbitrage calculator | Put-call parity C − P = S e^{−qT} − K e^{−rT}, synthetic call/put with stock + zero-coupon bonds, no-arbitrage bounds. When mispriced: numbered step-by-step trades, payoff table at T (closes at zero), locked-in profit today and at T, leg payoff chart. |
| Delta & delta-gamma hedging | HW7 static hedge (report, scenario P&L grid, 4 charts). Daily rebalancing engine (Greeks, trades, bank account at r, daily and cumulative P&L, plain-English instructions) on historical closes (Yahoo, ≤ 30 days, or typed) and on a GBM Spot Price Simulator (≤ 5 days, 1–1000 paths) with P&L distribution for unhedged / delta / delta-gamma. |
| SABR calculator | Hagan implied vol calculator; calibration per maturity with the course DE optimiser on the course Excel file or a simple CSV (sample included); market vs SABR smiles; smile grid across S and F = S e^{rT}; pricing table (K, F, T, σ_Hagan, SABR price, flat-vol BS price); mispriced-option relative-value strategy + parity check; 3D surface. |
| SABR Monte Carlo | Euler-Maruyama, full truncation or log-transform simulation of SABR (user-chosen number of simulations and steps); E[S_T] vs S0 e^{rT}, discounted average payoff with standard error and 95% CI, comparison with Hagan/Black and flat BS, paths, histogram, convergence, three-scheme table. |

Every page has a **Help** button (theory, assumptions, SymPy-rendered formulas with symbol legends, how to
use, limitations), CSV/Excel downloads for each results table and a printable **PDF report**.
Spot prices can be fetched from Yahoo Finance (8 s timeout, cached 5 min) or typed manually.

## Run locally

Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Tests (core math against `tests/reference_values.md`, plus every page loaded with `AppTest`):

```bash
pytest -q
```

## Deploy on Streamlit Community Cloud (phone access)

1. Create a **public** GitHub repository and push this folder to its `main` branch
   (`git init`, `git add .`, `git commit -m "Option Toolkit"`, `git remote add origin <url>`, `git push -u origin main`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repository, branch `main`,
   main file `app.py` (Advanced settings: Python 3.11 or 3.12) → **Deploy**.
3. Open the app URL on the phone → browser menu → **Add to Home screen**.

## Project layout

```
app.py                 navigation (st.navigation) and page registry
core/                  pure math (numpy / scipy, no Streamlit), fully tested
  black_scholes.py     price, d1/d2, Greeks (phi = ±1, q), Black-76, implied vol, days→years (/365)
  portfolio.py         Module 1 valuation and payoffs
  parity.py            Module 2 parity, bounds, synthetics, strategy builder
  hedging.py           Module 3 HW7 hedge, scenarios, daily rebalancing engine, GBM simulator
  sabr.py              Module 4 Hagan vol, DE (pablormier), calibration, data cleaning, pricing, mispricing
  sabr_mc.py           Module 5 Monte Carlo schemes
  market_data.py       Yahoo Finance with timeout
  reports.py           CSV / Excel / PDF (matplotlib PdfPages)
help/                  SymPy help content and the Help dialog
ui/common.py           shared widgets (spot input, downloads, PDF button, chart style)
pages/                 0_Home + the five modules
sample_data/           sabr_sample.csv and the script that generates it
tests/                 test_core.py, test_pages.py (AppTest), test_interactions.py
DECISIONS.md           choices made where the brief was ambiguous
```

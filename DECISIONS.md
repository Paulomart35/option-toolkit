# Decisions taken in autonomous mode

Each entry: what was ambiguous or blocked → what was chosen and why.

## Environment
- **venv location.** `python -m venv .venv` inside the project failed on this machine: the project path is
  ~200 characters long and Windows long-path support is off, so pip could not unpack wheels. The venv was
  created at `C:\Users\Xefi\.venvs\optkit` instead. On another machine `.venv` in the project works normally.
- **No git commits.** Git is not installed on this machine, so the "commit after each module" step was skipped.
  `.gitignore` is ready; run `git init && git add . && git commit -m "Option Toolkit"` once git is available.
- **Streamlit 1.66.** `use_container_width=True` is replaced by its successor `width="stretch"` (same
  full-width behaviour, no deprecation warnings).

## Architecture
- **Caching outside `core/`.** CLAUDE.md asks for `st.cache_data` in `core/market_data.py` but also for a
  Streamlit-free `core/`. `core/market_data.py` is pure (yfinance + 8 s timeout in a worker thread); the
  `st.cache_data(ttl=300)` wrappers live in `ui/common.py`.
- **Shared UI helpers** are in `ui/common.py` (spot widget, download buttons, PDF button, chart style).
- **Home page links.** `st.page_link` needs the `st.Page` objects; `app.py` stores them in session state so
  `pages/0_Home.py` also runs standalone (AppTest) with plain titles.

## Module 1 (Portfolio)
- One underlying per portfolio: the ticker column is informative and every position uses the shared spot S
  (a note is shown when several tickers are typed). Payoff/value charts need a single S axis anyway.
- Scenario P&L = payoff at maturity − portfolio value today (no discounting), as in HW "Module B".

## Module 2 (Arbitrage)
- The tolerance is treated as a **cost per share**: arbitrage is reported only if |gap| > tolerance and the
  locked-in profit is (|gap| − tolerance) × contracts × multiplier. Default 0.01 so that rounding of typed
  prices does not trigger "arbitrage" of a fraction of a cent.
- With q > 0 the stock leg is e^{−qT} shares; dividends are assumed reinvested so it is 1 share at T.
- Default market prices = T5 example (call 12.00, put 5.5735): the page opens on an opportunity.

## Module 3 (Hedging)
- Engine: premium received at t = 0 goes to the bank account, trades are paid from it, cash × e^{r/252}
  each day, P&L of day t computed with the holdings of day t−1 before rebalancing (self-financing).
- τ_t = T − t/252 for both the written and the hedging option; at τ ≤ 0 options are worth their payoff.
- The hedging-option inputs are always visible so the forecast tab can compare unhedged / delta /
  delta-gamma even when the selected strategy is "delta". If the hedging option has Γ ≈ 0 the delta-gamma
  comparison is dropped (and the delta-gamma strategy shows the HW7 error message).
- Historical mode: day 0 is the first close in the list (not the spot widget value).

## Module 4 (SABR)
- Default DE settings: population 15, 60 iterations (fast on Streamlit Cloud); HW8 used 50 / 1000.
  Both are adjustable. Bounds, mutation 0.8, crossover 0.7 and seed 42 as in HW8; `de()` copied unchanged.
- With β fixed, DE searches only (α, ρ, ν) with the HW8 bounds for those three.
- Simple file format: columns are matched case-insensitively (`strike`/`K`, `T`/`maturity` or
  `expiry`/`expirationDate`, `impliedVolatility`/`iv` or `price`); prices are inverted with Brent's method.
  The HW8 moneyness / IV / T filters are applied; the liquidity filter only exists for the course format.
- Sample data (`sample_data/make_sample.py`): S0 = 100, r = 5%, β = 1, T = 0.25 / 0.5 / 1 with
  (α, ρ, ν) = (0.32, −0.55, 1.40), (0.30, −0.50, 1.00), (0.28, −0.45, 0.70), 13 strikes in 0.85–1.15 F,
  Gaussian noise of 0.2 vol points (seed 7).
- Smile grid, pricing and mispricing tabs use the parameters of one selected calibrated maturity
  (or the manual Hagan-calculator parameters if there is no calibration). The "BS price" column uses the
  flat ATM Hagan vol σ(F, F) of that maturity.
- Mispricing strategy: delta from Black-Scholes with the Hagan vol; the bond leg finances the rest so the
  net cash at t = 0 is zero; expected profit = (|market − SABR| − tolerance) × units, labelled as
  model-based relative value, not riskless. Parity check runs when the other option's price is typed.

## Module 5 (SABR Monte Carlo)
- Random numbers: `numpy.random.default_rng(seed)`, Z1 then Z2 drawn each step (same as the reference script,
  so tests T7/T8 reproduce its values).
- Full truncation uses σ⁺ in the price drift/diffusion **and** in the vol diffusion (standard Lord et al.
  definition). The reference script kept σ (not σ⁺) in the vol diffusion; both give ≈ 10.72 for T8.
- β < 1: the forward F (driftless under the T-forward measure, F0 = S0 e^{rT}) is simulated with F⁺ and
  absorbed at 0 for every scheme (needed for F^β); the vol scheme still differs. S_T = F_T.
- "BS with σ = α" uses α·F^(β−1) when β < 1 (the equivalent lognormal level).
- Cap: simulations × steps ≤ 5·10⁷, otherwise a warning and no run.

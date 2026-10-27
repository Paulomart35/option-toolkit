# =============================================================
# Help content for every module: theory, assumptions, SymPy formulas (+ "where:" legends),
# how to use, limitations. Formulas are SymPy expressions (HW6 approach: custom N and n
# Functions for the standard normal cdf / pdf), rendered with st.latex(sympy.latex(expr)).
# =============================================================
import sympy as smp


# ---------- Standard normal distribution (HW6) ----------
class n(smp.Function):
    """Density n(x), with n'(x) = -x n(x)."""
    def fdiff(self, argindex=1):
        x = self.args[0]
        return -x * n(x)


class N(smp.Function):
    """Cumulative distribution N(x), with N'(x) = n(x)."""
    def fdiff(self, argindex=1):
        x = self.args[0]
        return n(x)


# ---------- Symbols ----------
S, K, sigma, tau, T, F = smp.symbols("S K sigma tau T F", positive=True)
r, q, d1, d2, phi = smp.symbols("r q d_1 d_2 phi", real=True)
C, P, V, Pi = smp.symbols("C P V Pi", real=True)
s_, n_c, m = smp.symbols("s n_c m", real=True)
Delta, Gamma = smp.symbols("Delta Gamma", real=True)

d1_formula = (smp.log(S / K) + (r - q + sigma**2 / 2) * tau) / (sigma * smp.sqrt(tau))
call_formula = S * smp.exp(-q * tau) * N(d1) - K * smp.exp(-r * tau) * N(d2)
put_formula = K * smp.exp(-r * tau) * N(-d2) - S * smp.exp(-q * tau) * N(-d1)

BS_LEGEND = ("S = spot price, K = strike, σ = volatility (decimal), r = risk-free rate (continuous), "
             "q = dividend yield (continuous), τ = time to maturity (years), N = standard normal "
             "cumulative distribution, n = standard normal density, φ = +1 for a call, −1 for a put.")


def _portfolio():
    Delta_w, Gamma_w = smp.symbols("Delta_i Gamma_i")
    i = smp.symbols("i", integer=True)
    return [
        ("d₁ and d₂", smp.Eq(d1, d1_formula), BS_LEGEND),
        ("", smp.Eq(d2, d1 - sigma * smp.sqrt(tau)), "d₂ is d₁ shifted by one standard deviation σ√τ."),
        ("Black-Scholes-Merton call", smp.Eq(C, call_formula), BS_LEGEND),
        ("Black-Scholes-Merton put", smp.Eq(P, put_formula), BS_LEGEND),
        ("Intrinsic value (= payoff if exercised now)",
         smp.Eq(smp.Symbol("IV"), smp.Max(phi * (S - K), 0)),
         "φ = +1 call: max(S − K, 0); φ = −1 put: max(K − S, 0)."),
        ("Time value", smp.Eq(smp.Symbol("TV"), smp.Symbol("Price") - smp.Symbol("IV")),
         "Part of the price that comes from the time left (optionality)."),
        ("Position market value", smp.Eq(V, s_ * n_c * m * smp.Symbol("Price")),
         "s = +1 long / −1 short, n_c = number of contracts, m = multiplier (shares per contract)."),
        ("Delta", smp.Eq(Delta, phi * smp.exp(-q * tau) * N(phi * d1)),
         "Change of the option price for a 1-unit move of S. Net delta (shares) = Σ s·n_c·m·Δ."),
        ("Gamma", smp.Eq(Gamma, smp.exp(-q * tau) * n(d1) / (S * sigma * smp.sqrt(tau))),
         "Change of delta for a 1-unit move of S (same for calls and puts)."),
        ("Portfolio payoff at maturity", smp.Eq(smp.Symbol("Payoff"),
                                                smp.Sum(smp.Indexed("s", i) * smp.Indexed("n_c", i) * smp.Indexed("m", i)
                                                        * smp.Max(smp.Indexed("phi", i) * (smp.Symbol("S_T") - smp.Indexed("K", i)), 0),
                                                        (i, 1, smp.Symbol("L")))),
         "S_T = spot at maturity, L = number of positions (HW Module B scenarios)."),
    ]


def _arbitrage():
    PVK = smp.Symbol("PV(K)")
    return [
        ("Put-call parity", smp.Eq(C - P, S * smp.exp(-q * T) - K * smp.exp(-r * T)),
         "C, P = European call/put prices (same K, T), S = spot, q = dividend yield, r = risk-free rate, T = maturity (years)."),
        ("Present value of the strike (zero-coupon bond)", smp.Eq(PVK, K * smp.exp(-r * T)),
         "Price today of a bond paying K at T."),
        ("Synthetic call", smp.Eq(smp.Symbol("C_syn"), P + S * smp.exp(-q * T) - K * smp.exp(-r * T)),
         "Long put + long e^{−qT} shares + borrow PV(K) (short the zero-coupon bond)."),
        ("Synthetic put", smp.Eq(smp.Symbol("P_syn"), C - S * smp.exp(-q * T) + K * smp.exp(-r * T)),
         "Long call + short e^{−qT} shares + lend PV(K) (buy the zero-coupon bond)."),
        ("Mispricing (gap)", smp.Eq(smp.Symbol("gap"), C - smp.Symbol("C_syn")),
         "gap > tolerance: call rich → sell call, buy synthetic. gap < −tolerance: call cheap → buy call, sell synthetic."),
        ("Call bounds", smp.And(smp.Ge(C, smp.Max(S * smp.exp(-q * T) - K * smp.exp(-r * T), 0)), smp.Le(C, S * smp.exp(-q * T))),
         "A call can never be worth more than the stock nor less than its discounted intrinsic value."),
        ("Put bounds", smp.And(smp.Ge(P, smp.Max(K * smp.exp(-r * T) - S * smp.exp(-q * T), 0)), smp.Le(P, K * smp.exp(-r * T))),
         "A put can never be worth more than PV(K)."),
        ("Locked-in profit", smp.Eq(smp.Symbol("Profit_0"), (smp.Abs(smp.Symbol("gap")) - smp.Symbol("c")) * n_c * m),
         "c = transaction cost / tolerance per share, n_c = contracts, m = multiplier. Value at T: Profit_0·e^{rT}."),
    ]


def _hedging():
    Nw, nh, ns = smp.symbols("N n_h n_s")
    Dw, Gw, Dh, Gh = smp.symbols("Delta_w Gamma_w Delta_h Gamma_h")
    mu, dt, Z, B = smp.symbols("mu Delta_t Z B")
    t = smp.symbols("t", integer=True)
    St = smp.Function("S")
    return [
        ("Delta hedge (underlying only)", smp.Eq(ns, Nw * Dw),
         "We are SHORT N written options; Δ_w = delta of one written option; n_s = shares to hold (positive = BUY)."),
        ("Delta-gamma hedge: hedging options", smp.Eq(nh, Nw * Gw / Gh),
         "Γ_w, Γ_h = gammas of the written and hedging options; requires Γ_h > 10⁻¹⁰."),
        ("Delta-gamma hedge: shares", smp.Eq(ns, Nw * Dw - nh * Dh), "Δ_h = delta of the hedging option."),
        ("Portfolio delta and gamma", smp.Eq(smp.Symbol("Delta_PF"), -Nw * Dw + nh * Dh + ns),
         "Both are 0 right after rebalancing a delta-gamma hedge."),
        ("", smp.Eq(smp.Symbol("Gamma_PF"), -Nw * Gw + nh * Gh), ""),
        ("Time left at day t", smp.Eq(smp.Symbol("tau_t"), T - t / 252), "Trading-day convention, Δt = 1/252."),
        ("Bank account (cash)", smp.Eq(smp.Function("B")(t), smp.Function("B")(t - 1) * smp.exp(r * dt)
                                      - smp.Symbol("trades_t")),
         "B(0) = premium received for the written options minus the initial hedge; trades are paid from the account."),
        ("Portfolio value and daily P&L", smp.Eq(smp.Function("V")(t), -Nw * smp.Symbol("C_w") + nh * smp.Symbol("H") + ns * St(t) + smp.Function("B")(t)),
         "C_w, H = prices of the written and hedging options; P&L_t = V(t) − V(t−1) with yesterday's holdings."),
        ("Spot Price Simulator (GBM, Hull 2021)",
         smp.Eq(St(t + 1), St(t) * smp.exp((mu - sigma**2 / 2) * dt + sigma * smp.sqrt(dt) * Z)),
         "μ = drift (default r), σ = volatility, Δt = 1/252, Z ~ N(0,1) independent each day."),
    ]


def _sabr():
    alpha, beta, rho, nu = smp.symbols("alpha beta rho nu", real=True)
    z, x = smp.symbols("z x")
    dF, dW, dZ, dalpha, dt = smp.symbols("dF dW dZ dalpha dt")
    FK = F * K
    logFK = smp.log(F / K)
    hagan = (alpha / (FK**((1 - beta) / 2) * (1 + (1 - beta)**2 / 24 * logFK**2 + (1 - beta)**4 / 1920 * logFK**4))
             * (z / smp.Function("x")(z))
             * (1 + ((1 - beta)**2 / 24 * alpha**2 / FK**(1 - beta)
                     + rho * beta * nu * alpha / (4 * FK**((1 - beta) / 2))
                     + (2 - 3 * rho**2) / 24 * nu**2) * T))
    leg = ("F = forward, K = strike, T = maturity (years), α = volatility level, β = elasticity (backbone; 1 for "
           "equities), ρ = correlation between price and volatility, ν = volatility of volatility.")
    i = smp.symbols("i", integer=True)
    return [
        ("SABR dynamics", smp.Eq(dF, alpha * F**beta * dW), leg),
        ("", smp.Eq(dalpha, nu * alpha * dZ), "The volatility α itself is random (lognormal)."),
        ("", smp.Eq(dW * dZ, rho * dt), "The two Brownian motions are correlated with ρ."),
        ("Forward (non-dividend stock)", smp.Eq(F, S * smp.exp(r * T)), "S = spot, r = risk-free rate."),
        ("Hagan (2002) implied volatility", smp.Eq(smp.Symbol("sigma_B"), hagan), leg),
        ("", smp.Eq(z, nu / alpha * FK**((1 - beta) / 2) * logFK), "At the money (z → 0), z/x(z) → 1."),
        ("", smp.Eq(smp.Function("x")(z), smp.log((smp.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))), ""),
        ("Option price with the Hagan vol (Black-76)",
         smp.Eq(smp.Symbol("V"), smp.exp(-r * T) * phi * (F * N(phi * d1) - K * N(phi * d2))),
         "d₁ = [ln(F/K) + σ_B²T/2]/(σ_B√T), d₂ = d₁ − σ_B√T, φ = +1 call / −1 put."),
        ("Calibration objective (per maturity)",
         smp.Eq(smp.Symbol("SSE"), smp.Sum((smp.Indexed("sigma^mkt", i) - smp.Indexed("sigma^SABR", i))**2, (i, 1, smp.Symbol("n")))),
         "Minimised with Differential Evolution (Rodriguez-Mier 2017). RMSE = √(SSE/n)."),
    ]


def _sabr_mc():
    s, nu, rho, dt, Z1, Z2, X, r_ = smp.symbols("sigma nu rho Delta_t Z_1 Z_2 X r")
    t = smp.symbols("t")
    st = smp.Function("sigma")
    Xf = smp.Function("X")
    W = rho * Z1 + smp.sqrt(1 - rho**2) * Z2
    M = smp.symbols("M", integer=True)
    j = smp.symbols("j", integer=True)
    return [
        ("Log-price update (all schemes, β = 1)",
         smp.Eq(Xf(t + dt), Xf(t) + (r_ - st(t)**2 / 2) * dt + st(t) * smp.sqrt(dt) * W),
         "X = ln(S/S₀), σ = volatility (our α at t=0), Z₁, Z₂ independent N(0,1), ρ = correlation. "
         "The professor's std_MC omits r: here the drift r is added and the payoff discounted by e^{−rT}."),
        ("1. Euler-Maruyama (professor's version)", smp.Eq(st(t + dt), st(t) + nu * st(t) * smp.sqrt(dt) * Z1),
         "ν = vol-of-vol. The vol can become negative."),
        ("2. Full truncation", smp.Eq(st(t + dt), st(t) + nu * smp.Max(st(t), 0) * smp.sqrt(dt) * Z1),
         "σ⁺ = max(σ, 0) is used in the drift and diffusion of both equations (production standard)."),
        ("3. Log transform (exact for the vol)", smp.Eq(st(t + dt), st(t) * smp.exp(nu * smp.sqrt(dt) * Z1 - nu**2 * dt / 2)),
         "The vol stays strictly positive."),
        ("Option price estimator", smp.Eq(smp.Symbol("hat{V}"), smp.exp(-r_ * T) / M * smp.Sum(smp.Max(phi * (smp.Indexed("S_T", j) - K), 0), (j, 1, M))),
         "M = number of simulations, S_T = simulated final price, φ = +1 call / −1 put."),
        ("Standard error and 95% confidence interval", smp.Eq(smp.Symbol("SE"), smp.Symbol("s_payoff") / smp.sqrt(M)),
         "CI = V̂ ± 1.96·SE; s_payoff = sample standard deviation of the discounted payoffs."),
        ("Martingale check", smp.Eq(smp.Symbol("E[S_T]"), smp.Symbol("S_0") * smp.exp(r_ * T)),
         "The simulated mean of S_T must be close to the forward."),
    ]


HELP = {
    "portfolio": {
        "title": "Portfolio Evaluation",
        "theory": ("A European option gives the right to buy (call) or sell (put) a stock at the strike K on the "
                   "maturity date. This module values a whole portfolio of such options on one stock. Each option "
                   "is priced with the Black-Scholes-Merton model. Its **intrinsic value** is what it would pay if "
                   "exercised right now; the rest of the price is **time value**. A long position adds value, a short "
                   "position (an option you sold) subtracts it."),
        "assumptions": ["European options (exercise only at maturity)", "Black-Scholes-Merton: lognormal stock, constant σ, r and q",
                        "Frictionless market (no costs, no taxes), continuous trading",
                        "All positions are on the same underlying, priced with the shared spot S"],
        "formulas": _portfolio,
        "how": ("1. Enter the spot S (fetch it online or type it), r and q. 2. Edit the positions table (add rows "
                "with the + button). Leave *Market price* empty to use the Black-Scholes price. 3. Read the totals, "
                "per-position table and charts; enter your own S_T scenarios. 4. Download CSV/Excel or the PDF report."),
        "limits": ("Black-Scholes uses one constant volatility per option (no smile). Market prices typed by the user "
                   "override the model price but the Greeks still come from the model."),
    },
    "arbitrage": {
        "title": "Arbitrage Calculator",
        "theory": ("Put-call parity links the price of a European call and a European put with the same strike and "
                   "maturity. A call can be **built synthetically** with a put, the stock and a loan (bond). If the "
                   "market call costs more than the synthetic one, you sell the expensive one and buy the cheap one: "
                   "the positions cancel exactly at maturity, so the difference collected today is a riskless profit. "
                   "Viewing it from the call side or the put side gives the same four trades."),
        "assumptions": ["European options on the same stock, same K and T", "Borrowing and lending at the risk-free rate r",
                        "Short selling allowed; dividends paid continuously at rate q and reinvested in the stock",
                        "Transaction costs summarised by one tolerance per share"],
        "formulas": _arbitrage,
        "how": ("Enter S, K, T, r, q and the observed market prices of the call and the put (plus a tolerance / cost "
                "per share and the number of contracts). The page shows both sides of parity, the synthetics, the "
                "no-arbitrage bounds and, if an opportunity exists, the numbered strategy, the payoff table at T "
                "and the locked-in profit."),
        "limits": ("Real markets have bid-ask spreads, margin and borrowing costs, and American early exercise; "
                   "the tolerance field is a simple way to account for them."),
    },
    "hedging": {
        "title": "Delta and Delta-Gamma Hedging",
        "theory": ("If you sell options, you lose money when the stock moves against you. **Delta hedging** holds "
                   "shares so that small stock moves cancel out. **Delta-gamma hedging** also buys another option so "
                   "that the hedge stays good for larger moves. Because delta and gamma change every day, the hedge "
                   "must be **rebalanced**: this module computes the trades day by day on a real (historical) or a "
                   "simulated (forecast) path of prices, and tracks the cash account and the P&L."),
        "assumptions": ["We are short N written European options (HW7 convention)", "Black-Scholes Greeks with constant σ and r",
                        "Daily rebalancing at the close, no transaction costs", "Cash earns/pays r continuously (e^{rΔt} per day)",
                        "Forecast spots follow a geometric Brownian motion (Hull 2021), at most 5 days"],
        "formulas": _hedging,
        "how": ("1. Set S, r, σ and the written option (type, K, T, N). 2. Choose delta or delta-gamma (then give the "
                "hedging option). 3. *Static* tab: the hedge today and the P&L after an instant spot move. "
                "4. *Historical* tab: fetch or type daily closes and replay the hedge. 5. *Forecast* tab: simulate up "
                "to 5 days of spots and compare the P&L distribution of each strategy. Download CSV and PDF reports."),
        "limits": ("Discrete rebalancing leaves residual risk; real hedges pay bid-ask costs; the model volatility "
                   "may differ from the realised one (that difference drives the hedged P&L)."),
    },
    "sabr": {
        "title": "SABR Calculator",
        "theory": ("Market implied volatilities are not flat: they form a **smile/skew** across strikes. The SABR "
                   "model (Hagan et al. 2002) lets the volatility itself move randomly and be correlated with the "
                   "price. Hagan's formula turns the four SABR parameters into an implied volatility for any strike, "
                   "which is then plugged into Black-Scholes to price options. Parameters are **calibrated** maturity "
                   "by maturity to market data with Differential Evolution.\n\n"
                   "**Notation mapping:** in this app α = volatility level and ν = vol-of-vol. In the professor's "
                   "Monte Carlo code and in the Alòs & García thesis, `sigma0`/`initial_volatility` = our α and "
                   "`alpha`/`vol_of_vol` = our ν."),
        "assumptions": ["Non-dividend stock: F = S·e^{rT}", "β = 1 for equities (lognormal backbone) unless 'calibrate β' is on",
                        "Hagan's formula is an asymptotic approximation (best for short T and strikes near F)",
                        "One parameter set per maturity (as Alòs & García 2021 and HW8)"],
        "formulas": _sabr,
        "how": ("1. Upload a CSV/Excel file (course file or simple strike, T, impliedVolatility) or use the sample. "
                "2. Set S, r and the valuation date, then run the calibration. 3. Inspect the smiles, the S/F grid, "
                "the pricing table and the 3D surface. 4. Enter a market price to test for a mispriced option. "
                "A manual-parameters mode lets you compute a Hagan vol directly."),
        "limits": ("The mispricing strategy is **model-based (relative value)**: it is profitable only if SABR is right "
                   "and needs rebalancing — it is not riskless (the riskless one is the Arbitrage module). "
                   "β and ρ both create skew, so their individual values are not well identified."),
    },
    "sabr_mc": {
        "title": "SABR Monte Carlo",
        "theory": ("Instead of Hagan's approximation, we can simulate the SABR equations step by step: many random "
                   "paths of the volatility and the stock price are drawn, the option payoff is computed at maturity "
                   "on each path, averaged and discounted. More simulations give a more precise price (the standard "
                   "error falls like 1/√M)."),
        "assumptions": ["Risk-neutral drift r for the stock (added to the professor's std_MC, which omits r)",
                        "β = 1: log-price simulation; β < 1: forward simulated and absorbed at 0",
                        "Correlation convention of std_MC: Z₁ drives the vol, the price shock is ρZ₁ + √(1−ρ²)Z₂",
                        "European payoff, discounted at e^{−rT}"],
        "formulas": _sabr_mc,
        "how": ("Enter S₀, K, T, r, α (initial vol), ν, ρ, β, the option type, the number of simulations and time "
                "steps, the seed and the scheme. Run: you get E[S_T], the option price with standard error and 95% "
                "CI, comparisons with the Hagan/Black and flat-BS prices, paths, histogram, convergence and a table "
                "comparing the three schemes."),
        "limits": ("Time discretisation adds a bias (more steps reduce it). Euler can produce negative volatility; "
                   "the cloud limits simulations × steps to 5·10⁷."),
    },
}

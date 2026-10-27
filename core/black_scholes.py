# =============================================================
# Black-Scholes-Merton pricing and Greeks (HW4 / HW5 conventions)
#  - continuous dividend yield q (default 0)
#  - phi = +1 call / -1 put
#  - tau = time to maturity in YEARS (use days_to_years for days)
#  - theta = -dV/dtau (= dV/dt), as in the class summary (HW5)
# Every function accepts scalars or numpy arrays.
# At tau <= 0 the option is at maturity: price = payoff, delta = 0/±1, other Greeks = 0.
# =============================================================
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

TAU_MIN = 1e-12


def days_to_years(days):
    """Calendar days -> years, /365 as in HW4 and HW8."""
    return np.asarray(days, dtype=float) / 365.0


def phi_of(kind):
    """'call' -> +1, 'put' -> -1 (also accepts +1 / -1)."""
    if isinstance(kind, str):
        k = kind.strip().lower()
        if k in ("call", "c"):
            return 1
        if k in ("put", "p"):
            return -1
        raise ValueError(f"Unknown option type: {kind!r} (use call or put)")
    return 1 if kind > 0 else -1


def d1_d2(S, K, sigma, r, q, tau):
    """Black-Scholes d1 and d2."""
    S, K, sigma, tau = (np.asarray(x, dtype=float) for x in (S, K, sigma, tau))
    t = np.maximum(tau, TAU_MIN)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2


def payoff(S, K, phi=1):
    """Payoff at maturity max(phi*(S-K), 0) = intrinsic value."""
    return np.maximum(phi * (np.asarray(S, dtype=float) - K), 0.0)


def _out(x):
    """Return a Python float for scalar results, an array otherwise."""
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def bs_price(S, K, sigma, r, q, tau, phi=1):
    """European option price (Black-Scholes-Merton with dividend yield q)."""
    d1, d2 = d1_d2(S, K, sigma, r, q, tau)
    tau = np.asarray(tau, dtype=float)
    price = phi * (S * np.exp(-q * tau) * norm.cdf(phi * d1)
                   - K * np.exp(-r * tau) * norm.cdf(phi * d2))
    return _out(np.where(tau > 0, price, payoff(S, K, phi)))


def bs_delta(S, K, sigma, r, q, tau, phi=1):
    """Delta = dP/dS = phi e^{-q tau} N(phi d1)."""
    d1, _ = d1_d2(S, K, sigma, r, q, tau)
    tau = np.asarray(tau, dtype=float)
    delta = phi * np.exp(-q * tau) * norm.cdf(phi * d1)
    at_T = np.where(phi * (np.asarray(S, dtype=float) - K) > 0, float(phi), 0.0)
    return _out(np.where(tau > 0, delta, at_T))


def bs_gamma(S, K, sigma, r, q, tau, phi=1):
    """Gamma = e^{-q tau} n(d1) / (S sigma sqrt(tau)), same for calls and puts."""
    d1, _ = d1_d2(S, K, sigma, r, q, tau)
    tau = np.asarray(tau, dtype=float)
    t = np.maximum(tau, TAU_MIN)
    gamma = np.exp(-q * t) * norm.pdf(d1) / (S * sigma * np.sqrt(t))
    return _out(np.where(tau > 0, gamma, 0.0))


def bs_vega(S, K, sigma, r, q, tau, phi=1):
    """Vega = S e^{-q tau} n(d1) sqrt(tau) (per 1.00 change of sigma)."""
    d1, _ = d1_d2(S, K, sigma, r, q, tau)
    tau = np.asarray(tau, dtype=float)
    t = np.maximum(tau, 0.0)
    return _out(S * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t))


def bs_theta(S, K, sigma, r, q, tau, phi=1):
    """Theta = -dP/dtau (per year)."""
    d1, d2 = d1_d2(S, K, sigma, r, q, tau)
    tau = np.asarray(tau, dtype=float)
    t = np.maximum(tau, TAU_MIN)
    theta = (phi * q * S * np.exp(-q * t) * norm.cdf(phi * d1)
             - phi * r * K * np.exp(-r * t) * norm.cdf(phi * d2)
             - S * np.exp(-q * t) * norm.pdf(d1) * sigma / (2 * np.sqrt(t)))
    return _out(np.where(tau > 0, theta, 0.0))


def black76_price(F, K, sigma, r, tau, phi=1):
    """Black (1976) price on a forward F, discounted at r.
    For a non-dividend stock F = S e^{r tau}, so this equals bs_price(S, ..., q=0)."""
    return bs_price(np.asarray(F, dtype=float) * np.exp(-r * np.asarray(tau, dtype=float)),
                    K, sigma, r, 0.0, tau, phi)


def implied_vol(price, S, K, r, q, tau, phi=1, lo=1e-4, hi=5.0):
    """Invert Black-Scholes for sigma (Brent root search). Returns nan if no root
    (price outside the no-arbitrage bounds)."""
    try:
        f = lambda s: bs_price(S, K, s, r, q, tau, phi) - price
        if f(lo) * f(hi) > 0:
            return float("nan")
        return float(brentq(f, lo, hi, xtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


def greeks_table(S, K, sigma, r, q, tau, phi=1):
    """Price and main Greeks in one dict."""
    args = (S, K, sigma, r, q, tau, phi)
    return {"price": bs_price(*args), "delta": bs_delta(*args), "gamma": bs_gamma(*args),
            "vega": bs_vega(*args), "theta": bs_theta(*args)}

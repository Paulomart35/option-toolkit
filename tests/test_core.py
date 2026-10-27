# Unit tests of core/ against tests/reference_values.md (T1-T12)
import numpy as np
import pandas as pd
import pytest

from core.black_scholes import (bs_price, bs_delta, bs_gamma, bs_theta, bs_vega, implied_vol,
                                black76_price, days_to_years)
from core import portfolio, parity, hedging, sabr, sabr_mc, reports


# ---------------- Black-Scholes ----------------
def test_T1_bs_call_put():
    assert bs_price(100, 100, 0.20, 0.05, 0, 1, 1) == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(100, 100, 0.20, 0.05, 0, 1, -1) == pytest.approx(5.5735, abs=1e-4)


def test_T2_bs_hw4():
    assert bs_price(200, 210, 0.40, 0.05, 0, days_to_years(120), 1) == pytest.approx(15.4553, abs=1e-4)


def test_T3_bs_dividend_hw5():
    assert bs_price(40, 40, 0.30, 0.05, 0.02, 0.5, 1) == pytest.approx(3.6233, abs=1e-4)


def test_greeks_finite_differences():
    S, K, s, r, q, t = 40, 40, 0.3, 0.05, 0.02, 0.5
    h = 1e-4
    for phi in (1, -1):
        p = lambda S_=S, s_=s, t_=t: bs_price(S_, K, s_, r, q, t_, phi)
        assert bs_delta(S, K, s, r, q, t, phi) == pytest.approx((p(S + h) - p(S - h)) / (2 * h), abs=1e-6)
        assert bs_gamma(S, K, s, r, q, t, phi) == pytest.approx((p(S + h) - 2 * p() + p(S - h)) / h**2, abs=1e-3)
        assert bs_vega(S, K, s, r, q, t, phi) == pytest.approx((p(s_=s + h) - p(s_=s - h)) / (2 * h), abs=1e-5)
        assert bs_theta(S, K, s, r, q, t, phi) == pytest.approx(-(p(t_=t + h) - p(t_=t - h)) / (2 * h), abs=1e-4)


def test_implied_vol_roundtrip_and_black76():
    c = bs_price(100, 110, 0.27, 0.03, 0, 0.75, 1)
    assert implied_vol(c, 100, 110, 0.03, 0, 0.75, 1) == pytest.approx(0.27, abs=1e-8)
    F = 100 * np.exp(0.03 * 0.75)
    assert black76_price(F, 110, 0.27, 0.03, 0.75, 1) == pytest.approx(c, abs=1e-10)


def test_at_maturity_is_payoff():
    assert bs_price(105, 100, 0.2, 0.05, 0, 0, 1) == pytest.approx(5.0)
    assert bs_price(95, 100, 0.2, 0.05, 0, 0, -1) == pytest.approx(5.0)


# ---------------- Portfolio ----------------
def test_T9_intrinsic_value():
    df = pd.DataFrame([["X", "call", "long", 95.0, 0.5, 2, 100, 0.2, None]], columns=portfolio.COLUMNS)
    res, tot = portfolio.value_portfolio(df, 100, 0.05)
    assert tot["Intrinsic value"] == pytest.approx(1000.0)
    assert tot["Market value"] == pytest.approx(200 * bs_price(100, 95, 0.2, 0.05, 0, 0.5, 1))
    assert tot["Time value"] == pytest.approx(tot["Market value"] - 1000.0)


def test_portfolio_module_b_payoffs():
    # HW Module B: long call 105, short 2 puts 95, short call 110 (multiplier 1)
    df = pd.DataFrame([["X", "call", "long", 105, 1, 1, 1, 0.2, None],
                       ["X", "put", "short", 95, 1, 2, 1, 0.2, None],
                       ["X", "call", "short", 110, 1, 1, 1, 0.2, None]], columns=portfolio.COLUMNS)
    pay = portfolio.payoff_at_maturity(df, [80, 90, 100, 105, 110, 120])
    assert np.allclose(pay, [-30, -10, 0, 0, 5, 5])


def test_portfolio_market_price_and_validation():
    df = pd.DataFrame([["X", "put", "short", 100, 1, 1, 100, 0.2, 7.0]], columns=portfolio.COLUMNS)
    res, tot = portfolio.value_portfolio(df, 100, 0.05)
    assert tot["Market value"] == pytest.approx(-700)
    bad = df.copy(); bad.loc[0, "K"] = -1
    with pytest.raises(ValueError):
        portfolio.value_portfolio(bad, 100, 0.05)


# ---------------- Parity / arbitrage ----------------
def test_T5_parity_arbitrage():
    C = bs_price(100, 100, 0.2, 0.05, 0, 1, 1)
    P = bs_price(100, 100, 0.2, 0.05, 0, 1, -1)
    res = parity.parity_analysis(100, 100, 1, 0.05, C, P)
    assert res["lhs"] == pytest.approx(4.8771, abs=1e-4)
    assert res["rhs"] == pytest.approx(4.8771, abs=1e-4)
    assert res["pv_K"] == pytest.approx(95.1229, abs=1e-4)
    res = parity.parity_analysis(100, 100, 1, 0.05, 12.0, P)
    assert res["decision"] == "call_overpriced"
    steps, pay, summ = parity.build_strategy(res, contracts=1, multiplier=1)
    assert summ["profit_today"] == pytest.approx(1.5494, abs=1e-4)
    assert list(steps["Action"])[:3] == ["SELL", "BUY", "BUY"]
    assert "BORROW" in steps["Action"].iloc[3]
    assert np.allclose(pay["Total"], 0.0)
    legs = parity.leg_payoffs(res, np.linspace(50, 150, 11), 1, 1)
    assert np.allclose(legs["Total (at T)"], 0.0)


def test_parity_underpriced_with_dividend_and_scaling():
    S, K, T, r, q = 100, 95, 0.5, 0.04, 0.03
    C = bs_price(S, K, 0.25, r, q, T, 1) - 1.0
    P = bs_price(S, K, 0.25, r, q, T, -1)
    res = parity.parity_analysis(S, K, T, r, C, P, q=q, tol=0.1)
    assert res["decision"] == "call_underpriced"
    steps, pay, summ = parity.build_strategy(res, contracts=3, multiplier=100)
    assert summ["profit_today"] == pytest.approx((1.0 - 0.1) * 300, abs=1e-6)
    assert np.allclose(pay["Total"], 0.0)
    assert steps["Quantity"].iloc[2] == pytest.approx(300 * np.exp(-q * T))


def test_T10_bounds():
    b = parity.check_bounds(100, 100, 1, 0.05, 101.0, 5.0)
    assert not b.loc[b["Bound"] == "Call upper bound", "OK"].iloc[0]


def test_T12_no_arbitrage_when_consistent():
    C = bs_price(100, 105, 0.3, 0.02, 0.01, 0.7, 1)
    P = bs_price(100, 105, 0.3, 0.02, 0.01, 0.7, -1)
    res = parity.parity_analysis(100, 105, 0.7, 0.02, C, P, q=0.01, tol=1e-6)
    assert res["decision"] == "none"
    assert res["bounds"]["OK"].all()


# ---------------- Hedging ----------------
HW7 = dict(S=100, r=0.03, sigma=0.20, N=10, written={"kind": "call", "K": 100, "T": 1.0},
           strategy="delta-gamma", hedge={"kind": "call", "K": 110, "T": 2.0}, n_override=None)


def test_T4_hw7_hedge():
    h = hedging.compute_hedge(HW7)
    assert h["d_w"] == pytest.approx(0.5987, abs=1e-4)
    assert h["g_w"] == pytest.approx(0.019333, abs=1e-6)
    assert h["d_h"] == pytest.approx(0.5066, abs=1e-4)
    assert h["g_h"] == pytest.approx(0.014103, abs=1e-6)
    assert h["n_h"] == pytest.approx(13.7089, abs=1e-4)
    assert h["n_s"] == pytest.approx(-0.9581, abs=1e-4)
    assert abs(h["port_delta"]) < 1e-12 and abs(h["port_gamma"]) < 1e-12
    hd = hedging.compute_hedge({**HW7, "strategy": "delta"})
    assert hd["n_s"] == pytest.approx(5.9871, abs=1e-4)
    assert hedging.action(hd["n_s"]) == "BUY"


def test_hedge_gamma_validation():
    p = {**HW7, "hedge": {"kind": "call", "K": 1000, "T": 0.01}}
    with pytest.raises(ValueError):
        hedging.compute_hedge(p)


def test_scenarios_hw7():
    h = hedging.compute_hedge(HW7)
    res = hedging.run_scenarios(HW7, h)
    mid = res.iloc[30]
    assert mid["S"] == pytest.approx(100)
    assert mid["P&L hedged"] == pytest.approx(0)
    # delta-gamma hedge is much flatter than unhedged near S0
    assert abs(res.iloc[27]["P&L hedged"]) < abs(res.iloc[27]["P&L unhedged"]) / 10


def test_T11_engine_constant_spot_dg():
    dt = 1e-5
    spots = np.full(6, 100.0)
    eng = hedging.rebalance_engine(spots, 0.03, 0.2, 10, HW7["written"], "delta-gamma", HW7["hedge"], dt=dt)
    assert np.max(np.abs(eng["pnl"])) < 1e-6
    assert np.allclose(eng["value"][:, 0], 0.0)
    # day-0 holdings equal the static HW7 hedge
    assert eng["n_h"][0, 0] == pytest.approx(13.7089, abs=1e-4)
    assert eng["n_s"][0, 0] == pytest.approx(-0.9581, abs=1e-4)


def test_engine_self_financing_and_report():
    spots = hedging.simulate_gbm(100, 0.03, 0.2, 5, n_paths=200, seed=1)
    assert spots.shape == (200, 6)
    pnl = {}
    for strat in ("unhedged", "delta", "delta-gamma"):
        eng = hedging.rebalance_engine(spots, 0.03, 0.2, 10, HW7["written"], strat, HW7["hedge"])
        assert np.allclose(eng["cum_pnl"][:, -1], eng["value"][:, -1])
        pnl[strat] = eng["cum_pnl"][:, -1]
    assert pnl["delta-gamma"].std() < pnl["delta"].std() < pnl["unhedged"].std()
    rep = hedging.daily_report(eng, 0)
    assert len(rep) == 6 and rep["Instruction"].iloc[0].startswith("Day 0")
    stats = hedging.pnl_distribution(pnl)
    assert list(stats["Strategy"]) == ["unhedged", "delta", "delta-gamma"]


def test_gbm_mean():
    S = hedging.simulate_gbm(100, 0.05, 0.2, 5, n_paths=200000, seed=3)
    assert S[:, -1].mean() == pytest.approx(100 * np.exp(0.05 * 5 / 252), rel=2e-3)
    with pytest.raises(ValueError):
        hedging.simulate_gbm(100, 0.05, 0.2, 6)


# ---------------- SABR ----------------
def test_T6_hagan_vol():
    K = np.array([300, 350, 410, 450, 500.0])
    v = sabr.sabr_vol(K, 410, 0.5, 0.55, 1.0, -0.4, 1.2)
    assert np.allclose(v, [0.655003, 0.602296, 0.556930, 0.537523, 0.524644], atol=1e-6)
    Kn = K[K != 410]
    th = sabr.SABR_calibration_thesis(Kn, 410, 0.5, 1.2, -0.4, 0.55)
    assert np.max(np.abs(sabr.sabr_vol(Kn, 410, 0.5, 0.55, 1.0, -0.4, 1.2) - th)) < 1e-12


def test_calibration_recovers_params():
    p_true = dict(alpha=0.30, beta=1.0, rho=-0.5, nu=0.9)
    F, T = 105.0, 0.5
    K = np.linspace(0.85, 1.15, 13) * F
    iv = sabr.sabr_vol(K, F, T, **p_true)
    df = pd.DataFrame({"expiry": "x", "T": T, "F": F, "strike": K, "iv": iv})
    out = sabr.calibrate(df, beta=1.0, popsize=20, its=150, seed=42)
    assert out["RMSE"].iloc[0] < 2e-3
    assert out["alpha"].iloc[0] == pytest.approx(0.30, abs=0.01)
    assert out["beta"].iloc[0] == 1.0


def test_sample_csv_loads_and_calibrates():
    raw = pd.read_csv("sample_data/sabr_sample.csv")
    df = sabr.clean_simple_file(raw, S0=100.0, r=0.05)
    assert df["T"].nunique() == 3
    out = sabr.calibrate(df, beta=1.0, popsize=15, its=40, seed=42)
    assert (out["RMSE"] < 0.01).all()


def test_simple_file_with_prices():
    S, r, T = 100.0, 0.05, 0.5
    K = np.array([90, 100, 110.0])
    prices = bs_price(S, K, 0.25, r, 0, T, 1)
    raw = pd.DataFrame({"strike": K, "T": T, "price": prices})
    df = sabr.clean_simple_file(raw, S, r)
    assert np.allclose(df["iv"], 0.25, atol=1e-6)


def test_pricing_table_and_mispricing():
    p = dict(alpha=0.2, beta=1.0, rho=-0.4, nu=0.5)
    tab = sabr.pricing_table([90, 100, 110], 100, 1.0, 0.05, p)
    assert tab["SABR price"].iloc[1] == pytest.approx(10.6972, abs=1e-3)
    res = sabr.mispricing_strategy(12.0, 100, 1.0, 100, 0.05, p, phi=1, tol=0.05, contracts=1, multiplier=100)
    assert res["decision"] == "overpriced"
    assert res["steps"]["Action"].iloc[0] == "SELL"
    assert res["steps"]["Action"].iloc[1] == "BUY"
    assert res["steps"]["Cash flow at t=0"].sum() == pytest.approx(0.0, abs=1e-8)
    grid = sabr.smile_grid([90, 100], [90, 100, 110], 1.0, 0.05, p)
    assert len(grid) == 6


# ---------------- SABR Monte Carlo ----------------
def test_T7_mc_bs_limit():
    res = sabr_mc.simulate(100, 100, 1, 0.05, 0.20, 1e-8, 0.0, 1.0, 200000, 100, "log", 42)
    assert abs(res["price"] - 10.4506) < 3 * res["se"]
    assert res["mean_ST"] == pytest.approx(105.127, rel=3e-3)


@pytest.mark.parametrize("scheme", ["euler", "full", "log"])
def test_T8_mc_sabr(scheme):
    res = sabr_mc.simulate(100, 100, 1, 0.05, 0.20, 0.5, -0.4, 1.0, 200000, 252, scheme, 42, n_store=5)
    b = sabr_mc.benchmarks(100, 100, 1, 0.05, 0.20, 0.5, -0.4)
    assert b["hagan_vol"] == pytest.approx(0.20657, abs=1e-5)
    assert b["black_hagan"] == pytest.approx(10.6972, abs=1e-4)
    assert abs(res["price"] - b["black_hagan"]) < 3.5 * res["se"]
    assert res["price"] == pytest.approx(10.72, abs=0.1)
    assert res["S_paths"].shape == (5, 253)


def test_mc_beta_below_one_runs():
    res = sabr_mc.simulate(100, 100, 1, 0.05, 2.0, 0.3, -0.3, 0.5, 20000, 50, "full", 1)
    assert res["mean_ST"] == pytest.approx(100 * np.exp(0.05), rel=0.02)
    tab = sabr_mc.compare_schemes(100, 100, 1, 0.05, 0.2, 0.5, -0.4, 1.0, 5000, 20)
    assert len(tab) == 3


# ---------------- Reports ----------------
def test_reports():
    df = pd.DataFrame({"a": range(50), "b": np.linspace(0, 1, 50)})
    assert reports.to_csv_bytes(df).startswith(b"a,b")
    assert reports.to_excel_bytes({"x": df})[:2] == b"PK"
    fig = reports.line_figure([0, 1], {"l": ([0, 1], "g-")}, "t", "x", "y")
    pdf = reports.make_pdf("Test", [("title", "T"), ("text", "hello"), ("table", ("tab", df)), ("figure", fig)])
    assert pdf[:4] == b"%PDF"

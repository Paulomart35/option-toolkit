# Interaction tests: PDF reports build on every module, alternative inputs work, course-format cleaning.
import pathlib

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import sabr
from core.black_scholes import bs_price

ROOT = pathlib.Path(__file__).resolve().parents[1]
PDF_KEYS = {"pages/1_Portfolio.py": "pf_pdf", "pages/2_Arbitrage.py": "arb_pdf", "pages/3_Hedging.py": "hdg_pdf",
            "pages/4_SABR.py": "sabr_pdf", "pages/5_SABR_Monte_Carlo.py": "mc_pdf"}


def _at(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("page,key", list(PDF_KEYS.items()))
def test_pdf_report_builds(page, key):
    at = _at(page)
    at.button(key=f"{key}_build").click().run()
    _ok(at)
    assert at.session_state[f"{key}_pdf"][:4] == b"%PDF"


def test_arbitrage_no_arbitrage_case():
    at = _at("pages/2_Arbitrage.py")
    at.number_input(key="arb_C").set_value(round(bs_price(100, 100, 0.2, 0.05, 0, 1, 1), 4)).run()
    _ok(at)
    assert any("No arbitrage" in i.value for i in at.info)


def test_arbitrage_underpriced_with_dividend():
    at = _at("pages/2_Arbitrage.py")
    at.number_input(key="arb_q").set_value(0.02).run()
    at.number_input(key="arb_C").set_value(5.0).run()
    _ok(at)
    assert any("UNDERPRICED" in s.value for s in at.success)


def test_hedging_delta_put():
    at = _at("pages/3_Hedging.py")
    at.radio(key="hdg_strat").set_value("delta").run()
    at.selectbox(key="hdg_wk").set_value("put").run()
    _ok(at)


def test_hedging_bad_hedge_option_message():
    at = _at("pages/3_Hedging.py")
    at.number_input(key="hdg_hK").set_value(100000.0).run()
    at.number_input(key="hdg_hT").set_value(0.01).run()
    assert not at.exception
    assert any("Gamma" in e.value for e in at.error)


def test_sabr_free_beta_and_put_pricing():
    at = _at("pages/4_SABR.py")
    at.toggle(key="sabr_fb").set_value(True).run()
    at.radio(key="pr_kind").set_value("put").run()
    _ok(at)
    calib = at.session_state["sabr_calib"]
    assert (calib["beta"] < 1.0 + 1e-12).all() and len(calib) == 3


def test_mc_scheme_switch_and_cap():
    at = _at("pages/5_SABR_Monte_Carlo.py")
    at.selectbox(key="mc_scheme").set_value("euler").run()
    _ok(at)
    at.number_input(key="mc_n").set_value(2_000_000).run()
    at.number_input(key="mc_steps").set_value(100).run()
    assert not at.exception
    assert any("exceeds" in w.value for w in at.warning)


def test_portfolio_bad_scenarios_message():
    at = _at("pages/1_Portfolio.py")
    at.text_input(key="pf_sc").set_value("abc").run()
    assert not at.exception
    assert any("Scenarios" in e.value for e in at.error)


def test_course_format_cleaning():
    # small synthetic file in the course layout: two snapshots of TSLA; only the last is kept
    S0, r, val = 400.0, 0.05, pd.Timestamp("2025-11-17")
    rows = []
    for snap in range(2):
        rows.append({"contractSymbol": "TSLA251219P00400000", "expirationDate": "2025-12-19", "Call_Put": "P",
                     "strike": 400, "bid": 1, "ask": 1.2, "volume": 5, "impliedVolatility": 0.5, "lastTradeDate": ""})
        for exp in ("2025-12-19", "2026-01-16"):
            for K in (300, 360, 400, 440, 470, 600):
                rows.append({"contractSymbol": f"TSLA251219C{K:08d}", "expirationDate": exp, "Call_Put": "C",
                             "strike": K, "bid": 1.0, "ask": 1.2, "volume": 10 + snap,
                             "impliedVolatility": 0.5 + 0.1 * snap, "lastTradeDate": "2025-11-17"})
    raw = pd.DataFrame(rows)
    assert sabr.is_course_format(raw) and sabr.course_tickers(raw) == ["TSLA"]
    df = sabr.clean_course_file(raw, "TSLA", S0, r, val)
    assert set(df["strike"]) == {360, 400, 440, 470}             # 0.8 <= K/F <= 1.2
    assert np.allclose(df["iv"], 0.6)                            # last snapshot only
    assert df["T"].nunique() == 2

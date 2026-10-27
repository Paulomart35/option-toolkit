# Smoke tests: every page loads with default inputs without exception (streamlit.testing.v1.AppTest),
# the Help dialog opens, and every SymPy help formula renders to LaTeX.
import pathlib

import pytest
import sympy as smp
from streamlit.testing.v1 import AppTest

from help.help_content import HELP

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGES = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "pages").glob("*.py"))
TIMEOUT = 120


def _run(path):
    at = AppTest.from_file(str(ROOT / path), default_timeout=TIMEOUT)
    at.run()
    return at


def _no_errors(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_app_shell_loads():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    _no_errors(at)


@pytest.mark.parametrize("page", PAGES)
def test_page_loads_with_defaults(page):
    at = _run(page)
    _no_errors(at)


@pytest.mark.parametrize("page", [p for p in PAGES if not p.endswith("0_Home.py")])
def test_help_button_opens(page):
    at = _run(page)
    btns = [b for b in at.button if b.key and b.key.startswith("help_btn_")]
    assert btns, "Help button missing"
    btns[0].click().run()
    _no_errors(at)
    assert len(at.latex) > 0


@pytest.mark.parametrize("key", list(HELP))
def test_help_formulas_render(key):
    forms = HELP[key]["formulas"]()
    assert len(forms) >= 3
    for title, expr, legend in forms:
        assert smp.latex(expr)

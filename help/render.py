# Help button -> dialog with theory, assumptions, SymPy formulas (st.latex(sympy.latex(expr))),
# how to use and limitations.
import streamlit as st
import sympy as smp

from help.help_content import HELP


def _latex(expr):
    return smp.latex(expr, ln_notation=True)


def render_help(key):
    h = HELP[key]
    st.markdown(f"### {h['title']}")
    t1, t2, t3, t4 = st.tabs(["Theory", "Formulas", "How to use", "Limits"])
    with t1:
        st.markdown(h["theory"])
        st.markdown("**Assumptions**")
        st.markdown("\n".join(f"- {a}" for a in h["assumptions"]))
    with t2:
        for title, expr, legend in h["formulas"]():
            if title:
                st.markdown(f"**{title}**")
            st.latex(_latex(expr))
            if legend:
                st.caption("where: " + legend)
    with t3:
        st.markdown(h["how"])
    with t4:
        st.markdown(h["limits"])


@st.dialog("Help", width="large")
def _dialog(key):
    render_help(key)


def help_button(key):
    """Visible Help button opening the module's help dialog."""
    if st.button("Help", icon=":material/help:", key=f"help_btn_{key}",
                 help="Theory, assumptions, formulas and how to use this module"):
        _dialog(key)

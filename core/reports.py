# =============================================================
# Export helpers: CSV / Excel bytes and printable PDF reports (matplotlib PdfPages, no kaleido).
# A report is a list of blocks: ("title", str) | ("text", str) | ("table", DataFrame) | ("figure", Figure)
# =============================================================
import io
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

A4 = (8.27, 11.69)
ROWS_PER_PAGE = 32
MAX_COLS = 9


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(sheets):
    """sheets: dict name -> DataFrame. Returns xlsx bytes (openpyxl)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=str(name)[:31], index=False)
    return buf.getvalue()


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{v:,.4f}" if abs(v) < 1e5 else f"{v:,.0f}"
    return str(v)


def _table_pages(pdf, df, title):
    df = df.copy()
    col_chunks = [df.columns[i:i + MAX_COLS] for i in range(0, len(df.columns), MAX_COLS)] or [df.columns]
    for cols in col_chunks:
        for start in range(0, max(len(df), 1), ROWS_PER_PAGE):
            part = df[cols].iloc[start:start + ROWS_PER_PAGE]
            fig, ax = plt.subplots(figsize=A4)
            ax.axis("off")
            ax.set_title(title + (f" (rows {start + 1}-{start + len(part)})" if len(df) > ROWS_PER_PAGE else ""),
                         fontsize=11, loc="left")
            cells = [[textwrap.shorten(_fmt(v), 40) for v in row] for row in part.values] or [[""] * len(cols)]
            tab = ax.table(cellText=cells, colLabels=[textwrap.fill(str(c), 14) for c in cols],
                           loc="upper center", cellLoc="right")
            tab.auto_set_font_size(False)
            tab.set_fontsize(6.5)
            tab.scale(1, 1.25)
            pdf.savefig(fig)
            plt.close(fig)


def make_pdf(title, blocks):
    """Build a PDF report and return its bytes."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        text_buf = [title, "Generated " + datetime.now().strftime("%Y-%m-%d %H:%M"), ""]

        def flush():
            if not text_buf:
                return
            fig = plt.figure(figsize=A4)
            y = 0.95
            for line in text_buf:
                for sub in (textwrap.wrap(line, 95) or [""]):
                    fig.text(0.06, y, sub, fontsize=9, family="monospace", va="top")
                    y -= 0.018
                    if y < 0.05:
                        pdf.savefig(fig); plt.close(fig)
                        fig = plt.figure(figsize=A4); y = 0.95
            pdf.savefig(fig)
            plt.close(fig)
            text_buf.clear()

        for kind, content in blocks:
            if kind in ("title", "text"):
                text_buf.extend(([""] if kind == "title" else []) + [str(content)] +
                                (["-" * len(str(content))] if kind == "title" else []))
            elif kind == "table":
                flush()
                name, df = content if isinstance(content, tuple) else ("Table", content)
                _table_pages(pdf, df, name)
            elif kind == "figure":
                flush()
                pdf.savefig(content)
                plt.close(content)
        flush()
    return buf.getvalue()


def line_figure(x, series, title, xlabel, ylabel, vline=None):
    """Small matplotlib helper for PDF charts. series: dict label -> (y, style)."""
    fig, ax = plt.subplots(figsize=(8.27, 5))
    for label, (y, style) in series.items():
        ax.plot(x, y, style, label=label)
    if vline is not None:
        ax.axvline(vline, color="grey", ls=":")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

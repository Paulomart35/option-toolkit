# Generates sample_data/sabr_sample.csv: Hagan (2002) vols for 3 maturities + small noise.
# Spot S0 = 100, r = 0.05, beta = 1, valuation as of "today". Run: python sample_data/make_sample.py
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core.sabr import sabr_vol, forward  # noqa: E402

S0, r = 100.0, 0.05
TRUE = {0.25: (0.32, -0.55, 1.40), 0.50: (0.30, -0.50, 1.00), 1.00: (0.28, -0.45, 0.70)}  # alpha, rho, nu
rng = np.random.default_rng(7)
rows = []
for T, (a, rho, nu) in TRUE.items():
    F = float(forward(S0, r, T))
    for m in np.linspace(0.85, 1.15, 13):
        K = round(m * F, 1)
        iv = float(sabr_vol(K, F, T, a, 1.0, rho, nu)) + rng.normal(0, 0.002)
        rows.append({"strike": K, "T": T, "impliedVolatility": round(iv, 5)})
out = pathlib.Path(__file__).with_name("sabr_sample.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out, len(rows), "rows")

# Reference values for pytest (computed independently; script: tests/reference_values_script.py)

| # | Test | Inputs | Expected |
|---|------|--------|----------|
| T1 | BS call / put | S=100, K=100, T=1, r=0.05, σ=0.20, q=0 | call 10.4506, put 5.5735 (tol 1e-4) |
| T2 | BS vs HW4 | S=200, K=210, T=120/365, r=0.05, σ=0.40 | call 15.4553 (HW4 binomial N=110: 15.4460) |
| T3 | BS with dividend (HW5) | S=40, K=40, σ=0.30, r=0.05, q=0.02, τ=0.5 | call 3.6233 |
| T4 | HW7 hedge defaults | S=100, r=0.03, σ=0.20; short 10 calls K=100 T=1; hedge call K=110 T=2 | Δ_w 0.5987, Γ_w 0.019333, Δ_h 0.5066, Γ_h 0.014103; delta hedge = BUY 5.9871 shares; delta-gamma: BUY 13.7089 hedge calls, SELL 0.9581 shares; residual Δ=Γ=0 |
| T5 | Put-call parity | S=100, K=100, T=1, r=0.05, C, P from T1 | C−P = S−PV(K) = 4.8771, PV(K)=95.1229. Market call = 12.00 → call overpriced: sell call, buy put + stock, borrow 95.1229 → profit today 1.5494 per share |
| T6 | Hagan vol | F=410, T=0.5, α=0.55, β=1, ρ=−0.4, ν=1.2; K=300,350,410,450,500 | 0.655003, 0.602296, 0.556930, 0.537523, 0.524644 ; equals thesis formula (diff < 1e-12) |
| T7 | MC limit ν→0, ρ=0 | S0=100, K=100, T=1, r=0.05, α=0.20, 200k paths, 100 steps, seed 42 | price ≈ 10.4506 within 3 SE (SE ≈ 0.033); E[S_T] ≈ 105.127 |
| T8 | MC SABR β=1 | same, ν=0.5, ρ=−0.4, 200k paths, 252 steps | Euler / Full trunc / Log ≈ 10.72 (SE 0.031); Hagan vol 0.20657 → Black price 10.6972; MC within ~3 SE of Hagan-Black |
| T9 | Intrinsic value | long 2 contracts (×100) call K=95, S=100 | intrinsic 1,000 |
| T10 | Bounds | C > S e^{−qT} must flag arbitrage | flagged |
| T11 | DG engine | path with S constant and dt→0 | daily P&L ≈ 0 for delta-gamma hedge |
| T12 | Parity in arbitrage module | C, P priced at BS same σ | "no arbitrage" |

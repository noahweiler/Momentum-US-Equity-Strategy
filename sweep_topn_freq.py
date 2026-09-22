"""
Book size x rebalance frequency sweep.

`top_n` and `rebalance_freq` are the two first-order parameters in MomentumConfig that
were never measured -- they are the only fields without a justifying comment, because
nothing was ever run for them. Everything else in the config carries a number.

    python sweep_topn_freq.py              # in-sample sweep, the decision set
    python sweep_topn_freq.py oos          # ONE confirmation run, finalists only

READ PLATEAUS, NOT PEAKS. This grid is ~24 cells; at p < 0.05 you would expect more than
one spurious winner by chance, and this project has already caught a false positive that
reversed a headline result (no-trade band: +0.042 at p=0.022 in-sample, -0.088 out). A
value that wins while both its neighbours lose is noise. A broad region that wins is
structure.

Two design points that would otherwise confound the result:

1. `exit_buffer` is RELATIVE. b=10 on n=30 is a 33% buffer; holding b=10 while sweeping n
   down to 10 would silently turn it into a 100% buffer and change two things at once. It
   is scaled as b = round(n/3) to hold the ratio fixed.

2. `top_n` is a CAP, not a target. The gate (signal > 0) means fewer than n names qualify
   in a downtrend, so part of what this measures is how often the gate binds. `fill%` is
   reported for exactly that reason -- without it a large-n row is uninterpretable.
"""
import sys
import numpy as np
import pandas as pd

import backtest as bt
from Utils import performance_metrics
from data import load_benchmarks

IN_SAMPLE = ("2012-01-01", "2019-12-31")
OUT_SAMPLE = ("2022-01-01", "2026-06-30")

TOP_N = [10, 15, 20, 25, 30, 40, 50, 75]
FREQS = ["W", "2W", "M"]
FREQ_LABEL = {"W": "weekly", "2W": "fortnightly", "M": "monthly"}

# Panels are expensive and identical across cells; build once per window.
_panels, _orig_build = {}, bt.build_panel


def _cached_build_panel(panel_start, panel_end):
    key = (panel_start, panel_end)
    if key not in _panels:
        _panels[key] = _orig_build(panel_start=panel_start, panel_end=panel_end)
    return _panels[key]


bt.build_panel = _cached_build_panel
bt.panel_diagnostics = lambda *a, **k: None


def run_cell(start, end, top_n, freq):
    """One (top_n, freq) cell. Returns a metrics dict."""
    config = bt.MomentumConfig(
        top_n=top_n,
        exit_buffer=int(round(top_n / 3)),      # hold the buffer RATIO fixed, not its size
        rebalance_freq=freq,
    )
    portfolio, hist = bt.run_backtest(config, start, end)
    equity = portfolio.equity_series()
    m = performance_metrics(equity)

    trades = portfolio.trades_frame()
    notional = float((trades["qty"] * trades["fill"]).sum()) if len(trades) else 0.0
    fees = float(trades["commission"].sum()) + notional * 0.0005 if len(trades) else 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25

    # How often the gate binds: holdings below the cap means fewer than top_n names had
    # a positive signal, so this row is not really testing "top_n" at all.
    fill_pct = float(hist["n_holdings"].mean() / top_n) if len(hist) else float("nan")

    return {
        "top_n": top_n, "freq": freq,
        "cagr": m["cagr"], "sharpe": m["sharpe"], "vol": m["annualized_volatility"],
        "max_dd": m["max_drawdown"], "calmar": m["calmar"],
        "trades": len(trades), "fees_pa": fees / years if years else 0.0,
        "fill": fill_pct, "returns": equity.pct_change().dropna(),
    }


def paired_sharpe_test(returns_a, returns_b, reps=4000, block=20, seed=20260907):
    """Stationary block bootstrap on the PAIRED daily returns. Same method as elsewhere."""
    joined = pd.concat([returns_a, returns_b], axis=1, join="inner").dropna()
    a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    n = len(a)
    sharpe = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    observed = sharpe(a) - sharpe(b)

    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for r in range(reps):
        idx, pos = [], 0
        while pos < n:
            s = rng.integers(0, n)
            length = min(rng.geometric(1 / block), n - pos)
            idx.extend((s + np.arange(length)) % n)
            pos += length
        diffs[r] = sharpe(a[np.array(idx[:n])]) - sharpe(b[np.array(idx[:n])])
    return observed, diffs.std(), float(np.mean(np.abs(diffs - diffs.mean()) >= abs(observed)))


def grid(rows, key, fmt, title):
    print(f"\n{title}")
    print(f"{'top_n':>7}" + "".join(f"{FREQ_LABEL[f]:>14}" for f in FREQS))
    print("-" * (7 + 14 * len(FREQS)))
    for n in TOP_N:
        line = f"{n:>7}"
        for f in FREQS:
            r = next((x for x in rows if x["top_n"] == n and x["freq"] == f), None)
            line += f"{fmt(r[key]):>14}" if r else f"{'-':>14}"
        print(line)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "insample"
    start, end = OUT_SAMPLE if which == "oos" else IN_SAMPLE
    label = ("OUT-OF-SAMPLE (confirmation -- look once)" if which == "oos"
             else "IN-SAMPLE (decision set)")

    bench = load_benchmarks(start, end)["S&P 500"]
    bm = performance_metrics(bench / bench.iloc[0] * 100_000)

    rows = []
    for n in TOP_N:
        for f in FREQS:
            rows.append(run_cell(start, end, n, f))
            r = rows[-1]
            print(f"  n={n:<3} {FREQ_LABEL[f]:<12} "
                  f"CAGR {r['cagr']:>7.2%}  Sharpe {r['sharpe']:>5.2f}  "
                  f"DD {r['max_dd']:>7.2%}  trades {r['trades']:>5,}")

    print("\n" + "=" * 92)
    print(f"{label}   {start} .. {end}")
    print(f"SPY: CAGR {bm['cagr']:.2%}  Sharpe {bm['sharpe']:.2f}  "
          f"MaxDD {bm['max_drawdown']:.2%}")
    print("=" * 92)

    grid(rows, "sharpe", lambda v: f"{v:.2f}", "SHARPE  (read the plateau, not the peak)")
    grid(rows, "cagr", lambda v: f"{v:.2%}", "CAGR")
    grid(rows, "max_dd", lambda v: f"{v:.1%}", "MAX DRAWDOWN")
    grid(rows, "fees_pa", lambda v: f"${v:,.0f}", "COSTS PER YEAR")
    grid(rows, "fill", lambda v: f"{v:.0%}",
         "BOOK FILL  (holdings / top_n -- below 100% means the GATE binds, not top_n)")

    base = next(r for r in rows if r["top_n"] == 30 and r["freq"] == "W")
    best = max(rows, key=lambda r: r["sharpe"])
    print(f"\nCurrent config is n=30 weekly: Sharpe {base['sharpe']:.2f}, "
          f"CAGR {base['cagr']:.2%}")
    print(f"Best cell is n={best['top_n']} {FREQ_LABEL[best['freq']]}: "
          f"Sharpe {best['sharpe']:.2f}, CAGR {best['cagr']:.2%}")

    if best is not base:
        d, se, p = paired_sharpe_test(best["returns"], base["returns"])
        print(f"\nBest vs current, paired block bootstrap: diff {d:+.3f}  SE {se:.3f}  p {p:.3f}")
        print("  A single best cell out of 24 is the textbook multiple-comparisons trap.")
        print("  Treat the neighbourhood in the Sharpe grid as the evidence, not this p.")

    # Marginal effects, which are far more robust than any single cell.
    print("\nMARGINALS (median Sharpe across the other axis -- more robust than one cell)")
    print(f"{'top_n':>7}{'median Sharpe':>16}")
    for n in TOP_N:
        vals = [r["sharpe"] for r in rows if r["top_n"] == n]
        print(f"{n:>7}{np.median(vals):>16.3f}")
    print(f"\n{'freq':>7}{'median Sharpe':>16}{'median trades':>16}{'median $/yr':>14}")
    for f in FREQS:
        vals = [r["sharpe"] for r in rows if r["freq"] == f]
        tr = [r["trades"] for r in rows if r["freq"] == f]
        fe = [r["fees_pa"] for r in rows if r["freq"] == f]
        print(f"{FREQ_LABEL[f]:>7}{np.median(vals):>16.3f}{np.median(tr):>16,.0f}"
              f"{np.median(fe):>14,.0f}")
    print()


if __name__ == "__main__":
    main()

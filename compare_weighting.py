"""
Rank vs equal weighting, decided in-sample and confirmed out-of-sample.

    IN-SAMPLE  (decide here) : 2012-01-01 .. 2019-12-31
    OUT-OF-SAMPLE (test once): 2022-01-01 .. 2026-06-30

The 2020-2021 gap is deliberate -- it keeps the COVID crash and rebound out of both
sets rather than letting one regime dominate whichever window it lands in.

Usage:
    python compare_weighting.py insample
    python compare_weighting.py oos
"""
import sys
import numpy as np
import pandas as pd

from data import build_panel, trading_calendar, weekly_signal_dates
from Algorithms import momentum_signal, rank_universe
from backtest import _execute_rebalance, MomentumConfig
from portfolio import Portfolio, Fees
from Utils import performance_metrics, rolling_volatility

IN_SAMPLE = ("2012-01-01", "2019-12-31")
OUT_SAMPLE = ("2022-01-01", "2026-06-30")

MA_SHORT, MA_LONG, VOL_WINDOW = 50, 200, 30      # slow sleeve, vol=30 per the plan
TOP_N = 30

# No-trade band, applied to EVERY arm so the comparison is not rigged. It suppresses only
# small adjustments to positions already held, so the two `sized at entry` arms are
# untouched by it (they never adjust). Without it a continuously-varying scheme under
# reweight=True nudges all 30 names weekly and the run measures turnover, not weighting.
MIN_TRADE = 2_000.0

# Cap on any single name's weight. Uncapped inverse-vol piles into the quietest stock --
# measured at 27.2% of the book in-sample, which is a concentration bet, not a risk
# control, and the README already flags unbounded concentration as a known limitation.
# 2/TOP_N is twice the equal weight and was specified before the runs, not chosen after.
CAP = 2.0 / TOP_N

CONFIGS = [
    ("equal",  False, None, "A equal,  sized at entry "),   # current champion -- the bar
    ("equal",  True,  None, "B equal,  re-sized weekly"),   # isolates cost of re-weighting
    ("invvol", True,  None, "C invvol, re-sized weekly"),   # THE TEST, uncapped
    ("invvol", False, None, "D invvol, sized at entry "),   # vol sizing at entry only
    ("invvol", True,  CAP,  "E invvol, re-sized, capped"),  # THE TEST, capped at 2/N
]


def run(opens, closes, in_index, start, end, weighting, reweight, cap=None):
    """
    One weekly backtest.

    Returns (metrics, portfolio, daily returns, max position weight, risk-share ratio).

    `risk-share ratio` is the mechanism check: each holding's share of portfolio variance
    approximated as (w_i * sigma_i)^2 normalised, reported as max/min across the book. It
    is what inverse-vol is supposed to compress toward 1.0, and it is measurable whether
    or not performance moves.
    """
    masked = closes.where(in_index)
    signals = momentum_signal(masked, MA_SHORT, MA_LONG, VOL_WINDOW)
    vol_panel = masked.apply(lambda s: rolling_volatility(s, VOL_WINDOW))

    calendar = trading_calendar(closes)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    execution_of = {x: sig for sig, x in weekly_signal_dates(calendar)
                    if s <= sig <= e and x <= e}

    config = MomentumConfig(top_n=TOP_N, exit_buffer=0, ma_short=MA_SHORT, ma_long=MA_LONG,
                            vol_window=VOL_WINDOW, weighting=weighting, reweight=reweight,
                            weight_cap=cap, min_trade_notional=MIN_TRADE)
    portfolio = Portfolio(starting_cash=100_000, fees=Fees(1.0, 5.0))
    shortfall, concentration, risk_ratio = [], [], []

    for date in calendar[(calendar >= s) & (calendar <= e)]:
        if date in execution_of:
            signal_date = execution_of[date]
            targets = rank_universe(signals.loc[signal_date], top_n=TOP_N, gate_threshold=0.0)
            # Weights are decided on the signal close, applied at the next open, so the
            # volatility must be the signal date's -- reading `date` here is look-ahead.
            assert signal_date < date, f"{signal_date} must precede {date}"
            vols = vol_panel.loc[signal_date]
            _execute_rebalance(portfolio, targets, [], opens.loc[date].dropna(), date,
                               opens, config, [], shortfall, [], False, vols, None)

        day_close = closes.loc[date].dropna()
        portfolio.mark_to_market(day_close, date)

        if date in execution_of and portfolio.positions:
            values = pd.Series({t: sh * day_close.get(t, 0.0)
                                for t, sh in portfolio.positions.items()})
            if values.sum() > 0:
                concentration.append(values.max() / values.sum())
                w = values / values.sum()
                sig = vol_panel.loc[execution_of[date]].reindex(w.index)
                contrib = ((w * sig) ** 2).dropna()
                contrib = contrib[contrib > 0]
                if len(contrib) > 1:
                    risk_ratio.append(contrib.max() / contrib.min())

    equity = portfolio.equity_series()
    return (performance_metrics(equity), portfolio, equity.pct_change().dropna(),
            max(concentration) if concentration else float("nan"),
            float(np.median(risk_ratio)) if risk_ratio else float("nan"))


def paired_sharpe_test(returns_a, returns_b, reps=4000, block=20, seed=20260820):
    """
    Stationary block bootstrap on the PAIRED daily returns.

    The SE of a single Sharpe (~0.4 on 7 years) is the wrong yardstick for comparing two
    variants run on the same dates from correlated streams -- it overstates the noise by
    roughly 2x. Resampling both series with common block indices preserves the pairing.
    """
    joined = pd.concat([returns_a, returns_b], axis=1).dropna()
    a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    n = len(a)
    sharpe = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    observed = sharpe(a) - sharpe(b)

    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for r in range(reps):
        idx, pos = [], 0
        while pos < n:
            start = rng.integers(0, n)
            length = min(rng.geometric(1 / block), n - pos)
            idx.extend((start + np.arange(length)) % n)
            pos += length
        idx = np.array(idx[:n])
        diffs[r] = sharpe(a[idx]) - sharpe(b[idx])

    se = diffs.std()
    p = float(np.mean(np.abs(diffs - diffs.mean()) >= abs(observed)))
    return observed, se, p


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "insample"
    start, end = IN_SAMPLE if which == "insample" else OUT_SAMPLE
    label = "IN-SAMPLE (decision set)" if which == "insample" else "OUT-OF-SAMPLE (test set)"

    opens, closes, in_index, _ = build_panel(start, end, write_outputs=False)

    rows, daily = [], {}
    for weighting, reweight, cap, name in CONFIGS:
        metrics, portfolio, returns, max_pos, risk_ratio = run(
            opens, closes, in_index, start, end, weighting, reweight, cap)
        trades = pd.DataFrame(portfolio.trades)
        notional = (trades.qty * trades.fill).sum()
        rows.append({
            "config": name,
            "cagr": metrics["cagr"],
            "sharpe": metrics["sharpe"],
            "max_dd": metrics["max_drawdown"],
            "calmar": metrics["calmar"],
            "trades": len(trades),
            "fees": trades.commission.sum() + notional * 0.0005,
            "max_pos": max_pos,
            "risk_ratio": risk_ratio,
        })
        daily[name] = returns

    print()
    print("=" * 104)
    print(f"{label}   {start} .. {end}   slow sleeve {MA_SHORT}/{MA_LONG}, vol={VOL_WINDOW}, "
          f"top {TOP_N}, no-trade band ${MIN_TRADE:,.0f}")
    print("=" * 104)
    print(f"{'config':26}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'Calmar':>8}{'trades':>9}"
          f"{'fees':>10}{'max pos':>10}{'risk max/min':>14}")
    print("-" * 104)
    for r in rows:      # listed in protocol order A-D, NOT sorted by result
        print(f"{r['config']:26}{r['cagr']:>8.2%}{r['sharpe']:>9.2f}{r['max_dd']:>9.1%}"
              f"{r['calmar']:>8.2f}{r['trades']:>9,}{r['fees']:>10,.0f}{r['max_pos']:>9.1%}"
              f"{r['risk_ratio']:>13.2f}x")
    print("-" * 104)
    print("`risk max/min` is the mechanism check: the ratio of the largest to smallest")
    print("share of portfolio variance across holdings. Inverse-vol should compress it.")

    # The two comparisons that were pre-registered. C vs B isolates the weighting scheme
    # at matched turnover; C vs A is the adoption question against the current champion.
    A, B, C, E = (CONFIGS[0][3], CONFIGS[1][3], CONFIGS[2][3], CONFIGS[4][3])
    print(f"\nPre-registered comparisons (paired stationary block bootstrap, "
          f"4000 reps, mean block 20d):")
    for lhs, rhs, why in [(C, B, "isolates the weighting scheme at matched turnover"),
                          (E, B, "same, with the 2/N concentration cap on"),
                          (E, A, "ADOPTION TEST: does the package beat the champion?"),
                          (B, A, "cost of re-weighting itself, with the band in place")]:
        diff, se, p = paired_sharpe_test(daily[lhs], daily[rhs])
        flag = "" if abs(diff) > se else "   <- inside 1 SE, not distinguishable"
        print(f"  {lhs.strip():<24} - {rhs.strip():<24} "
              f"diff {diff:+.3f}  SE {se:.3f}  p {p:.3f}{flag}")
        print(f"      {why}")
    print()


if __name__ == "__main__":
    main()

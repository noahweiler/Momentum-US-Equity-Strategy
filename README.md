# S&P 500 Momentum Backtester

A backtester for a weekly-rebalanced momentum strategy on the S&P 500 — long-only by
default, with an optional long/short mode (see *The short leg* below).

Most of the engineering here is not in the strategy — it is in removing the ways a naive
backtest lies to you. The headline finding of the project is that the naive version of
this same strategy reported **9.0% higher CAGR** than the honest one, entirely from
survivorship bias.

```powershell
python momentum_main.py backtest --start 2022-01-01 --end 2026-06-30 --save-plot performance.png
```

---

## The strategy

| | |
|---|---|
| Universe | Point-in-time S&P 500 members, 200 trading days seasoned |
| Signal | `(SMA50 / SMA200 - 1) / 30-day volatility` |
| Gate | `signal > 0` (equivalent to MA50 > MA200) |
| Entry | Rank ≤ 30 |
| Exit | Rank > 40 (hysteresis buffer of 10) |
| Weighting | Equal, fixed at entry, never re-sized |
| Rebalance | Weekly — signal on the week's last close |
| Execution | Fill at the **next session's open** |
| Costs | $1/trade + 5 bps slippage, charged both directions |

The signal is a trend measure normalised by risk. `SMA50/SMA200 - 1` is how stretched the
trend is; dividing by 30-day volatility means a quiet utility 5% above its 200-day average
outranks a volatile biotech at the same gap.

---

## Results

Net of costs, benchmarked against **SPY total return** (not the price index — see
*Benchmark basis* below).

Periods end on half-year boundaries. Exact dates are given because the result is sensitive
to them — see the third point below.

| Period | | CAGR | Sharpe | Max DD | Calmar | Trades |
|---|---|---|---|---|---|---|
| **2012-01-01 → 2019-12-31** *(in-sample)* | Strategy | 14.80% | 1.07 | **−15.8%** | **0.94** | 2,824 |
| | SPY | 14.52% | **1.13** | −19.4% | 0.75 | |
| **2022-01-01 → 2026-06-30** *(held out)* | Strategy | **18.61%** | **1.00** | **−19.2%** | **0.97** | 1,542 |
| | SPY | 11.79% | 0.72 | −24.5% | 0.48 | |

**Read this honestly — three points.**

*In-sample the strategy essentially matches the index* on return (14.80% vs 14.52%) and is
slightly **behind** on Sharpe (1.07 vs 1.13). The return outperformance appears only in
2022-2026, a choppier regime; through the 2012-2019 bull run it merely kept pace.

*What holds in every period is drawdown.* −15.8% vs −19.4%, then −19.2% vs −24.5%, with
shorter recoveries both times. The most defensible claim is "index-like returns with
materially lower drawdown", not "higher returns".

*One month moves CAGR by 3.8 points.* Extending the end date by a single month to
2026-07-30 takes final equity from $214,952 to $187,818 — a **12.6% loss in July 2026**
against SPY's 1.6% — and the figures become 14.80% CAGR / 0.81 Sharpe. Neither end date is
more correct than the other; the half-year boundary is simply the convention used here. Any
decimal-precise figure from a backtest this length should be treated with suspicion, and the
honest out-of-sample summary is a **Sharpe in the range 0.8-1.0**.

---

## The short leg

`--short` turns the strategy long/short: the same ranking that picks the top `top_n` to
buy also picks the weakest `short_n` to sell short. A short is simply a negative share
count, so

```
payoff per share = price when shorted − price when covered
equity           = cash + Σ(shares × price)          # unchanged; shares just go negative
```

Short proceeds land in cash and the long buy phase spends them, so the book levers up:
at the default `--short-exposure 0.5` it runs **long ~150% / short ~50%, net ~100%,
gross ~200%**. Net market exposure is therefore the same as long-only; what changes is
gross.

Each weekly rebalance runs in this order, which is what lets a week's new shorts pay for
that week's covers:

1. sell longs that dropped out of the target
2. mark to market → the equity that sizes everything
3. **open** new shorts → credits cash
4. **cover** shorts that left the target → costs cash; if cash is short, trim every long
   holding by the *same fraction* to fund it
5. buy new longs with what remains, then deploy any idle cash (buys only, never trims)

Step 4's trim was needed on only **6 of 417** in-sample rebalances, largest 2.3% of the
book — the recycling in step 3 covers almost everything.

### It does not work

Run against the same two windows, with the borrow fee charged at 40 bps/yr:

| Window | Config | CAGR | Sharpe | Vol | Max DD | Calmar | Gross |
|---|---|---|---|---|---|---|---|
| **2012-2019** | long-only | 14.94% | **1.08** | 13.79% | −15.8% | 0.94 | 1.00x |
| | L/S 150/50 | **16.00%** | 0.91 | 18.18% | −15.5% | **1.03** | 2.00x |
| **2022-2026** | long-only | 18.59% | 1.00 | 18.89% | **−19.1%** | 0.97 | 0.99x |
| | L/S 150/50 | **26.41%** | 1.01 | 26.91% | −21.4% | **1.23** | 1.98x |

The CAGR looks transformative and is not. Volatility rises in exact proportion, so the
Sharpe difference is **−0.172 in-sample (paired block bootstrap SE 0.175, p = 0.29)** and
**+0.009 out-of-sample (SE 0.174, p = 0.97)** — indistinguishable from zero both times,
and negative where the point estimate is largest. The extra return is leverage, not
selection.

Pairing each short open against its cover in the trade log makes it explicit:

| | 2012-2019 | 2022-2026 |
|---|---|---|
| Closed short round trips | 665 | 500 |
| Profitable | 251 (**38%**) | 223 (**45%**) |
| Realised short P&L | **−$93,589** | −$2,155 |
| Borrow paid (40 bps) | −$3,464 | −$1,171 |
| **Net short-leg contribution** | **−$97,053** | **−$3,326** |
| Total profit | $227,388 | $186,094 |

**The short leg lost money in both windows.** In-sample it gave back $97k while the
portfolio made $227k — every dollar of outperformance came from the levered long leg,
and shorting was a tax on it. This is the expected result, not a bug: shorting the
weakest large-cap momentum names means shorting beaten-down S&P 500 constituents, which
either mean-revert or get acquired, and doing it through the 2012-2019 bull market means
a 38% hit rate.

The one genuine effect is on drawdown. Doubling gross exposure normally doubles
drawdown; here it moved from −15.8% to −15.5% in-sample. The shorts *do* hedge — but the
hedge shows up in the tail, not in the body of the distribution, which is why Calmar
improves while Sharpe does not.

**Default is off.** `allow_short=False`, and the numbers in *Results* above are all
long-only.

```powershell
# reproduce the table above
python momentum_main.py backtest --start 2012-01-01 --end 2019-12-31 --short --borrow-bps 40 --no-plot
```

**Borrow cost defaults to 0 and the run warns loudly when it is.** A real short book pays
stock loan every day it is open — 25-50 bps for general collateral, far more for anything
hard-to-borrow, which a bottom-ranked momentum screen actively selects for. A flat rate is
a floor on the true cost, not an estimate of it, so the honest reading of the table is
that the short leg is *at best* as bad as it looks.

---

## How it works

### 1. Point-in-time universe (`data.py`)

The core correctness problem. Using *today's* S&P 500 list across history is catastrophic
for a momentum strategy, because index membership is granted *after* a stock has already
run — so today's list is a filter for "had great momentum," handed to a strategy that
looks for great momentum.

The fix reconstructs who was actually in the index on each date, walking Wikipedia's
add/remove log backwards from today's constituents:

```
S_before = (S_after \ Added_d) ∪ Removed_d
```

- `load_sp500_changes()` — scrapes the change log (add-only, remove-only and swap rows all occur)
- `membership_spells()` — returns `ticker | spell_id | start | end | exit_reason | truncated`
- `load_sp500_membership()` — the same as `{ticker: [(start, end), ...]}`; a **list** because
  names leave and rejoin (16 do, e.g. AMD out 2013-09-20, back 2017-03-20)
- `membership_matrix()` — expands spells to a boolean dates × tickers frame

Intervals are **half-open** `[start, end)`. S&P changes take effect before the open, so a
closed interval double-counts both sides of every swap.

*Validation:* the reconstructed universe holds **498-505 names** across 2011-2026. Systematic
drift away from ~500 would mean the walk is dropping entries.

### 2. Prices

`load_universe_data()` downloads open and close for the **union** of everyone who was ever a
member in the window — ~790 tickers for a 2013 start, against 503 today. Chunked with
backoff; Yahoo rate-limits bursts at this size.

`load_universe_data_cached()` matches by **coverage**, not by exact key: a cached window whose
ticker set and date range *contain* the request is sliced in memory. Keying on the exact
`(tickers, start, end)` triple meant one day's change to `--end` re-downloaded all 790 names,
which is exactly the burst that gets throttled.

### 3. Panel assembly

`build_panel()` returns `(opens, closes, in_index, spells)` — wide frames on identical axes —
and writes a tidy `sp500_panel.parquet` of `date | ticker | open | close | in_index`.

**Raw prices are never masked.** `in_index` is the only representation of membership. Masking
at storage would be irreversible, would make a data gap indistinguishable from an unseasoned
stock, and would leave no price to book removal exits against.

### 4. Signal and seasoning (`Algorithms.py`)

`momentum_signal()` is computed on `closes.where(in_index)` — prices masked to in-index days.
That masking is what enforces the **200-day seasoning rule**: rolling `min_periods` means a
name needs 200 trading days of continuous membership before it produces a signal at all, and
re-entry resets the clock.

This is why `momentum_signal` deliberately does **not** call `.dropna()`. Doing so closes the
masked gaps and splices the moving average across them — silently, and in the permissive
direction (seasoning collapses from 200 days to ~50).

`rank_universe()` applies the gate and returns tickers strongest-first.

### 5. Execution (`backtest.py`, `portfolio.py`)

The weekly loop: execute the previous week's decision at today's open → mark to market at
today's close → if today is a signal date, decide for the next session.

- **Hysteresis** — `targets_for()` retains a held name anywhere in the top `top_n + exit_buffer`
  and only sells past that. The book stays at 30; the buffer widens the *exit* threshold.
- **Sizing** — new entries are funded from that week's exit proceeds, split equally, rather
  than sized off total equity (held positions keep whatever they have grown to, so sizing
  against equity would demand cash that isn't there).
- **Removal exits** — a name leaving the index is sold at the next execution date. If it has
  stopped trading entirely, it exits at the last available open and is logged.
- **Stale marks** — `Portfolio` carries the last known price forward when one is missing.
  Valuing a gap at zero is *not* conservative: Yahoo dropped 44% of tickers on 2026-07-21/22,
  which craters equity for two days and fully recovers, indistinguishable from a crash.

### 6. Diagnostics (printed on every build)

- **Universe size** — pass/fail band 495-510
- **Priceable subset** — 74-91% of the index, rising over time; this *is* the survivorship gap
- **Coverage by exit reason** — 100% for current members, 14% for companies that ceased trading
- **Known-failure check** — SIVB, SBNY, FRC, SUNE, CHK, HTZ and others, explicitly probed
- **Ticker recycling** — a delisted symbol can be reassigned (BEAM, ADT and DELL all re-IPO'd
  under their old tickers) and Yahoo will serve the new company's history under it. Flagged
  series are cut 30 days past the exit — enough to book the removal fill, not enough to splice.

---

## What each bias was worth

Every figure measured, not assumed:

| Bias | Cost | How it was removed |
|---|---|---|
| **Survivorship / index inclusion** | **9.04% CAGR** | Point-in-time membership |
| Benchmark basis | 1.5-2.2%/yr | SPY total return, not `^GSPC` price index |
| Same-bar look-ahead | 0.22% CAGR | Signal on close, fill at next open |
| Re-weighting drag | 0.5-1.5% CAGR | Size at entry, never re-size |
| Trading frictions | ~1.75%/yr | Modelled, not eliminated |

**Survivorship, concretely:** the naive version selected **TSLA on 45 separate Fridays before
it joined the index** on 2020-12-21 — capturing all of its 740% year with a screen that could
not have been looking at it. After the fix: zero.

**Benchmark basis:** `^GSPC` is price-return and excludes dividends, while the strategy's
prices are dividend-adjusted. Comparing the two flattered the strategy by 2.19%/yr over
2013-2020. `BENCHMARKS` now uses SPY/QQQ/DIA as primary, with the indices as fallback.

---

## How decisions were validated

Choices were made on **2012-2019** and checked once on **2022-2026**, with 2020-21
deliberately excluded so COVID doesn't dominate either set.

Sharpe differences are tested by **paired stationary block bootstrap** on daily returns, not
eyeballed. The standard single-Sharpe error bar (~0.4 on seven years) is roughly twice too
wide for comparing two variants run on the same dates from correlated streams; the correct
paired SE here is 0.13-0.33.

It earned its keep immediately — a rank-weighted variant placed **2nd in-sample and last
out-of-sample** (Sharpe 1.02 → 0.54).

**Rejected on evidence:**

- *Dual-timescale ensemble* (10/50 sleeve blended with 50/200). Holdings overlap was only
  0.22, but return correlation was 0.73-0.84 — different names, same bet. Lost in all 12
  capital splits; significant at the highest-power point (p = 0.0063).
- *Rank-proportional weighting* `w_i = (N+1-rank_i) / Σ(N+1-rank_j)`. Indistinguishable
  in-sample (p = 0.85) and worse out. Consistent with forward returns being statistically
  **flat from rank 21 to rank 80** — a weighting scheme can only monetise a gradient that exists.
- *Inverse-volatility weighting* `w_i ∝ 1/σ_i`. Motivated by a real inefficiency: the vol
  spread inside one top-30 selection runs a **median 3.7x (p90 6.3x)**, so equal *dollar*
  weight is nowhere near equal *risk* weight. The mechanism works exactly as intended —
  the max/min share of portfolio variance compresses from 13.2x to 3.0x in-sample and
  16.2x to 5.2x out — but it loses on both windows, and by more out-of-sample
  (−0.019, p = 0.64 in-sample; **−0.251, p = 0.087** out). It reliably lowers drawdown
  (−17.0% vs −19.8% out-of-sample) and lowers return by more, so Sharpe falls. A low-vol
  tilt that gives up too much return. Concentration was ruled out as the cause: capping at
  2/N cut the worst position from 27.2% to 7.4% and moved Sharpe by 0.00.

- *Tuning `top_n`.* Swept 10-75 against three rebalance cadences (24 cells). In-sample the
  result looked textbook — a broad, monotone plateau at n=40-75 beating n=30 in **every**
  frequency column, exactly the "plateau not peak" structure that is supposed to
  distinguish signal from noise. It **inverted out-of-sample**: rank correlation of the
  eight `top_n` marginals between the two windows is **ρ = −0.898 (p = 0.0024)**, and
  across all 24 cells ρ = −0.719 (p = 0.0001). The in-sample ranking is a near-perfect
  *inverse* predictor.

  | top_n | in-sample | held out | change |
  |---|---|---|---|
  | 10 | 0.99 | **1.17** | +0.18 |
  | 30 *(shipped)* | 1.08 | 1.00 | −0.08 |
  | 40 | **1.15** | 0.86 | **−0.29** |
  | 75 | 1.09 | 0.82 | −0.27 |

  The likely reading is regime rather than pure noise: 2012-2019 was a broad advance where
  a wider book captured more of it; 2022-2026 was narrow and mega-cap-led, where
  concentration won. Either way **there is no stable optimum**, and picking a `top_n` means
  implicitly forecasting market breadth. n=30 is kept because it sits mid-range in both
  windows — never best, never worst — not because it is optimal.

**A false positive that the replication rule caught.** Adding a **$2,000 no-trade band**
(skip small adjustments to existing holdings) appeared to *reverse* the re-weighting
result — weekly re-sizing beat sizing-at-entry by +0.042 Sharpe at **p = 0.022** in-sample,
against the −0.025 (p = 0.013) originally measured without a band. It did not replicate:
**−0.088 (p = 0.230)** out-of-sample, sign flipped. Nothing was adopted. This is the case
the "decide in-sample, confirm out-of-sample" rule exists for — p = 0.022 on its own would
have been enough to change the strategy.

**Kept on evidence:**

- *Hysteresis at b=10.* Baseline turnover was 23% "flicker" — names exiting and re-entering
  within four weeks. b=10 removes 82% of flicker notional in both windows while cutting genuine
  entries and exits only 14-17%. Justified on the cost curve (monotone, window-consistent), not
  the Sharpe curve, most of which is composition rather than friction.
- *Monthly rebalancing is available and free* (`--rebalance M`). Tested against weekly at the
  shipped config across three windows: Sharpe **−0.004 (p = 0.96)** in-sample, **+0.058
  (p = 0.58)** held out, **+0.041 (p = 0.65)** through COVID — no measurable performance
  difference — while cutting trades and costs **20-30% in every window**. The case for it is
  not alpha, it is that the same result is delivered with a fifth less trading, which also
  reduces exposure to the 5 bps slippage assumption, the model's largest unmeasured input.
  The default remains `W` only because every documented figure above was produced with it.

---

## Layout

| File | Role |
|---|---|
| `data.py` | Membership reconstruction, price download, panel assembly, caching, diagnostics |
| `Algorithms.py` | Signal computation and ranking |
| `backtest.py` | `MomentumConfig`, the weekly loop, order execution |
| `portfolio.py` | Cash, positions, fills, fees, equity history |
| `Utils.py` | Moving averages, volatility, performance metrics |
| `plotting.py` | Growth-of-100 and drawdown charts |
| `momentum_main.py` | CLI — `scan` and `backtest` |
| `scanner.py` | Live ranking of the current index (no simulation) |
| `compare_weighting.py` | In-sample / out-of-sample harness with paired bootstrap |
| `optimize_volatility.py` | Parameter sweep over `vol_window` |
| `sweep_topn_freq.py` | Book size x rebalance cadence sweep, in-sample and held out |

**Outputs (all git-ignored, rebuilt on first run):** `sp500_panel.parquet`,
`membership_spells.csv`, `coverage_report.csv`, and `.cache/` for raw downloads
(~250 MB once populated).

## Configuration

`MomentumConfig` in `backtest.py`; each field carries the measurement that justifies it.

```
top_n         30       exit_buffer   10      ma_short  50    ma_long  200
rebalance_freq "W"      (2W and M also available -- see Rebalance frequency below)
vol_window    30       weighting  equal      reweight  False
commission   1.00      slippage_bps   5       use_open_fills  True

allow_short   False    short_n       20      short_exit_buffer   10
short_exposure  0.50   borrow_bps_annual  0.0   short_gate_threshold  0.0
deploy_idle_cash  None (auto: on when shorting)   idle_cash_threshold  0.02
```

CLI overrides: `--top`, `--exit-buffer`, `--weighting {equal,rank,invvol}`, `--reweight`,
`--weight-cap`, `--min-trade`, `--close-fills`, `--cash`, `--no-plot`, `--save-plot PATH`,
`--short`, `--short-n`, `--short-exit-buffer`, `--short-exposure`, `--borrow-bps`, `--rebalance {W,2W,M}`.

`invvol` sizes each name inversely to its own volatility so holdings contribute equal
*variance* rather than equal *dollars*. Correlations are deliberately ignored — a full
covariance matrix over 30 names means 465 parameters, which is where an optimiser starts
chasing estimation error instead of risk; the diagonal needs 30 and cannot drive a name to
zero. Volatility is read from the **signal** date, never the execution date. It is
implemented and tested but **not adopted** — see *Rejected on evidence*.

`--min-trade` sets a no-trade band on adjustments to existing positions. It does not
suppress entries or exits, which are decisions rather than resizes.

`short_gate_threshold` is the mirror of `gate_threshold`: a name is shortable only when
its signal is **below** it, so at 0.0 the short leg requires MA50 < MA200 — an actual
downtrend, not merely the least strong uptrend. Taking `.tail(n)` of the long ranking
instead would short the *best of the worst* in a rising market, which is not a thesis.

`deploy_idle_cash` exists because short proceeds arrive every week while `reweight=False`
only touches the long book at entry, so that cash would otherwise sit idle and net
exposure would sag. It **buys only** — it never trims, which is the distinction that
matters, since `reweight=True` lost precisely by trimming winners to fund laggards.

`ma_long` must equal `data.SEASONING_DAYS` — the seasoning gate *is* the MA_long rolling
window, so changing one without the other silently changes the eligibility rule.
`run_backtest()` raises if they diverge.

## Known limitations

- **Membership is a good-faith reconstruction**, from Wikipedia rather than a vendor
  point-in-time database (CRSP `msp500list` would be authoritative). 21 add entries in the log
  have no matching membership state and are logged and ignored.
- **Residual survivorship bias remains.** Only 14% of companies that ceased trading are
  priceable — Yahoo deletes them. The favourable part of that split is that the missing names
  are mostly acquisitions; market-cap demotions, which drive the bias, mostly still trade.
- **Reconstruction is unreliable before ~2011.** The change log is dense from 2011 but has 10
  rows total before 2000.
- **Concentration is unbounded.** Never re-sizing lets a winner compound; max position reached
  5.3% in-sample but **19.8%** out-of-sample.
- **Integer share counts strand cash.** A high-priced name against a small budget buys one
  share and leaves the remainder idle.
- **5 bps slippage is an assumption**, and the largest single cost. Reasonable for 30 liquid
  large-caps, but it is a modelling choice, not a measurement.
- **The reported figures drift as Wikipedia updates.** Membership is reconstructed by
  walking the change log *backwards from today's constituent list*, so every real S&P
  addition or deletion shifts the reconstructed history slightly. The in-sample table was
  computed against an 808-ticker universe; a re-run in September 2026 requested 807 and
  returned 14.94% / 1.08 rather than 14.80% / 1.07. SPY is unchanged, so the difference is
  the panel, not the code — verified by running the pre-short and post-short execution
  paths against one panel and confirming the equity curves match to $0.00. A vendor
  point-in-time database would pin this down; Wikipedia cannot.
- **Short borrow is modelled as a flat annual rate, defaulting to 0.** Real borrow is
  name-specific and spikes for exactly the names a bottom-ranked screen picks, so any
  long/short result here is optimistic by an unknown margin.
- **Shorting can overdraw cash by a few dollars.** The borrow fee is charged whether or not
  the book is fully deployed, so a fully invested account can dip slightly negative
  (worst observed: −$12 on a $300k book). It is tracked and reported rather than hidden;
  there is no margin-interest model behind it.

## Requirements

Python 3.11+, `pandas`, `numpy`, `yfinance`, `matplotlib`, `pyarrow`, `lxml`.

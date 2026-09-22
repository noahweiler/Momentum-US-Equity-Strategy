"""
Historical backtester for the momentum strategy.

Point-in-time universe: a stock is selectable only while it was actually an S&P 500
member, and only after SEASONING_DAYS of continuous membership. Signals are computed
on the last close of each week and executed at the next session's open.

Long-only by default. With `allow_short` the same ranking drives a second leg: the
weakest `short_n` names are sold short, the proceeds fund a larger long book, and the
portfolio runs at gross exposure above 100%.
"""
from dataclasses import dataclass
import math
import pandas as pd

from data import (
    build_panel,
    panel_diagnostics,
    trading_calendar,
    signal_dates,
    SEASONING_DAYS,
)
from Algorithms import momentum_signal, rank_universe, rank_universe_short
from Utils import rolling_volatility, inverse_vol_weights, apply_weight_cap
from portfolio import Portfolio, Fees


@dataclass
class MomentumConfig:
    """All tuneable parameters for the momentum backtest."""
    top_n: int = 30
    exit_buffer: int = 10               # Rank hysteresis: enter at rank <= top_n, but only
                                        # SELL once a name falls past top_n + exit_buffer.
                                        # Baseline turnover was 23% "flicker" -- names exiting
                                        # and re-entering within 4 weeks, paying a round trip
                                        # each time for no change in conviction. b=10 removes
                                        # 82% of flicker notional in both windows while cutting
                                        # genuine entries/exits only 14-17%, and forward returns
                                        # are statistically flat from rank 21 to rank 80 so
                                        # nothing measurable is given up. Set 0 to disable.
    gate_threshold: float = 0.0
    ma_short: int = 50
    ma_long: int = 200                  # must equal data.SEASONING_DAYS -- see run_backtest
    vol_window: int = 30                # 30 beat 20 in both windows (+1.6 / +4.2 pp CAGR).
                                        # Specified up front in the strategy plan, not
                                        # selected by searching this data.
    rebalance_freq: str = "W"           # "W" weekly, "2W" fortnightly, "M" monthly.
                                        # Weekly is the original choice and was never
                                        # compared against anything -- a design decision,
                                        # not a measured one.
    starting_cash: float = 100_000.0
    commission_per_trade: float = 1.0
    slippage_bps: float = 5.0
    use_open_fills: bool = True         # False reverts to same-bar close fills (look-ahead)
    weighting: str = "equal"            # "equal"  -> 1/N
                                        # "rank"   -> (N+1-rank_i)/sum_j(N+1-rank_j)
                                        # "invvol" -> (1/sigma_i)/sum_j(1/sigma_j)
                                        # In-sample 2012-2019 equal and rank were
                                        # indistinguishable (Sharpe diff 0.013, paired SE
                                        # 0.065, p=0.85). Rank buys no measurable edge,
                                        # consistent with forward returns being flat from
                                        # rank 21 to rank 80, so the simpler scheme wins on
                                        # parsimony. invvol targets a different thing: equal
                                        # DOLLAR weight is not equal RISK weight, and the
                                        # vol spread inside one top-30 selection runs a
                                        # median 3.7x.
    weight_cap: float | None = None     # Max weight per name, None = uncapped. Inverse-vol
                                        # concentrates into the quietest name -- at the
                                        # observed p90 vol spread of 6.3x that could reach
                                        # ~20% of the book, which is already a flagged
                                        # limitation of never re-sizing. 2/top_n is the
                                        # natural test value.
    min_trade_notional: float = 0.0     # No-trade band, in dollars. Load-bearing whenever
                                        # reweight=True with a continuously-varying scheme:
                                        # without it every one of 30 names trades every
                                        # week and the run measures turnover cost rather
                                        # than weighting quality. 0 disables, which is what
                                        # keeps the historical results reproducible.
    reweight: bool = False              # THE ONE SIGNIFICANT RESULT: re-sizing every week
                                        # costs 0.025 Sharpe (paired p=0.013) and triples the
                                        # trade count. Re-weighting to a fixed target sells
                                        # winners and buys losers, which fights the momentum
                                        # signal. Sizing at entry and leaving positions alone
                                        # won in all three windows tested.
                                        # NOTE: rank weighting only actually holds if this is
                                        # True, so "rank" + False is not really rank weighting.

    # --- short leg -----------------------------------------------------------
    allow_short: bool = False           # Off by default: every result recorded in the README
                                        # is long-only, and flipping this changes the strategy
                                        # itself, not just a parameter of it.
    short_n: int = 20                   # how many of the WEAKEST names to short
    short_exit_buffer: int = 10         # Hysteresis mirrored onto the short leg: short a name
                                        # when it enters the worst short_n, cover it only once
                                        # it climbs out of the worst (short_n + buffer). Same
                                        # justification as the long side, and it bites harder
                                        # here -- a flicker round trip on a short pays two
                                        # commissions, two slippage hits AND the borrow.
    short_gate_threshold: float = 0.0   # Only short names whose signal is BELOW this. At 0.0
                                        # that means the 50d MA sits under the 200d MA -- an
                                        # actual downtrend, not merely the least strong uptrend.
    short_exposure: float = 0.5         # Target short notional as a fraction of equity. 0.5
                                        # gives the classic 150/50 book: short 50% of equity,
                                        # the proceeds land in cash, and the long buy phase
                                        # spends them -- so long ~150%, short ~50%, NET ~100%
                                        # (the same market exposure as long-only) and GROSS
                                        # ~200%. Raising it levers BOTH legs; it is not a
                                        # free hedge.
    borrow_bps_annual: float = 0.0      # Stock-loan fee charged daily on short notional.
                                        # 0 by default so the run matches the spec exactly,
                                        # but run_backtest warns, because a real short book
                                        # always pays this.
    deploy_idle_cash: bool | None = None
                                        # None = auto (on when shorting, off otherwise). Short
                                        # proceeds arrive as cash every week, but with
                                        # reweight=False the long book is only touched at entry,
                                        # so in a quiet week that cash would sit idle and net
                                        # exposure would sag. This puts it to work by BUYING
                                        # ONLY -- it never trims a holding, so it does not
                                        # reintroduce the momentum-fighting behaviour that made
                                        # reweight=True lose.
    idle_cash_threshold: float = 0.02   # only deploy once idle cash exceeds 2% of equity


def _target_weights(
    target_tickers: list[str],
    scheme: str,
    vols: pd.Series | None = None,
    cap: float | None = None,
    fallback_counter: list | None = None,
) -> pd.Series:
    """
    Portfolio weights for an already-ranked target list (strongest conviction first).

    rank -- w_i = (N + 1 - rank_i) / sum_j (N + 1 - rank_j)

    The denominator is the triangular number N(N+1)/2, so for N = 30 it is 465: rank 1
    takes 30/465 = 6.45% and rank 30 takes 1/465 = 0.22%, a 30:1 spread. Conviction
    scales linearly with rank rather than every name counting the same.

    equal -- 1/N, every holding the same.

    invvol -- w_i proportional to 1/sigma_i, so each name contributes about the same
    variance rather than the same dollars. Requires `vols`, which MUST be measured as of
    the SIGNAL date, not the execution date; see the assertion in run_backtest.

    Used for both legs. On the short leg position 0 is the WEAKEST name, because
    rank_universe_short returns worst first, so conviction still falls down the list.
    """
    n = len(target_tickers)
    if n == 0:
        return pd.Series(dtype=float)

    if scheme == "equal":
        raw = pd.Series(1.0, index=target_tickers)
    elif scheme == "rank":
        # the ranking functions return strongest conviction first, so position i holds
        # rank i+1 and (N + 1 - rank) is simply (n - i).
        raw = pd.Series([float(n - i) for i in range(n)], index=target_tickers)
    elif scheme == "invvol":
        if vols is None:
            raise ValueError("weighting='invvol' needs a volatility series.")
        weights, n_fallback = inverse_vol_weights(vols.reindex(target_tickers), cap=cap)
        if fallback_counter is not None and n_fallback:
            fallback_counter.append(n_fallback)
        return weights
    else:
        raise ValueError(
            f"Unknown weighting scheme {scheme!r} -- use 'equal', 'rank' or 'invvol'."
        )

    weights = raw / raw.sum()
    if cap is not None:
        weights = apply_weight_cap(weights, cap)
    return weights


def _uniform_trim_longs(
    portfolio: Portfolio,
    deficit: float,
    fill_prices: pd.Series,
    date: pd.Timestamp,
    trim_log: list,
) -> float:
    """
    Raise `deficit` dollars by selling the SAME FRACTION of every long holding.

    The funding path of last resort when a short has to be covered and the cash is not
    there. Trimming uniformly is deliberate: it leaves the long book's relative
    composition untouched, so an emergency on the short leg does not silently re-rank
    the long leg. Selling the worst longs instead would be a second, unmodelled trading
    decision hiding inside a funding operation.

    The fraction is grossed up for slippage and one commission per position, then given
    a 2% margin because integer share counts always round down. Returns cash raised.
    """
    if deficit <= 0:
        return 0.0

    priced = {t: sh for t, sh in portfolio.positions.items()
              if sh > 0 and t in fill_prices.index}
    if not priced:
        return 0.0

    book = sum(sh * float(fill_prices[t]) for t, sh in priced.items())
    if book <= 0:
        return 0.0

    slip = portfolio.fees.slippage_bps / 10_000.0
    commissions = len(priced) * portfolio.fees.commission_per_trade
    need_gross = (deficit + commissions) / max(1.0 - slip, 1e-9)
    fraction = min(need_gross / book * 1.02, 1.0)

    cash_before = portfolio.cash
    for ticker, shares in priced.items():
        qty = min(shares, int(math.ceil(fraction * shares)))
        if qty > 0:
            portfolio.sell(ticker, qty, float(fill_prices[ticker]), date=date)

    raised = portfolio.cash - cash_before
    trim_log.append({
        "date": date, "deficit": float(deficit), "fraction": float(fraction),
        "raised": float(raised), "n_positions": len(priced), "long_book": float(book),
    })
    return raised


def _open_shorts(
    portfolio: Portfolio,
    short_targets: list[str],
    fill_prices: pd.Series,
    date: pd.Timestamp,
    weighting: str,
    total_equity: float,
    short_exposure: float,
    vols: pd.Series | None = None,
    cap: float | None = None,
) -> None:
    """
    Open shorts in the target names we are not already short. Credits cash.

    Sizing mirrors the long side's "size at entry, never re-size": existing shorts are
    left exactly as they are, and only the gap between the target notional
    (`short_exposure` x equity) and the notional being KEPT is allocated across the new
    names. Positions about to be covered are excluded from that retained figure on
    purpose -- their capacity is what funds the new shorts, which is exactly the
    recycling the strategy spec describes.
    """
    target_notional = short_exposure * total_equity
    keep = set(short_targets)

    retained = 0.0
    for ticker, shares in portfolio.positions.items():
        if shares < 0 and ticker in keep and ticker in fill_prices.index:
            retained += abs(shares) * float(fill_prices[ticker])

    budget = max(target_notional - retained, 0.0)
    new_names = [t for t in short_targets
                 if t in fill_prices.index and portfolio._get_position(t) == 0]
    if budget <= 0 or not new_names:
        return

    weights = _target_weights(short_targets, weighting, vols=vols, cap=cap)
    share = weights[new_names] / weights[new_names].sum()

    for ticker in new_names:
        qty = int(budget * share[ticker] / float(fill_prices[ticker]))
        if qty > 0:
            portfolio.sell(ticker, qty, float(fill_prices[ticker]), date=date)


def _cover_shorts(
    portfolio: Portfolio,
    short_targets: list[str],
    fill_prices: pd.Series,
    date: pd.Timestamp,
    opens: pd.DataFrame,
    stranded_log: list,
    shortfall_log: list,
    trim_log: list,
) -> None:
    """
    Buy back every short that has left the target list.

    The cost is totalled first, including slippage and commission, and compared against
    cash. If cash falls short the long book is trimmed uniformly to make up the
    difference -- the order the strategy spec asks for: open new shorts, check the cash,
    cover, and downsize the longs only if still short.

    A short in a name that stopped trading is covered at the last open ever seen, the
    mirror of the long-side stranded exit. Leaving it open would carry a permanent
    negative position that mark-to-market keeps revaluing off a stale price.
    """
    keep = set(short_targets)
    leaving = [t for t, sh in portfolio.positions.items() if sh < 0 and t not in keep]
    if not leaving:
        return

    slip = portfolio.fees.slippage_bps / 10_000.0
    commission = portfolio.fees.commission_per_trade

    cover_px: dict[str, float] = {}
    for ticker in leaving:
        if ticker in fill_prices.index:
            cover_px[ticker] = float(fill_prices[ticker])
            continue
        prior = opens[ticker].loc[:date].dropna() if ticker in opens.columns else pd.Series(dtype=float)
        if prior.empty:
            stranded_log.append({"date": date, "ticker": ticker,
                                 "shares": portfolio._get_position(ticker), "price": None})
            continue
        price = float(prior.iloc[-1])
        cover_px[ticker] = price
        stranded_log.append({"date": date, "ticker": ticker,
                             "shares": portfolio._get_position(ticker), "price": price})

    if not cover_px:
        return

    cost = sum(abs(portfolio._get_position(t)) * px * (1 + slip) + commission
               for t, px in cover_px.items())
    if cost > portfolio.cash:
        _uniform_trim_longs(portfolio, cost - portfolio.cash, fill_prices, date, trim_log)

    for ticker, price in cover_px.items():
        qty = abs(portfolio._get_position(ticker))
        if qty <= 0:
            continue
        try:
            portfolio.buy(ticker, qty, price, date=date)
        except ValueError:
            # Could not raise the cash even after trimming. The short stays open into
            # next week, which is a real (small) exposure -- logged, not silent.
            shortfall_log.append({"date": date, "ticker": ticker, "shares": qty, "side": "COVER"})


def _deploy_idle_cash(
    portfolio: Portfolio,
    long_targets: list[str],
    fill_prices: pd.Series,
    date: pd.Timestamp,
    weighting: str,
    threshold_frac: float,
    total_equity: float,
    vols: pd.Series | None = None,
    cap: float | None = None,
) -> None:
    """
    Put leftover cash to work across the current long targets. BUYS ONLY.

    With shorting on, cash arrives every week from newly opened shorts. Under
    `reweight=False` the long book is only touched when a name enters, so in a week
    with no entries that cash would sit idle and net exposure would sag below target.
    Topping up existing holdings fixes that without ever selling, and that distinction
    is the whole point: `reweight=True` lost because it TRIMMED winners to fund
    laggards. Adding to a winner is not the same trade as cutting one.
    """
    if not long_targets or portfolio.cash <= threshold_frac * total_equity:
        return

    priced = [t for t in long_targets if t in fill_prices.index]
    if not priced:
        return

    weights = _target_weights(long_targets, weighting, vols=vols, cap=cap)
    share = weights[priced] / weights[priced].sum()
    reserve = len(priced) * portfolio.fees.commission_per_trade
    budget = max(portfolio.cash - reserve, 0.0)

    for ticker in priced:
        qty = int(budget * share[ticker] / float(fill_prices[ticker]))
        if qty <= 0:
            continue
        try:
            portfolio.buy(ticker, qty, float(fill_prices[ticker]), date=date)
        except ValueError:
            pass        # rounding drift on the last name; nothing material left to place


def _execute_rebalance(
    portfolio: Portfolio,
    long_targets: list[str],
    short_targets: list[str],
    fill_prices: pd.Series,
    date: pd.Timestamp,
    opens: pd.DataFrame,
    config: MomentumConfig,
    stranded_log: list,
    shortfall_log: list,
    trim_log: list,
    deploy_idle: bool,
    vols: pd.Series | None = None,
    fallback_counter: list | None = None,
) -> None:
    """
    Move the portfolio to its new targets, filling at `fill_prices`.

    Order of operations -- steps 1, 2 and 5 are the original long-only flow, 3 and 4
    are the short leg, sequenced as the strategy spec requires:

      1. SELL longs that are no longer targets      -> credits cash
      2. mark to market, giving the equity used for sizing
      3. OPEN new shorts                            -> credits cash
      4. COVER shorts that left the target          -> costs cash; if the cash is not
         there, trim the long book uniformly first
      5. BUY new longs with whatever cash remains, then deploy any idle remainder

    Opening before covering is what lets a week's new shorts pay for that week's covers,
    so the long book only has to be touched when the two do not net out.
    """
    target_set = set(long_targets)

    # --- 1. SELL phase: exit long holdings not in the new target --------------
    for ticker in list(portfolio.positions):
        shares = portfolio._get_position(ticker)
        if shares <= 0 or ticker in target_set:
            continue        # shares < 0 is a short -- handled in step 4

        if ticker in fill_prices.index:
            portfolio.sell(ticker, shares, fill_prices[ticker], date=date)
            continue

        # No price today. Either a data gap or the company stopped trading outright
        # (acquisition completed, bankruptcy). Exit at the last open we ever saw and
        # log it -- leaving the position in place would value it at zero forever.
        prior = opens[ticker].loc[:date].dropna() if ticker in opens.columns else pd.Series(dtype=float)
        if prior.empty:
            stranded_log.append({"date": date, "ticker": ticker, "shares": shares, "price": None})
            continue
        price = float(prior.iloc[-1])
        portfolio.sell(ticker, shares, price, date=date)
        stranded_log.append({"date": date, "ticker": ticker, "shares": shares, "price": price})

    # --- 2. Mark after sells so sizing uses post-sale equity ------------------
    # This writes equity_history[date], which the end-of-day close mark then
    # overwrites -- intentional.
    portfolio.mark_to_market(fill_prices, date)
    total_equity = portfolio.equity_history[date]

    # --- 3 / 4. Short leg -----------------------------------------------------
    if config.allow_short:
        if short_targets:
            _open_shorts(portfolio, short_targets, fill_prices, date, config.weighting,
                         total_equity, config.short_exposure, vols, config.weight_cap)
        _cover_shorts(portfolio, short_targets, fill_prices, date, opens,
                      stranded_log, shortfall_log, trim_log)

    if not long_targets:
        return

    # --- 5. Long leg ----------------------------------------------------------
    weights = _target_weights(long_targets, config.weighting, vols=vols,
                              cap=config.weight_cap, fallback_counter=fallback_counter)
    priced = [t for t in long_targets if t in fill_prices.index]   # skip halts / data gaps

    if config.reweight:
        desired = {t: int(total_equity * weights[t] / fill_prices[t]) for t in priced}
    else:
        # Fund new entries from what the exits just freed rather than off total equity --
        # held positions keep whatever they have grown to, so sizing against equity would
        # demand cash the portfolio does not have. Weights are renormalised over the new
        # names so their relative ranking is still respected. With shorting on this cash
        # also contains the proceeds of step 3, which is how the short leg levers up the
        # long leg.
        new_names = [t for t in priced if portfolio._get_position(t) == 0]
        if not new_names:
            if deploy_idle:
                _deploy_idle_cash(portfolio, long_targets, fill_prices, date,
                                  config.weighting, config.idle_cash_threshold, total_equity)
            return
        share = weights[new_names] / weights[new_names].sum()
        reserve = len(new_names) * portfolio.fees.commission_per_trade
        budget = max(portfolio.cash - reserve, 0.0)
        desired = {t: int(budget * share[t] / fill_prices[t]) for t in new_names}

    # --- TRIM first, so the freed cash funds the buys -------------------------
    # Under rank weighting the spread is 30:1 top to bottom, so a name sliding down the
    # ranking releases a lot of cash and a name climbing needs a lot. Interleaving buys
    # and trims in rank order would exhaust cash on the leaders before the laggards were
    # cut, and the shortfall would silently land on the highest-conviction names.
    # The no-trade band suppresses tiny ADJUSTMENTS to positions already held. It does not
    # apply to opening or closing a position -- those are decisions, not resizes, and
    # skipping them would silently change the book. Without a band, a continuously-varying
    # weighting scheme under reweight=True nudges all 30 names every week and the run ends
    # up measuring turnover cost rather than the weighting.
    band = config.min_trade_notional

    def _too_small(ticker: str, diff: int) -> bool:
        if band <= 0 or portfolio._get_position(ticker) == 0:
            return False
        return abs(diff) * float(fill_prices[ticker]) < band

    for ticker, want in desired.items():
        diff = want - portfolio._get_position(ticker)
        if diff < 0 and not _too_small(ticker, diff):
            portfolio.sell(ticker, abs(diff), fill_prices[ticker], date=date)

    # --- then BUY, walking in rank order so the strongest names fund first ----
    for ticker in long_targets:
        if ticker not in desired:
            continue
        diff = desired[ticker] - portfolio._get_position(ticker)
        if diff <= 0 or _too_small(ticker, diff):
            continue
        try:
            portfolio.buy(ticker, diff, fill_prices[ticker], date=date)
        except ValueError:
            # Not enough cash. Because we walk in rank order the names dropped are
            # systematically the lowest-ranked ones. Logged, not silent.
            shortfall_log.append({"date": date, "ticker": ticker, "shares": diff, "side": "BUY"})

    if deploy_idle:
        _deploy_idle_cash(portfolio, long_targets, fill_prices, date, config.weighting,
                          config.idle_cash_threshold, total_equity, vols, config.weight_cap)


def run_backtest(
    config: MomentumConfig,
    start_date: str,
    end_date: str,
) -> tuple[Portfolio, pd.DataFrame]:
    """
    Run the momentum backtest over a historical period.

    Returns
    -------
    portfolio : Portfolio with full equity history and trade log
    scanner_history : DataFrame logging which tickers were held at each rebalance
    """
    # The seasoning rule is enforced by min_periods on the masked close series, which
    # means it IS ma_long. If the two drift apart the rule silently changes.
    if config.ma_long != SEASONING_DAYS:
        raise ValueError(
            f"ma_long ({config.ma_long}) must equal SEASONING_DAYS ({SEASONING_DAYS}). "
            "The seasoning gate is enforced by the MA_long rolling window, so changing "
            "one without the other silently changes the eligibility rule."
        )
    if config.allow_short and config.short_exposure <= 0:
        raise ValueError("allow_short is on but short_exposure is 0 -- nothing would be shorted.")

    deploy_idle = (config.deploy_idle_cash if config.deploy_idle_cash is not None
                   else config.allow_short)

    # --- Step 1: panel --------------------------------------------------------
    opens, closes, in_index, spells = build_panel(panel_start=start_date, panel_end=end_date)
    panel_diagnostics(opens, closes, in_index, spells)

    # --- Step 2: signals on membership-masked closes --------------------------
    # Masking here rather than in the stored panel keeps raw prices available to book
    # removal exits against, and lets min_periods enforce seasoning and reset it on
    # re-entry. Raw `opens` and `closes` stay unmasked.
    print("Computing signals on membership-masked closes...")
    masked = closes.where(in_index)
    signals = momentum_signal(
        masked,
        ma_short=config.ma_short,
        ma_long=config.ma_long,
        vol_window=config.vol_window,
    )

    # Volatility panel for risk-based weighting. Built on the SAME masked closes as the
    # signal, so a membership gap yields NaN rather than an estimate spliced across it,
    # and reusing config.vol_window rather than introducing a second window keeps this
    # from adding a free parameter that could be tuned in-sample.
    vol_panel = None
    if config.weighting == "invvol":
        vol_panel = masked.apply(lambda s: rolling_volatility(s, config.vol_window))

    # --- Step 3: signal / execution calendar ----------------------------------
    calendar = trading_calendar(closes)
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    pairs = [(s, x) for s, x in signal_dates(calendar, config.rebalance_freq)
             if start <= s <= end and x <= end]
    execution_of = {x: s for s, x in pairs}

    bt_dates = calendar[(calendar >= start) & (calendar <= end)]

    fill_style = "next session open" if config.use_open_fills else "same-bar close"
    sizing = (f"{config.weighting}-weighted, re-sized weekly" if config.reweight
              else f"{config.weighting}-weighted at entry, never re-sized")
    book = (f"long {config.top_n} / short {config.short_n} at {config.short_exposure:.0%} of equity"
            if config.allow_short else f"top {config.top_n} long only")
    cadence = {"W": "weekly", "2W": "fortnightly", "M": "monthly"}[config.rebalance_freq]
    print(f"Running backtest {start.date()} to {end.date()} "
          f"({len(pairs)} {cadence} rebalances, {book}, fills at {fill_style}, "
          f"{sizing})...\n")

    # --- Step 4: portfolio ----------------------------------------------------
    fees = Fees(
        commission_per_trade=config.commission_per_trade,
        slippage_bps=config.slippage_bps,
        borrow_bps_annual=config.borrow_bps_annual,
    )
    if config.allow_short and fees.borrow_bps_annual <= 0:
        print("WARNING: shorting is on with borrow_bps_annual = 0, so the short book pays no "
              "stock-loan fee. These results are optimistic by roughly the borrow rate times "
              "short exposure per year -- order 0.1-0.3%/yr at general-collateral rates, and "
              "far more if any name goes hard-to-borrow, which a bottom-ranked momentum screen "
              "actively selects for. Pass --borrow-bps to charge it.\n")

    portfolio = Portfolio(starting_cash=config.starting_cash,
                          allow_short=config.allow_short, fees=fees)

    scanner_log: list[dict] = []
    stranded_log: list[dict] = []
    shortfall_log: list[dict] = []
    trim_log: list[dict] = []
    fallback_log: list[int] = []        # names that fell back to 1/N for want of a vol

    def _select(signal_row: pd.Series, n: int, buffer: int,
                held: set[str], short: bool) -> list[str]:
        """
        Rank one side of the book, applying the hysteresis buffer.

        A name already held is retained while it sits anywhere inside the top
        (n + buffer); only names past that are closed, and the freed slots go to the
        best-ranked names not currently held. The book stays at n either way -- the
        buffer widens the EXIT threshold, never the book size. On the short leg
        "best-ranked" means most negative signal.
        """
        rank_fn = rank_universe_short if short else rank_universe
        gate = config.short_gate_threshold if short else config.gate_threshold

        if buffer <= 0:
            return rank_fn(signal_row, n, gate)

        ranked = rank_fn(signal_row, n + buffer, gate)
        kept = [t for t in ranked if t in held][:n]         # survivors, still inside the buffer
        additions = [t for t in ranked if t not in held][:n - len(kept)]
        chosen = set(kept) | set(additions)
        return [t for t in ranked if t in chosen]           # rank order preserved

    def targets_for(signal_date: pd.Timestamp, held_long: set[str],
                    held_short: set[str]) -> tuple[list[str], list[str]]:
        """Long and short target lists as of `signal_date`. Non-members are already NaN."""
        row = signals.loc[signal_date]
        longs = _select(row, config.top_n, config.exit_buffer, held_long, short=False)
        if not config.allow_short:
            return longs, []

        shorts = _select(row, config.short_n, config.short_exit_buffer, held_short, short=True)
        # The opposite gates make an overlap impossible in normal conditions (a signal
        # cannot be both > 0 and < 0), but the two hysteresis buffers are independent, so
        # this guard is cheap insurance against ever holding a name long and short at
        # once -- which a single integer share count cannot represent.
        long_set = set(longs)
        shorts = [t for t in shorts if t not in long_set]
        return longs, shorts

    # --- Step 5: main loop ----------------------------------------------------
    for date in bt_dates:
        # Holdings as they stand before this rebalance -- the hysteresis rule needs to know
        # what is already owned to decide what survives.
        held_long = {t for t, shares in portfolio.positions.items() if shares > 0}
        held_short = {t for t, shares in portfolio.positions.items() if shares < 0}

        # 1. Execute the decision made on the previous week's close, at today's open.
        if config.use_open_fills and date in execution_of:
            signal_date = execution_of[date]
            longs, shorts = targets_for(signal_date, held_long, held_short)
            # Weights are DECIDED on the signal close and APPLIED at the next open, so the
            # volatility they read must be the signal date's. Taking `date` here instead
            # would price the weights off a bar the decision could not have seen -- the
            # same class of look-ahead already fixed once in this codebase, and one that
            # raises nothing on its own.
            vols = None
            if vol_panel is not None:
                assert signal_date < date, (
                    f"signal date {signal_date} must precede execution date {date}"
                )
                vols = vol_panel.loc[signal_date]
            _execute_rebalance(portfolio, longs, shorts, opens.loc[date].dropna(), date,
                               opens, config, stranded_log, shortfall_log, trim_log,
                               deploy_idle, vols, fallback_log)
            scanner_log.append({"signal_date": signal_date, "execution_date": date,
                                "n_holdings": len(longs), "tickers": longs,
                                "n_shorts": len(shorts), "shorts": shorts})

        # Same-bar close fills, kept only for A/B comparison against the biased version.
        elif not config.use_open_fills and date in {s for s, _ in pairs}:
            longs, shorts = targets_for(date, held_long, held_short)
            vols = vol_panel.loc[date] if vol_panel is not None else None
            _execute_rebalance(portfolio, longs, shorts, closes.loc[date].dropna(), date,
                               opens, config, stranded_log, shortfall_log, trim_log,
                               deploy_idle, vols, fallback_log)
            scanner_log.append({"signal_date": date, "execution_date": date,
                                "n_holdings": len(longs), "tickers": longs,
                                "n_shorts": len(shorts), "shorts": shorts})

        # 2. Charge the day's borrow on the short book, then value at today's close.
        day_closes = closes.loc[date].dropna()
        if config.allow_short:
            portfolio.accrue_borrow_cost(day_closes, date)
        portfolio.mark_to_market(day_closes, date)

    # --- Step 6: report the things that would otherwise be silent -------------
    if stranded_log:
        print(f"Closed {len(stranded_log)} position(s) at a stale open (ticker stopped trading):")
        for row in stranded_log[:10]:
            price = f"${row['price']:.2f}" if row["price"] is not None else "NO PRICE - stranded"
            print(f"  {row['date'].date()}  {row['ticker']:<6} {row['shares']:>6} sh @ {price}")
        if len(stranded_log) > 10:
            print(f"  ... and {len(stranded_log) - 10} more")

    if shortfall_log:
        n_cover = sum(1 for r in shortfall_log if r.get("side") == "COVER")
        print(f"Skipped {len(shortfall_log)} order(s) for insufficient cash: {n_cover} cover(s) "
              f"left open into the next week, the rest buys (systematically the lowest-ranked "
              f"targets).")

    if trim_log:
        raised = sum(r["raised"] for r in trim_log)
        worst = max(r["fraction"] for r in trim_log)
        print(f"Trimmed the long book uniformly on {len(trim_log)} of {len(pairs)} rebalances to "
              f"fund short covers (${raised:,.0f} raised, largest single trim {worst:.1%} of "
              f"the book).")

    if fallback_log:
        print(f"Inverse-vol weighting fell back to 1/N on {sum(fallback_log)} name-week(s) "
              f"across {len(fallback_log)} of {len(pairs)} rebalances (no usable volatility "
              f"estimate). Reported rather than hidden -- if this is not rare, the weights "
              f"are closer to equal than the label suggests.")

    if portfolio.borrow_paid > 0:
        print(f"Stock-loan fees paid: ${portfolio.borrow_paid:,.2f}")

    if portfolio.margin_debit_days:
        print(f"Cash ran negative on {portfolio.margin_debit_days} day(s), worst "
              f"${portfolio.max_margin_debit:,.2f} -- the borrow fee is charged whether or "
              f"not the book is fully deployed, so it can overdraw a fully invested "
              f"account. Immaterial at this scale; reported so it can never be silent.")

    return portfolio, pd.DataFrame(scanner_log)

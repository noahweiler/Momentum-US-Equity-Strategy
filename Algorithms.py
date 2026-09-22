import pandas as pd
import numpy as np
from Utils import SMA, rolling_volatility


def momentum_signal(
    prices: pd.DataFrame,
    ma_short: int = 50,
    ma_long: int = 200,
    vol_window: int = 50,
) -> pd.DataFrame:
    """
    Compute the volatility-normalised MA ratio for every ticker on every date.

    Signal = (MA_short / MA_long - 1) / volatility

    Parameters
    ----------
    prices : DataFrame with DatetimeIndex rows and ticker columns,
             containing daily close prices.
    ma_short : int, default 50
        Short-term moving average window
    ma_long : int, default 200
        Long-term moving average window
    vol_window : int, default 20
        Volatility calculation window

    Returns
    -------
    DataFrame of same shape containing signal values.
    NaN where there is insufficient history or zero volatility.
    """
    signals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for ticker in prices.columns:
        # NaNs are retained deliberately. When `prices` has been masked to in-index days,
        # the gaps sit inside the rolling windows and fail min_periods, which is what
        # enforces the seasoning rule and resets it on re-entry. Calling .dropna() here
        # would close those gaps and splice the MA across them -- silently, and in the
        # permissive direction (seasoning collapses from 200 days to ~50).
        px = prices[ticker]
        if px.notna().sum() < ma_long:
            continue  # insufficient history for long MA

        ma_s = SMA(px, ma_short)
        ma_l = SMA(px, ma_long)
        vol = rolling_volatility(px, vol_window)

        raw = (ma_s / ma_l - 1.0) / vol
        raw = raw.replace([np.inf, -np.inf], np.nan)

        signals[ticker] = raw

    return signals


def rank_universe(
    signals_row: pd.Series,
    top_n: int = 30,
    gate_threshold: float = 0.0,
) -> list[str]:
    """
    Given a single row (one date) of signal values across tickers,
    apply the gate (signal > threshold) and return the top N tickers
    sorted by signal descending.
    """
    valid = signals_row.dropna()
    gated = valid[valid > gate_threshold]
    ranked = gated.sort_values(ascending=False)
    return ranked.head(top_n).index.tolist()


def rank_universe_short(
    signals_row: pd.Series,
    bottom_n: int = 20,
    gate_threshold: float = 0.0,
) -> list[str]:
    """
    Mirror of rank_universe for the SHORT leg: the weakest names, worst first.

    The gate is flipped -- a name is only shortable if its signal is BELOW the
    threshold, so at the default 0.0 we short only stocks whose 50d MA sits under
    their 200d MA. This matters: `.tail(top_n)` of the long ranking would return the
    least-bad names in a strong market, which is not a short thesis, it is just the
    bottom of a list of positives.

    Returns weakest first, so position 0 is the highest-conviction short. That
    ordering is what `_target_weights` reads, so rank weighting puts the most money
    on the worst name on both sides of the book.
    """
    valid = signals_row.dropna()
    gated = valid[valid < gate_threshold]
    ranked = gated.sort_values(ascending=True)
    return ranked.head(bottom_n).index.tolist()

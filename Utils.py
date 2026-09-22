
import numpy as np
import pandas as pd


def normalize_data_to_100(data: pd.Series) -> pd.Series: 
    """ Normalize data to start at 0 and evolve from there, inputs are panda series """
    
    normalised_data_to_100 = (data / data.iloc[0])*100
    return normalised_data_to_100
    

def calculate_returns(data: pd.Series) -> pd.Series:
    """ Calculate the returns of a price series, inputs are panda series """
    
    returns = data.pct_change().fillna(0)        #pct_change calculates the percentage change between the current and a prior element, fillna replaces any NaN values with 0
    return returns



def SMA(data: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA) for a given window size."""
    sma = data.rolling(window=window).mean()
    return sma


def rolling_volatility(data: pd.Series, window: int = 30) -> pd.Series:
    """Rolling standard deviation of daily returns."""
    # fill_method=None so gaps stay gaps. On a membership-masked series the default
    # forward-fill would invent 0% returns across the masked stretch and understate vol.
    daily_returns = data.pct_change(fill_method=None)
    return daily_returns.rolling(window=window).std()


def apply_weight_cap(weights: pd.Series, cap: float) -> pd.Series:
    """
    Enforce `w_i <= cap` while keeping the weights summing to 1.

    Excess above the cap is redistributed proportionally across the names still under it,
    which can push a second name over, so it iterates to convergence. If the cap is so
    tight that every name is capped (cap * n <= 1) the constraint is infeasible and the
    only answer is equal weights, which is what gets returned.
    """
    n = len(weights)
    if n == 0 or cap is None or cap <= 0:
        return weights
    if cap * n <= 1.0 + 1e-12:
        # Infeasible or exactly saturated -- every name sits at the cap.
        return pd.Series(1.0 / n, index=weights.index)

    w = weights.astype(float).copy()
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        room = w[under]
        if room.sum() <= 0:
            break
        w[under] = room + excess * room / room.sum()

    return w / w.sum()


def inverse_vol_weights(
    vols: pd.Series,
    cap: float | None = None,
) -> tuple[pd.Series, int]:
    """
    Risk-balanced weights: `w_i = (1/sigma_i) / sum_j (1/sigma_j)`.

    Equal DOLLAR weight is not equal RISK weight. Measured on this universe, the spread
    between the most and least volatile name inside a single top-30 selection runs a
    median 3.7x (p90 6.3x), so under 1/N the most volatile holding contributes roughly
    four times the portfolio variance of the least volatile one for the same money. This
    sizes each name inversely to its own volatility so those contributions equalise.

    Correlations are deliberately ignored. Using them would mean estimating a full
    covariance matrix -- 30 variances plus 435 covariances -- from a few hundred days of
    history, which is the setup where an optimiser starts chasing estimation error rather
    than risk. The diagonal needs only n parameters and cannot drive a name to zero.

    Any name whose sigma is missing, zero, negative or non-finite falls back to the equal
    weight 1/n rather than being dropped or given infinite weight.

    Returns
    -------
    weights : pd.Series indexed like `vols`, summing to 1
    n_fallback : int, how many names used the 1/n fallback -- reported, never silent
    """
    n = len(vols)
    if n == 0:
        return pd.Series(dtype=float), 0

    sigma = pd.to_numeric(vols, errors="coerce").astype(float)
    usable = sigma.notna() & np.isfinite(sigma) & (sigma > 0)
    n_fallback = int((~usable).sum())

    raw = pd.Series(1.0 / n, index=vols.index, dtype=float)   # fallback value for the rest
    if usable.any():
        inv = 1.0 / sigma[usable]
        # Scale the usable block so it occupies its share of the book, leaving the
        # fallback names on a straight 1/n each.
        share = float(usable.sum()) / n
        raw[usable] = inv / inv.sum() * share

    weights = raw / raw.sum()
    if cap is not None:
        weights = apply_weight_cap(weights, cap)
    return weights, n_fallback


def performance_metrics(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute standard backtest performance metrics from an equity curve.
    Returns dict with: total_return, cagr, sharpe, sortino,
    max_drawdown, max_drawdown_duration_days, annualized_volatility, calmar.
    """
    returns = equity.pct_change().dropna()
    total_days = (equity.index[-1] - equity.index[0]).days
    years = total_days / 365.25

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (1 + total_return) ** (1 / years) - 1.0 if years > 0 else 0.0

    excess = returns - risk_free_rate / periods_per_year
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (excess.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else 0.0

    downside_std = returns[returns < 0].std()
    sortino = (excess.mean() / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0

    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()

    # Max drawdown duration (in trading days)
    is_underwater = drawdown < 0
    groups = (~is_underwater).cumsum()
    if is_underwater.any():
        dd_durations = is_underwater.groupby(groups).sum()
        max_dd_duration = int(dd_durations.max())
    else:
        max_dd_duration = 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_duration_days": max_dd_duration,
        "calmar": calmar,
    }
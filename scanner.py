"""
Live momentum scanner.
Run weekly to see current top N S&P 500 stocks by risk-adjusted momentum.

Usage:
    python scanner.py                  # default top 30
    python scanner.py --top 50         # top 50
    python scanner.py --output csv     # save to CSV
"""
import argparse
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from data import get_sp500_tickers, load_universe_data
from Algorithms import momentum_signal, rank_universe


def run_scanner(top_n: int = 30) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Fetch current S&P 500 prices, compute signals, return ranked table.

    Returns
    -------
    result : DataFrame with columns: ticker, name, signal, close, ma_50, ma_200, vol_20d
    latest_date : Timestamp of the most recent trading day in the data
    """
    tickers = get_sp500_tickers()

    # Need at least 200 trading days of history for MA_200
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")

    print(f"Downloading data for {len(tickers)} tickers...")
    _, prices = load_universe_data(tickers, start, end)      # (opens, closes) -- scanner needs closes
    print(f"Got data for {len(prices.columns)} tickers.")

    signals = momentum_signal(prices)

    latest_date = signals.index[-1]
    latest_signals = signals.loc[latest_date]

    top_tickers = rank_universe(latest_signals, top_n=top_n)

    # Fetch company names
    print("Fetching company names...")
    ticker_names = {}
    for t in top_tickers:
        try:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            # Try longName first, fallback to shortName, then ticker
            name = info.get('longName') or info.get('shortName') or t
            # Truncate long names to 30 chars
            ticker_names[t] = name[:30] if len(name) > 30 else name
        except:
            ticker_names[t] = t  # fallback to ticker if fetch fails

    rows = []
    for t in top_tickers:
        px = prices[t]
        rows.append({
            "ticker": t,
            "name": ticker_names[t],
            "signal": round(latest_signals[t], 4),
            "close": round(px.iloc[-1], 2),
            "ma_50": round(px.rolling(50).mean().iloc[-1], 2),
            "ma_200": round(px.rolling(200).mean().iloc[-1], 2),
            "vol_20d": round(px.pct_change().rolling(20).std().iloc[-1], 4),
        })

    result = pd.DataFrame(rows)
    result.index = range(1, len(result) + 1)
    result.index.name = "rank"
    return result, latest_date


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Momentum Scanner -- S&P 500")
    parser.add_argument("--top", type=int, default=30, help="Number of top stocks to show")
    parser.add_argument("--output", choices=["console", "csv"], default="console")
    args = parser.parse_args()

    result, as_of = run_scanner(top_n=args.top)

    print(f"\nMomentum Scanner -- {as_of.strftime('%Y-%m-%d')}")
    print("=" * 110)
    print(result.to_string())
    print("=" * 110)

    if args.output == "csv":
        fname = f"scan_{as_of.strftime('%Y%m%d')}.csv"
        result.to_csv(fname)
        print(f"\nSaved to {fname}")

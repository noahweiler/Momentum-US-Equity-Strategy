"""
Volatility Window Optimization

Tests different volatility window lengths to find the optimal parameter.
Runs backtests with various vol_window values and compares performance.

Usage:
    python optimize_volatility.py --start 2020-01-01 --end 2025-12-31
    python optimize_volatility.py --start 2020-01-01 --end 2025-12-31 --metric sharpe
"""
import argparse
import pandas as pd
from backtest import run_backtest, MomentumConfig
from Utils import performance_metrics


def optimize_volatility_window(
    start_date: str,
    end_date: str,
    vol_windows: list[int] = None,
    top_n: int = 30,
    optimization_metric: str = "sharpe",
) -> pd.DataFrame:
    """
    Run backtests with different volatility windows and compare results.

    Parameters
    ----------
    start_date : str
        Backtest start date (YYYY-MM-DD)
    end_date : str
        Backtest end date (YYYY-MM-DD)
    vol_windows : list[int], optional
        List of volatility windows to test. Default: [10, 20, 30, 40, 50, 60, 90, 120]
    top_n : int, default 30
        Number of stocks to hold
    optimization_metric : str, default "sharpe"
        Metric to optimize: sharpe, cagr, sortino, calmar

    Returns
    -------
    DataFrame with columns: vol_window, total_return, cagr, sharpe, sortino,
                            max_drawdown, max_dd_duration, calmar
    """
    if vol_windows is None:
        vol_windows = [10, 20, 30, 40, 50, 60, 90, 120]

    results = []

    print(f"\n{'='*70}")
    print(f"VOLATILITY WINDOW OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Top N holdings: {top_n}")
    print(f"Testing {len(vol_windows)} different volatility windows: {vol_windows}")
    print(f"Optimization metric: {optimization_metric.upper()}")
    print(f"{'='*70}\n")

    for i, vol_window in enumerate(vol_windows, 1):
        print(f"[{i}/{len(vol_windows)}] Testing vol_window = {vol_window}...", end=" ", flush=True)

        config = MomentumConfig(
            top_n=top_n,
            vol_window=vol_window,
        )

        portfolio, _ = run_backtest(config, start_date, end_date)
        equity = portfolio.equity_series()
        metrics = performance_metrics(equity)

        results.append({
            "vol_window": vol_window,
            "total_return": metrics["total_return"],
            "cagr": metrics["cagr"],
            "sharpe": metrics["sharpe"],
            "sortino": metrics["sortino"],
            "max_drawdown": metrics["max_drawdown"],
            "max_dd_duration": metrics["max_drawdown_duration_days"],
            "ann_volatility": metrics["annualized_volatility"],
            "calmar": metrics["calmar"],
            "total_trades": len(portfolio.trades),
        })

        print(f"CAGR: {metrics['cagr']:.2%}, Sharpe: {metrics['sharpe']:.2f}")

    results_df = pd.DataFrame(results)
    return results_df


def display_results(results_df: pd.DataFrame, metric: str = "sharpe"):
    """Display optimization results in a formatted table."""
    print(f"\n{'='*90}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*90}")
    print(f"{'Vol Win':>8} {'Total Ret':>10} {'CAGR':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Max DD':>9} {'DD Days':>9} {'Calmar':>8} {'Trades':>8}")
    print("-" * 90)

    for _, row in results_df.iterrows():
        print(f"{row['vol_window']:>8.0f} "
              f"{row['total_return']:>10.2%} "
              f"{row['cagr']:>8.2%} "
              f"{row['sharpe']:>8.2f} "
              f"{row['sortino']:>8.2f} "
              f"{row['max_drawdown']:>9.2%} "
              f"{row['max_dd_duration']:>9.0f} "
              f"{row['calmar']:>8.2f} "
              f"{row['total_trades']:>8.0f}")

    print("-" * 90)

    # Highlight best result
    best_idx = results_df[metric].idxmax()
    best_row = results_df.loc[best_idx]

    print(f"\n*** OPTIMAL: vol_window = {int(best_row['vol_window'])} ***")
    print(f"    {metric.upper()}: {best_row[metric]:.4f}")
    print(f"    CAGR: {best_row['cagr']:.2%}")
    print(f"    Max Drawdown: {best_row['max_drawdown']:.2%}")
    print(f"    Total Trades: {int(best_row['total_trades'])}")
    print(f"{'='*90}\n")

    return best_row


def main():
    parser = argparse.ArgumentParser(
        description="Optimize volatility window parameter for momentum strategy"
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--top", type=int, default=30, help="Number of top stocks to hold")
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=None,
        help="Volatility windows to test (e.g., --windows 10 20 30 50)",
    )
    parser.add_argument(
        "--metric",
        choices=["sharpe", "cagr", "sortino", "calmar"],
        default="sharpe",
        help="Metric to optimize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to CSV file (e.g., optimization_results.csv)",
    )

    args = parser.parse_args()

    results_df = optimize_volatility_window(
        start_date=args.start,
        end_date=args.end,
        vol_windows=args.windows,
        top_n=args.top,
        optimization_metric=args.metric,
    )

    best_row = display_results(results_df, metric=args.metric)

    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

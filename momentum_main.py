"""
Momentum Strategy -- Scanner and Backtester

Usage:
    python momentum_main.py scan                                        # live scan, top 30
    python momentum_main.py scan --top 50                               # live scan, top 50
    python momentum_main.py scan --top 30 --output csv                  # save to CSV
    python momentum_main.py backtest --start 2020-01-01 --end 2025-12-31
    python momentum_main.py backtest --start 2020-01-01 --end 2025-12-31 --top 20

    # long/short: long the top 20, short the weakest 20, borrow charged at 40 bps
    python momentum_main.py backtest --start 2022-01-01 --end 2026-06-30 \
        --top 20 --short --short-n 20 --borrow-bps 40
"""
import argparse


def _date_arg(value: str) -> str:
    """argparse type= that rejects impossible dates up front rather than at download time."""
    from data import parse_date
    try:
        parse_date(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return value


def main():
    parser = argparse.ArgumentParser(description="Momentum Scanner & Backtester")
    subparsers = parser.add_subparsers(dest="command")

    # --- scan subcommand ---
    scan_parser = subparsers.add_parser("scan", help="Run live momentum scanner")
    scan_parser.add_argument("--top", type=int, default=30, help="Number of top stocks")
    scan_parser.add_argument("--output", choices=["console", "csv"], default="console")

    # --- backtest subcommand ---
    bt_parser = subparsers.add_parser("backtest", help="Run historical backtest")
    bt_parser.add_argument("--start", required=True, type=_date_arg, help="Start date YYYY-MM-DD")
    bt_parser.add_argument("--end", required=True, type=_date_arg, help="End date YYYY-MM-DD")
    bt_parser.add_argument("--top", type=int, default=30, help="Number of top stocks to hold")
    bt_parser.add_argument("--exit-buffer", type=int, default=10,
                           help="Rank hysteresis: hold until rank > top+buffer. 0 disables.")
    bt_parser.add_argument("--rebalance", choices=["W", "2W", "M"], default="W",
                           help="Rebalance cadence: W weekly (default), 2W fortnightly, "
                                "M monthly. Monthly is Sharpe-neutral across three windows "
                                "and cuts trades and costs by 20-30%%.")
    bt_parser.add_argument("--cash", type=float, default=100_000, help="Starting capital")
    bt_parser.add_argument("--weighting", choices=["equal", "rank", "invvol"], default="equal",
                           help="equal: 1/N (default). rank: w_i=(N+1-rank_i)/sum(N+1-rank_j). "
                                "invvol: w_i proportional to 1/sigma_i -- equal risk rather "
                                "than equal dollars. invvol only really holds with --reweight.")
    bt_parser.add_argument("--weight-cap", type=float, default=None, metavar="FRAC",
                           help="Max weight per name, e.g. 0.067 for 2/30. Inverse-vol "
                                "concentrates into the quietest name without one.")
    bt_parser.add_argument("--min-trade", type=float, default=0.0, metavar="USD",
                           help="No-trade band: skip adjustments to existing positions below "
                                "this notional. Needed with --weighting invvol --reweight, "
                                "else all 30 names trade every week.")
    bt_parser.add_argument("--reweight", action="store_true",
                           help="Re-size every holding to its target weight each week. Off by "
                                "default: it costs 0.025 Sharpe (p=0.013) and triples turnover.")
    bt_parser.add_argument("--close-fills", action="store_true",
                           help="Fill at the signal bar's close instead of the next open. "
                                "Same-bar look-ahead -- for A/B comparison only.")
    bt_parser.add_argument("--short", action="store_true",
                           help="Long/short: also sell short the weakest names. Changes the "
                                "strategy, not just a parameter -- every README result is "
                                "long-only.")
    bt_parser.add_argument("--short-n", type=int, default=20,
                           help="How many of the weakest names to short (default 20).")
    bt_parser.add_argument("--short-exit-buffer", type=int, default=10,
                           help="Rank hysteresis on the short leg: cover only once a name "
                                "climbs out of the worst short-n+buffer. 0 disables.")
    bt_parser.add_argument("--short-exposure", type=float, default=0.5,
                           help="Short notional as a fraction of equity (default 0.5 = a "
                                "150/50 book: net ~100%%, gross ~200%%).")
    bt_parser.add_argument("--borrow-bps", type=float, default=0.0,
                           help="Annualised stock-loan fee charged daily on short notional. "
                                "0 by default, which flatters the short leg -- 40 is a "
                                "reasonable general-collateral estimate.")
    bt_parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    bt_parser.add_argument("--save-plot", metavar="PATH",
                           help="Write the chart to a PNG instead of opening a window")

    args = parser.parse_args()

    if args.command == "scan":
        from scanner import run_scanner

        result, as_of = run_scanner(top_n=args.top)
        print(f"\nMomentum Scanner -- {as_of.strftime('%Y-%m-%d')}")
        print("=" * 70)
        print(result.to_string())

        if args.output == "csv":
            fname = f"scan_{as_of.strftime('%Y%m%d')}.csv"
            result.to_csv(fname)
            print(f"\nSaved to {fname}")

    elif args.command == "backtest":
        from backtest import run_backtest, MomentumConfig
        from Utils import performance_metrics
        from plotting import performance_figure
        from data import load_benchmarks
        import matplotlib
        if args.save_plot:
            matplotlib.use("Agg")           # headless: no window, just the file
        import matplotlib.pyplot as plt

        config = MomentumConfig(
            top_n=args.top,
            exit_buffer=args.exit_buffer,
            rebalance_freq=args.rebalance,
            starting_cash=args.cash,
            use_open_fills=not args.close_fills,
            weighting=args.weighting,
            reweight=args.reweight,
            weight_cap=args.weight_cap,
            min_trade_notional=args.min_trade,
            allow_short=args.short,
            short_n=args.short_n,
            short_exit_buffer=args.short_exit_buffer,
            short_exposure=args.short_exposure,
            borrow_bps_annual=args.borrow_bps,
        )
        portfolio, scan_hist = run_backtest(config, args.start, args.end)
        equity = portfolio.equity_series()
        label = (f"Momentum L{args.top}/S{args.short_n}" if args.short
                 else f"Momentum Top-{args.top}")
        equity.name = label

        metrics = performance_metrics(equity)

        # Load S&P 500 for comparison
        benchmarks = load_benchmarks(args.start, args.end)
        sp500_prices = benchmarks["S&P 500"]
        # Normalize to starting capital for comparison
        sp500_equity = sp500_prices / sp500_prices.iloc[0] * args.cash
        sp500_metrics = performance_metrics(sp500_equity)

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period:              {args.start} to {args.end}")
        if args.short:
            print(f"Holdings:            Long top {args.top} / short bottom {args.short_n}, "
                  f"{args.weighting}-weighted, "
                  f"{'re-sized weekly' if args.reweight else 'sized at entry'}")
            print(f"Short sizing:        {args.short_exposure:.0%} of equity, proceeds fund "
                  f"the long leg")
            print(f"Borrow cost:         {args.borrow_bps:.0f} bps/yr"
                  f"{'  (NOT CHARGED -- results flattered)' if args.borrow_bps <= 0 else ''}")
        else:
            print(f"Holdings:            Top {args.top}, {args.weighting}-weighted, "
                  f"{'re-sized weekly' if args.reweight else 'sized at entry'}")
        print(f"Signal:              SMA{config.ma_short}/SMA{config.ma_long} "
              f"/ {config.vol_window}d volatility")
        exit_rule = (f"hold until rank > {args.top + args.exit_buffer} "
                     f"(enter at <= {args.top})" if args.exit_buffer
                     else f"exit on rank > {args.top}")
        print(f"Exit rule:           {exit_rule}")
        if args.short:
            cover_rule = (f"cover once out of the worst {args.short_n + args.short_exit_buffer} "
                          f"(short at worst {args.short_n})" if args.short_exit_buffer
                          else f"cover once out of the worst {args.short_n}")
            print(f"Cover rule:          {cover_rule}")
        print(f"Universe:            Point-in-time S&P 500 members, 200d seasoned")
        cadence = {"W": "Weekly", "2W": "Fortnightly", "M": "Monthly"}[args.rebalance]
        print(f"Rebalance:           {cadence}, signal on the period's last close")
        print(f"Fills:               {'Same-bar close (LOOK-AHEAD)' if args.close_fills else 'Next session open'}")
        print(f"Starting capital:    ${args.cash:,.0f}")
        print()
        print(f"{'Metric':<25} {'Strategy':>15} {'S&P 500':>15}")
        print("-" * 60)
        print(f"{'Final equity':<25} ${equity.iloc[-1]:>14,.2f} ${sp500_equity.iloc[-1]:>14,.2f}")
        print(f"{'Total return':<25} {metrics['total_return']:>14.2%} {sp500_metrics['total_return']:>14.2%}")
        print(f"{'CAGR':<25} {metrics['cagr']:>14.2%} {sp500_metrics['cagr']:>14.2%}")
        print(f"{'Sharpe ratio':<25} {metrics['sharpe']:>14.2f} {sp500_metrics['sharpe']:>14.2f}")
        print(f"{'Sortino ratio':<25} {metrics['sortino']:>14.2f} {sp500_metrics['sortino']:>14.2f}")
        print(f"{'Max drawdown':<25} {metrics['max_drawdown']:>14.2%} {sp500_metrics['max_drawdown']:>14.2%}")
        print(f"{'Max DD duration':<25} {metrics['max_drawdown_duration_days']:>11} days {sp500_metrics['max_drawdown_duration_days']:>11} days")
        print(f"{'Ann. volatility':<25} {metrics['annualized_volatility']:>14.2%} {sp500_metrics['annualized_volatility']:>14.2%}")
        print(f"{'Calmar ratio':<25} {metrics['calmar']:>14.2f} {sp500_metrics['calmar']:>14.2f}")
        print("-" * 60)
        print(f"{'Excess return vs S&P':<25} {(metrics['total_return'] - sp500_metrics['total_return']):>14.2%}")
        print(f"{'Excess CAGR vs S&P':<25} {(metrics['cagr'] - sp500_metrics['cagr']):>14.2%}")
        print()
        print(f"Total trades:        {len(portfolio.trades)}")
        if args.short:
            expo = portfolio.exposure_frame()
            sides = portfolio.trades_frame()["side"].value_counts()
            print(f"  of which shorts:   {int(sides.get('SHORT', 0))} opened, "
                  f"{int(sides.get('COVER', 0))} covered")
            print(f"Avg gross exposure:  {expo['gross_exposure'].mean():.2f}x  "
                  f"(max {expo['gross_exposure'].max():.2f}x)")
            print(f"Avg net exposure:    {expo['net_exposure'].mean():.2f}x  "
                  f"(min {expo['net_exposure'].min():.2f}x)")
            print(f"Borrow fees paid:    ${portfolio.borrow_paid:,.2f}")
        print("=" * 60)

        if not args.no_plot or args.save_plot:
            performance_figure(
                equity, benchmarks,
                title=f"{label} vs Benchmarks  |  {args.start} to {args.end}",
                save_path=args.save_plot,
            )
            if not args.save_plot:
                plt.show()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

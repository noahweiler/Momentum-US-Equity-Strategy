# portfolio.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd


@dataclass
class Fees:
    """Transaction cost settings."""
    commission_per_trade: float = 0.0     # flat fee charged once per order (e.g., $1.00)
    slippage_bps: float = 0.0             # 5 bps = 0.0005; applied to price (up for buys, down for sells), its essentially the small change on what you actually buy and sell for, your order slightly changes the prices and there is price movements during the execution so the slippage is the difference between the price you intended to buy and the actual price you bought at
    borrow_bps_annual: float = 0.0        # stock-loan fee on SHORT notional, accrued daily.
                                          # Default 0 so long-only results are untouched, but a
                                          # real short book pays this every day it is open --
                                          # run_backtest warns when shorting runs at 0.


class Portfolio:
    """
    Minimal portfolio that tracks cash, positions (shares), and equity.
    Provides buy() / sell() methods with fees and slippage, and mark_to_market()
    to store daily equity values for later plotting.

    A SHORT is simply a negative share count. Selling stock you do not own credits
    cash immediately; the position then carries negative market value, so

        payoff per share = price_when_shorted - price_when_covered

    falls straight out of `equity = cash + sum(shares * price)` with no special case.
    What shorting changes is that `cash` can exceed equity, because part of it is the
    proceeds of borrowed stock that still has to be bought back.
    """

    def __init__(self, starting_cash: float = 100_000.0, allow_short: bool = False, fees: Fees = Fees()) -> None:
        self.cash: float = float(starting_cash)     # Sets value to starting_cash if it isnt entered
        self.positions: Dict[str, int] = {}         # ticker -> shares owned; NEGATIVE = short
        self.allow_short = allow_short              # allows negative shares
        self.fees = fees                            # sets fixed transaction fees

        # Logs
        self.trades: list[dict] = []                                # each item: {date, ticker, side, qty, fill, commission, cash_after, shares_after}
        self.equity_history: dict[pd.Timestamp, float] = {}         # date -> equity
        self.last_price: Dict[str, float] = {}                      # ticker -> most recent price seen
        self.stale_marks: list[dict] = []                           # days a holding was valued off a stale price
        self.exposure_history: dict[pd.Timestamp, dict] = {}        # date -> {cash, long_value, short_value, equity}
        self.borrow_paid: float = 0.0                               # cumulative stock-loan fees
        self.max_margin_debit: float = 0.0                          # worst negative cash balance seen
        self.margin_debit_days: int = 0                             # days spent with cash < 0


    # ---------- helpers ----------
    def _apply_slippage(self, price: float, side: str) -> float:
        """Adjust price by slippage bps (buy worse / sell worse)."""
        if self.fees.slippage_bps <= 0:
            return float(price)
        adj = self.fees.slippage_bps / 10_000.0     # 10 000 bps is 100%

        if side.lower() == "buy":
            return float(price * (1 + adj))         # here we are buying slightly higher because of slippage
        else:
            return float(price * (1 - adj))         # here we are selling slightly lower because of slippage



    def _get_position(self, ticker: str) -> int:
        return int(self.positions.get(ticker, 0))               # the 0 here is the default value the dictionary outputs if that key doesn't exist, so it doesnt error out


    def _set_position(self, ticker: str, new_shares: int) -> None:
        if new_shares == 0:
            self.positions.pop(ticker, None)                # If we are initially adding no shares then we remove that ticker from the dictionray
        else:
            self.positions[ticker] = int(new_shares)



    # ---------- public API ----------
    def buy(self, ticker: str, qty: int, price: float, date: Optional[pd.Timestamp] = None) -> None:
        """
        Buy `qty` shares of `ticker` at `price` (will apply slippage & commission).
        If the position is currently short this is a COVER -- it still costs cash, which
        is why the rebalance has to raise the money before calling it.
        Raises if not enough cash.
        """
        if qty <= 0:
            return

        current = self._get_position(ticker)
        if current < 0 and qty > -current:
            # Crossing short -> long in one order would make the trade log ambiguous
            # (part cover, part new long at a single fill). Callers split it in two.
            raise ValueError(
                f"Order would cross {ticker} from short ({current}) to long in one trade. "
                "Cover and open separately."
            )

        fill = self._apply_slippage(price, "buy")                               # Buying with slippage
        commission = self.fees.commission_per_trade if qty != 0 else 0.0        # using the fees class here that is why we have two .
        cost = qty * fill + commission

        if self.cash < cost:
            raise ValueError(f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}")

        self.cash -= cost
        new_shares = current + qty
        self._set_position(ticker, new_shares)

        self.trades.append({                                                    # Adding new trade to portfolio
            "date": pd.Timestamp(date) if date is not None else pd.NaT,
            "ticker": ticker,
            "side": "COVER" if current < 0 else "BUY",
            "qty": int(qty),
            "fill": float(fill),
            "commission": float(commission),
            "cash_after": float(self.cash),
            "shares_after": int(new_shares),
        })



    def sell(self, ticker: str, qty: int, price: float, date: Optional[pd.Timestamp] = None) -> None:
        """
        Sell `qty` shares of `ticker` at `price` (will apply slippage & commission).
        If shorting is not allowed, cannot sell more than current shares.
        With shorting allowed and no position, this OPENS a short: cash is credited now
        and the share count goes negative.
        """
        if qty <= 0:
            return

        current = self._get_position(ticker)
        if not self.allow_short and qty > current:
            raise ValueError(f"Cannot sell {qty} shares; only {current} available and shorting disabled.")
        if self.allow_short and current > 0 and qty > current:
            # Same reasoning as buy(): never flip sides inside one order.
            raise ValueError(
                f"Order would cross {ticker} from long ({current}) to short in one trade. "
                "Close and open separately."
            )

        fill = self._apply_slippage(price, "sell")
        commission = self.fees.commission_per_trade if qty != 0 else 0.0
        proceeds = qty * fill - commission

        self.cash += proceeds
        new_shares = current - qty
        self._set_position(ticker, new_shares)

        self.trades.append({
            "date": pd.Timestamp(date) if date is not None else pd.NaT,
            "ticker": ticker,
            "side": "SELL" if current > 0 else "SHORT",
            "qty": int(qty),
            "fill": float(fill),
            "commission": float(commission),
            "cash_after": float(self.cash),
            "shares_after": int(new_shares),
        })


    def accrue_borrow_cost(self, prices: Dict[str, float] | pd.Series, date: pd.Timestamp) -> float:
        """
        Charge one day of stock-loan fee on the short book and return the amount.

        Borrowing stock is not free. Large-cap S&P names are "general collateral" and
        cheap (roughly 25-50 bps a year), but a heavily shorted name gets expensive --
        and a bottom-ranked momentum screen selects for exactly the names that do. A
        flat rate is a floor on the true cost, not an estimate of it.
        """
        rate = self.fees.borrow_bps_annual
        if rate <= 0:
            return 0.0

        get_px = lambda t: float(prices.get(t, float("nan")))
        short_notional = 0.0
        for ticker, sh in self.positions.items():
            if sh >= 0:
                continue
            px = get_px(ticker)
            if pd.isna(px):
                px = self.last_price.get(ticker, float("nan"))
            if pd.isna(px):
                continue
            short_notional += abs(sh) * px

        fee = short_notional * (rate / 10_000.0) / 252.0
        self.cash -= fee
        self.borrow_paid += fee

        # buy() refuses to overdraw, but this fee is not optional -- the lender takes it
        # whether or not the book is fully invested. So cash CAN go slightly negative
        # here, which is a real margin debit, and the run must not hide it. Observed
        # worst case on the 2012-2019 book is about -$12 on a $300k account, i.e. a
        # rounding artefact of being fully deployed, not leverage. It is tracked so that
        # a genuinely large debit could never pass unnoticed.
        if self.cash < 0:
            self.margin_debit_days += 1
            self.max_margin_debit = min(self.max_margin_debit, self.cash)

        return float(fee)


    def mark_to_market(self, prices: Dict[str, float] | pd.Series, date: pd.Timestamp) -> float:
        """
        Revalue the portfolio using current prices (dict or Series of last prices).
        Stores equity at `date` and returns it.

        Short positions carry negative share counts, so `sh * px` is negative and the
        sum is net book value. The two legs are also recorded separately in
        `exposure_history` so gross and net exposure can be audited afterwards.
        """
        # dict and Series both support .get(key, default)
        get_px = lambda t: float(prices.get(t, float("nan")))

        # Remember every price we see, so a gap can be carried forward rather than
        # valued at zero.
        for ticker in self.positions:
            px = get_px(ticker)
            if not pd.isna(px):
                self.last_price[ticker] = px

        long_value = 0.0
        short_value = 0.0
        for ticker, sh in self.positions.items():
            px = get_px(ticker)
            if pd.isna(px):
                # Carry the last known price forward. Valuing at 0 was the old behaviour and
                # it is NOT conservative -- a vendor data gap (Yahoo dropped ~44% of tickers
                # on 2026-07-21/22) craters equity for a day and then fully recovers, which
                # is indistinguishable from a crash and poisons max-drawdown. A stale price
                # is wrong by a day's return; a zero is wrong by the whole position. For a
                # SHORT a zero is worse still -- it books the maximum possible profit.
                px = self.last_price.get(ticker, float("nan"))
                if pd.isna(px):
                    px = 0.0            # never had a price at all -- nothing better available
                else:
                    self.stale_marks.append({"date": pd.Timestamp(date), "ticker": ticker})
            if sh >= 0:
                long_value += sh * px
            else:
                short_value += sh * px          # negative

        equity = self.cash + long_value + short_value
        self.equity_history[pd.Timestamp(date)] = float(equity)         # Creates a historical (date indexed price record of the portfolio)
        self.exposure_history[pd.Timestamp(date)] = {
            "cash": float(self.cash),
            "long_value": float(long_value),
            "short_value": float(short_value),      # negative
            "equity": float(equity),
        }
        return float(equity)


    # ---------- convenience getters ----------
    def equity_series(self) -> pd.Series:
        """Return equity history as a pd.Series sorted by date."""
        if not self.equity_history:
            return pd.Series(dtype=float)               # If no valuation history, ie records of the price then return an empty panda series so program doesnt crash
        s = pd.Series(self.equity_history)              # converts dictionary with timestamp as key and price as value into a panda series
        return s.sort_index()                           # returns sorted equity_series by date, in ascending order


    def trades_frame(self) -> pd.DataFrame:
        """Return the trade log as a DataFrame."""
        if not self.trades:
            return pd.DataFrame(columns=["date","ticker","side","qty","fill","commission","cash_after","shares_after"]).set_index("date")
        df = pd.DataFrame(self.trades)
        return df.set_index("date").sort_index()                # sets data frame index to data and then sorts the dataframe by date in ascending order

    def positions_snapshot(self) -> pd.Series:
        """Return current positions as a Series (ticker -> shares). Negative = short."""
        if not self.positions:
            return pd.Series(dtype=int)
        return pd.Series(self.positions).astype(int).sort_index()       # converts dictionary (self.positions) into a panda series, with the index being the ticker and the number of shares displayed as an integer. This is then sorted alphabetically by ticker

    def long_positions(self) -> Dict[str, int]:
        """Ticker -> shares for the long leg only."""
        return {t: sh for t, sh in self.positions.items() if sh > 0}

    def short_positions(self) -> Dict[str, int]:
        """Ticker -> shares (negative) for the short leg only."""
        return {t: sh for t, sh in self.positions.items() if sh < 0}

    def exposure_frame(self) -> pd.DataFrame:
        """
        Daily cash / long / short / equity, plus gross and net exposure as a fraction
        of equity: gross = (long + |short|) / equity, net = (long - |short|) / equity.
        A long-only book sits at gross = net = ~1.0.
        """
        if not self.exposure_history:
            return pd.DataFrame(columns=["cash", "long_value", "short_value", "equity",
                                         "gross_exposure", "net_exposure"])
        df = pd.DataFrame(self.exposure_history).T.sort_index()
        eq = df["equity"].replace(0.0, float("nan"))
        df["gross_exposure"] = (df["long_value"] - df["short_value"]) / eq
        df["net_exposure"] = (df["long_value"] + df["short_value"]) / eq
        return df

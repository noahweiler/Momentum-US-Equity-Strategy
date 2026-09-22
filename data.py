### Loading data and printing

import os
import time
import json
import glob
import hashlib
import yfinance as yf
import pandas as pd


# ETFs are the PRIMARY benchmark, indices only the fallback -- this ordering matters.
# Under auto_adjust=True an ETF series is TOTAL RETURN (dividends reinvested), which is the
# same basis as the strategy's own prices. The bare indices (^GSPC, ^DJI) are PRICE RETURN
# and exclude dividends, so benchmarking against them understated the market by 2.19%/yr
# over 2013-2020 and 1.51%/yr over 2022-2026 -- flattering the strategy by that much.
# SPY/QQQ/DIA also carry their expense ratios inside their prices, so the comparison is
# net-of-costs on both sides, and they are what you would actually buy instead.
# QQQ is paired with ^NDX, not ^IXIC: QQQ tracks the Nasdaq-100 while ^IXIC is the ~3000-name
# Composite, so the old pairing swapped in a materially different index whenever it fired.
BENCHMARKS = {
    "S&P 500": {"symbol": "SPY", "fallback_etf": "^GSPC"},
    "Nasdaq 100": {"symbol": "QQQ", "fallback_etf": "^NDX"},
    "Dow Jones": {"symbol": "DIA", "fallback_etf": "^DJI"},
}


# --- Wikipedia sources -------------------------------------------------------
# Note the %26 encoding for the ampersand in both URLs.
SP500_LIST_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"                # current constituents
SP500_CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"  # historical add/remove log
WIKI_HEADERS = {                                            # without a User-Agent Wikipedia returns 403
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# --- Panel parameters --------------------------------------------------------
PANEL_START = "2013-01-01"      # first backtest date; reconstruction is validated back to 2011
WARMUP_DAYS = 400               # calendar days downloaded before PANEL_START so MA_200 is ready on day one
SEASONING_DAYS = 200            # trading days of continuous membership before a name becomes eligible.
                                # Must equal MomentumConfig.ma_long -- the rolling min_periods on the masked
                                # close series is what enforces this, so the two cannot drift apart.
                                # Asserted in backtest.run_backtest().


def load_price_data(ticker:str, start_date:str, end_date:str) -> pd.DataFrame:            #: are type hints for input and outputs, they don't restrict input

    # Downloading historical data
    data = yf.download(ticker, start_date, end_date, auto_adjust=True, progress=False)      #auto_adjust shows the data adjusted for when there were stock splits or dividends, where there are sharp price movements that aren't indicative of the actual stock price, progress shows you yf's progress for pulling data which isn't necessary to see

    if data.empty: 
        raise ValueError('No data returned. Check ticker or dates')

    # yfinance returns 2-level columns like ('Close','AAPL') even for a single ticker.
    # Drop the ticker level so columns are plain names and data['Close'] is a Series.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(-1)      #-1 is the last level, which holds the ticker

    if "Close" not in data.columns:
    # Some CSVs name it differently (e.g., 'Adj Close'). Prefer 'Adj Close' if present.
        if "Adj Close" in data.columns:
            data["Close"] = data["Adj Close"]

        else:
            raise ValueError("Data must include a 'Close' or 'Adj Close' column." )

    if "Open" not in data.columns:
        raise ValueError("Data must include an 'Open' column.")

    data = data.sort_index()                                    #sorts data frame by its index, its index type is DatetimeIndex so data is sorted in chronological order
    # Drop rows with missing close prices, these are excluded from returned dataframe
    data = data.dropna(subset=["Open", "Close"])                #this drops any row with missing values in either the open or close portions. It applies to every row and drops each row that has NaN
    return data[["Open", "Close"]]                              #two-column frame, DatetimeIndex sorted chronologically: Open for fills, Close for valuation



def load_benchmarks(start:str, end:str, use_etf_fallback: bool = True) -> dict[str, pd.Series]:
    '''Loading benchmark close price series as the value of a dict with name of index as the key, 
    the value is a panda series object '''

    out = {}
    for name, meta in BENCHMARKS.items():           #.items returns (key, value) tuples that allow you to iterate through the dictionary, it is equivalent to enumerate but for dictionaries
            sym = meta['symbol']
            try:
                 out[name] = load_price_data(sym, start, end)['Close']       #load_price_data returns an Open/Close frame; benchmarks only need the close series

            except Exception:
                 if use_etf_fallback:
                      etf = meta['fallback_etf']
                      out[name] = load_price_data(etf, start, end)['Close']
    return out



# def clean_for_plot(s: pd.Series) -> pd.Series:
#     # If you have a MultiIndex like ('AAPL', Timestamp(...)) drop extra levels
#     if isinstance(s.index, pd.MultiIndex):
#         # try to keep the last level (often the datetime level)
#         s = s.droplevel(list(range(s.index.nlevels - 1)))

#     # make sure it's datetime type
#     s.index = pd.to_datetime(s.index)

#     # strip intraday time -> just midnight of each date
#     s.index = s.index.normalize()

#     # drop missing values
#     s = s.dropna()

#     return s


def _normalise_ticker(value) -> str | None:
    """
    yfinance uses "-" where Wikipedia uses "." (e.g. BRK.B -> BRK-B).
    Returns None for blank cells so add-only / remove-only rows can be detected.
    """
    if pd.isna(value):
        return None
    ticker = str(value).strip().replace(".", "-")
    return ticker or None


def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    table = pd.read_html(SP500_LIST_URL, storage_options=WIKI_HEADERS)[0]
    tickers = table["Symbol"].map(_normalise_ticker).dropna().tolist()
    return sorted(tickers)


def _find_column(columns, *parts: str):
    """Locate a flattened column by the words it contains, rather than by position."""
    for col in columns:
        lowered = str(col).lower()
        if all(part.lower() in lowered for part in parts):
            return col
    raise ValueError(f"No column matching {parts} in {list(columns)}")


def load_sp500_changes() -> pd.DataFrame:
    """
    Scrape the historical S&P 500 add/remove log.

    Returns date | added | removed | reason, newest first.

    Rows are NOT paired. Three shapes occur and all are preserved:
      - add + remove  (a straight swap)
      - add only      (spin-off, e.g. HONA with nothing removed)
      - remove only   (no replacement that day, e.g. CAG)
    Treating every row as a swap silently drops entries and the universe count drifts.
    """
    table = pd.read_html(SP500_CHANGES_URL, storage_options=WIKI_HEADERS)[0]

    # The header is two-level ('Added','Ticker') etc -- flatten it before addressing columns.
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = ["_".join(str(level) for level in col) for col in table.columns]

    cols = table.columns
    changes = pd.DataFrame({
        "date":    pd.to_datetime(table[_find_column(cols, "effective")], errors="coerce"),
        "added":   table[_find_column(cols, "added", "ticker")].map(_normalise_ticker),
        "removed": table[_find_column(cols, "removed", "ticker")].map(_normalise_ticker),
        "reason":  table[_find_column(cols, "reason")].astype(str),
    })

    changes = changes.dropna(subset=["date"])
    return changes.sort_values("date", ascending=False).reset_index(drop=True)


def membership_spells(panel_start: str | None = None) -> pd.DataFrame:
    """
    Reconstruct point-in-time index membership by walking the change log backwards
    from today's constituent set:

        S_before = (S_after \\ Added_d) u Removed_d

    Returns ticker | spell_id | start | end | exit_reason | truncated

    `end` is NaT while a spell is still open. `truncated` marks spells that began
    before `panel_start` and were clipped to it -- those stocks were already
    long-standing members, so they must not start the seasoning clock. The warm-up
    window (WARMUP_DAYS before PANEL_START) is what gives them their 200 in-index
    observations before the backtest begins.
    """
    floor = pd.Timestamp(panel_start) if panel_start is not None else None
    changes = load_sp500_changes()

    members = set(get_sp500_tickers())          # state as of today, walked backwards
    spell_end = {t: pd.NaT for t in members}    # NaT == still a member
    spell_reason: dict[str, str] = {}
    rows: list[dict] = []
    orphan_adds = 0

    for change in changes.itertuples(index=False):
        # An "added" row means the ticker was NOT a member immediately before this date,
        # so the spell we have been tracking starts here and is now complete.
        if change.added:
            if change.added in members:
                rows.append({
                    "ticker": change.added,
                    "start": change.date,
                    "end": spell_end.get(change.added, pd.NaT),
                    "exit_reason": spell_reason.pop(change.added, None),
                })
                members.discard(change.added)
                spell_end.pop(change.added, None)
            else:
                orphan_adds += 1        # add with no matching later state; log-level noise

        # A "removed" row means the ticker WAS a member immediately before this date.
        if change.removed:
            spell_end[change.removed] = change.date
            spell_reason[change.removed] = change.reason
            members.add(change.removed)

    # Whatever is still a member after the full walk was already in the index before
    # the log begins -- start is unknown, so it is left as NaT and marked truncated.
    for ticker in sorted(members):
        rows.append({
            "ticker": ticker,
            "start": pd.NaT,
            "end": spell_end.get(ticker, pd.NaT),
            "exit_reason": spell_reason.get(ticker),
        })

    spells = pd.DataFrame(rows)
    if orphan_adds:
        print(f"  note: {orphan_adds} add entries had no matching membership state (ignored)")

    # Mark and clip spells that predate the panel.
    truncated = spells["start"].isna()
    if floor is not None:
        truncated = truncated | (spells["start"] < floor)
        spells["start"] = spells["start"].fillna(floor).clip(lower=floor)
        # Drop spells that ended before the panel opens -- they never overlap it.
        spells = spells[spells["end"].isna() | (spells["end"] >= floor)]
    spells["truncated"] = truncated.reindex(spells.index)

    spells = spells.sort_values(["ticker", "start"]).reset_index(drop=True)
    spells["spell_id"] = spells.groupby("ticker").cumcount()
    return spells[["ticker", "spell_id", "start", "end", "exit_reason", "truncated"]]


def load_sp500_membership(panel_start: str | None = None) -> dict[str, list[tuple]]:
    """
    Point-in-time membership as {ticker: [(start, end), ...]}.

    A list of spells rather than a single tuple, because stocks do leave and rejoin.
    `end` is None while the ticker is still a member.
    """
    spells = membership_spells(panel_start)
    out: dict[str, list[tuple]] = {}
    for spell in spells.itertuples(index=False):
        end = None if pd.isna(spell.end) else spell.end
        out.setdefault(spell.ticker, []).append((spell.start, end))
    return out


def membership_matrix(
    spells: pd.DataFrame,
    index: pd.DatetimeIndex,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Expand membership spells into a boolean frame (dates x tickers).

    Built once so the backtest loop can do a cheap row lookup instead of interval
    arithmetic at every rebalance.
    """
    if tickers is None:
        tickers = sorted(spells["ticker"].unique())

    matrix = pd.DataFrame(False, index=index, columns=tickers)
    known = set(matrix.columns)

    for spell in spells.itertuples(index=False):
        if spell.ticker not in known:
            continue                                    # no price data for this name
        # Half-open interval [start, end). S&P changes take effect before the open on the
        # effective date, so an added name is a member that day and a removed name is not.
        # Using a closed interval would double-count both sides of a swap.
        mask = index >= spell.start
        if pd.notna(spell.end):
            mask &= index < spell.end
        matrix.loc[mask, spell.ticker] = True

    return matrix


def load_universe_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    chunk_size: int = 50,
    pause: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Batch-download open and close prices for many tickers.

    Returns (opens, closes), each a DataFrame with DatetimeIndex rows and ticker
    columns. Downloads in chunks with a pause between them -- Yahoo rate limits
    aggressively on a universe this size.

    Tickers that return nothing at all are dropped. That is the graceful path for
    acquired companies Yahoo has deleted; the coverage report measures how many.
    """
    open_frames, close_frames = [], []
    n_chunks = (len(tickers) + chunk_size - 1) // chunk_size
    failed_chunks = 0

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        chunk_no = i // chunk_size + 1

        # Yahoo throttles bursts on a universe this size, and yfinance surfaces that as
        # whole-chunk failures ("no timezone found", "day is out of range for month")
        # rather than as an exception. Back off and retry rather than losing the chunk.
        raw = pd.DataFrame()
        for attempt in range(3):
            try:
                raw = yf.download(
                    chunk,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
            except Exception as exc:
                print(f"  chunk {chunk_no}/{n_chunks}: {type(exc).__name__} on attempt {attempt + 1}")
                raw = pd.DataFrame()

            if not raw.empty:
                break
            if attempt < 2:
                backoff = 5 * (3 ** attempt)          # 5s, 15s
                print(f"  chunk {chunk_no}/{n_chunks}: empty, retrying in {backoff}s "
                      f"(attempt {attempt + 2}/3)")
                time.sleep(backoff)

        if raw.empty:
            failed_chunks += 1
            print(f"  chunk {chunk_no}/{n_chunks}: FAILED after 3 attempts, skipped")
            continue

        # Multiple tickers -> MultiIndex columns (field, ticker); selecting the field
        # leaves the tickers as columns, which is the shape we want.
        if isinstance(raw.columns, pd.MultiIndex):
            open_frames.append(raw["Open"])
            close_frames.append(raw["Close"])
        else:
            open_frames.append(raw[["Open"]].rename(columns={"Open": chunk[0]}))
            close_frames.append(raw[["Close"]].rename(columns={"Close": chunk[0]}))

        if pause and i + chunk_size < len(tickers):
            time.sleep(pause)

    if failed_chunks:
        print(f"  {failed_chunks} of {n_chunks} chunks failed after retries.")

    if not close_frames:
        raise RuntimeError(
            f"All {n_chunks} download chunks failed. This is almost always Yahoo rate "
            "limiting rather than bad tickers -- individual downloads will still work. "
            "Wait a few minutes and re-run; nothing was cached, so no state is corrupted."
        )

    opens = pd.concat(open_frames, axis=1).sort_index().dropna(axis=1, how="all")
    closes = pd.concat(close_frames, axis=1).sort_index().dropna(axis=1, how="all")

    # Keep both frames on identical axes so they can be indexed interchangeably.
    shared = sorted(set(opens.columns) & set(closes.columns))
    return opens[shared], closes[shared]


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def _cache_key(tickers: list[str], start: str, end: str) -> str:
    content = f"{','.join(sorted(tickers))}|{start}|{end}"
    return hashlib.md5(content.encode()).hexdigest()


def _find_covering_cache(tickers: list[str], start: str, end: str) -> str | None:
    """
    Find a cached download whose ticker set and date window both CONTAIN this request.

    Keying strictly on (tickers, start, end) meant one day's change to --end missed the
    cache and re-downloaded all ~800 names, which is exactly the burst Yahoo throttles.
    A wider cached window already holds the answer, so slice it instead.
    """
    want = set(tickers)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    candidates = []

    for meta_path in glob.glob(os.path.join(CACHE_DIR, "ohlc_*.json")):
        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue

        if pd.Timestamp(meta["start"]) > start_ts or pd.Timestamp(meta["end"]) < end_ts:
            continue
        if not want.issubset(set(meta["requested"])):
            continue

        pkl = meta_path[:-5] + ".pkl"
        if os.path.exists(pkl):
            # Prefer the tightest covering window, so slices stay small.
            span = (pd.Timestamp(meta["end"]) - pd.Timestamp(meta["start"])).days
            candidates.append((span, pkl))

    return min(candidates)[1] if candidates else None


def load_universe_data_cached(
    tickers: list[str], start_date: str, end_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load (opens, closes), reusing any cached download that covers the request.

    Re-running the pipeline must not re-hit Yahoo: adjusted prices are restated as
    corporate actions accumulate, so a re-download is not guaranteed to reproduce
    earlier numbers.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    covering = _find_covering_cache(tickers, start_date, end_date)
    if covering:
        cached = pd.read_pickle(covering)
        opens, closes = cached["opens"], cached["closes"]
        cols = [t for t in tickers if t in closes.columns]
        opens = opens.loc[start_date:end_date, cols]
        closes = closes.loc[start_date:end_date, cols]
        print(f"Loading cached price data ({os.path.basename(covering)[:17]}, "
              f"sliced to {start_date}..{end_date})...")
        return opens, closes

    key = _cache_key(tickers, start_date, end_date)
    cache_path = os.path.join(CACHE_DIR, f"ohlc_{key}.pkl")
    opens, closes = load_universe_data(tickers, start_date, end_date)

    # Never cache a failed download. The old code pickled empty frames, so one flaky
    # network moment poisoned that key permanently and every later run returned nothing.
    if not closes.empty:
        pd.to_pickle({"opens": opens, "closes": closes}, cache_path)
        # Sidecar metadata so coverage can be tested without unpickling 25MB. `requested`
        # is the full ticker list asked for, not the columns returned -- otherwise names
        # Yahoo has deleted would look "missing" and defeat the cache every time.
        with open(cache_path[:-4] + ".json", "w") as fh:
            json.dump({"start": start_date, "end": end_date,
                       "requested": sorted(tickers)}, fh)
    else:
        print("  warning: download returned no data -- not cached, will retry next run")

    return opens, closes


# ---------------------------------------------------------------- trading calendar

def trading_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Trading days as the union of dates actually observed across the universe.

    Deliberately not pd.bdate_range -- that counts market holidays as sessions, which
    would put signal dates on days with no prices.
    """
    return pd.DatetimeIndex(prices.index).sort_values().unique()


REBALANCE_FREQS = ("W", "2W", "M")


def signal_dates(
    calendar: pd.DatetimeIndex,
    freq: str = "W",
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Pair each period's last trading day with the next session.

    Signal is computed on the period's final close (Thursday when Friday is a holiday)
    and executed at the next session's open (Tuesday when Monday is a holiday).

    freq -- "W" weekly, "2W" every second week, "M" monthly. The period boundary is
    always taken from the OBSERVED calendar, never from bdate_range, so a holiday can
    never produce a signal date on which nothing traded.

    Every pair is asserted to be strictly increasing: an off-by-one here is look-ahead
    bias and would not raise on its own.
    """
    if freq not in REBALANCE_FREQS:
        raise ValueError(f"freq must be one of {REBALANCE_FREQS}, got {freq!r}")

    calendar = pd.DatetimeIndex(calendar).sort_values()

    if freq == "M":
        frame = pd.DataFrame({"date": calendar,
                              "a": calendar.year, "b": calendar.month})
    else:
        iso = calendar.isocalendar()
        frame = pd.DataFrame({"date": calendar,
                              "a": iso["year"].values, "b": iso["week"].values})

    period_ends = frame.groupby(["a", "b"])["date"].max().sort_values()
    if freq == "2W":
        # Every second week end. Anchored at the first observed week, so the phase is a
        # property of the data window rather than of the calendar year.
        period_ends = period_ends.iloc[::2]

    positions = calendar.get_indexer(pd.DatetimeIndex(period_ends))
    pairs = [
        (calendar[p], calendar[p + 1])
        for p in positions
        if p >= 0 and p + 1 < len(calendar)
    ]

    for signal_date, execution_date in pairs:
        assert signal_date < execution_date, f"execution {execution_date} not after signal {signal_date}"

    return pairs


def weekly_signal_dates(calendar: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Weekly signal/execution pairs. Thin wrapper kept so existing callers still work."""
    return signal_dates(calendar, freq="W")


# ---------------------------------------------------------------- panel assembly

PANEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp500_panel.parquet")
SPELLS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "membership_spells.csv")
COVERAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coverage_report.csv")


def parse_date(value: str, label: str = "date") -> pd.Timestamp:
    """
    Parse a YYYY-MM-DD date, failing loudly on impossible ones.

    An invalid calendar date (2026-06-31 -- June has 30 days) otherwise flows straight
    through to yfinance, which fails EVERY download chunk with
    "ValueError: day is out of range for month". That looks exactly like rate limiting
    and sends you hunting in the wrong place.
    """
    try:
        return pd.Timestamp(value)
    except ValueError as exc:
        month_lengths = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
                         7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        hint = ""
        parts = str(value).split("-")
        if len(parts) == 3 and parts[1].isdigit() and int(parts[1]) in month_lengths:
            month = int(parts[1])
            name = pd.Timestamp(2001, month, 1).strftime("%B")
            hint = f" {name} has at most {month_lengths[month]} days."
        raise ValueError(f"Invalid {label} {value!r}: {exc}.{hint}") from None


def build_panel(
    panel_start: str = PANEL_START,
    panel_end: str | None = None,
    write_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assemble the point-in-time panel.

    Returns wide frames (opens, closes, in_index) on identical axes, plus the spell
    frame the diagnostics need.

    Raw prices are never masked. `in_index` is the only representation of membership,
    which keeps a data gap distinguishable from an unseasoned stock and leaves a price
    available to book removal exits against after membership ends.

    Downloads start WARMUP_DAYS before `panel_start` so names already in the index on
    day one arrive with their 200 in-index observations already accumulated -- that is
    what stops truncated spells from starting the seasoning clock.
    """
    start_ts = parse_date(panel_start, "start date")
    end_ts = parse_date(panel_end, "end date") if panel_end else pd.Timestamp.today()
    if end_ts <= start_ts:
        raise ValueError(f"End date {end_ts.date()} must be after start date {start_ts.date()}.")

    panel_end = end_ts.strftime("%Y-%m-%d")
    download_start = (start_ts - pd.DateOffset(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

    print(f"Reconstructing membership from {download_start}...")
    spells = membership_spells(download_start)
    tickers = sorted(spells["ticker"].unique())
    print(f"  {len(spells)} spells across {len(tickers)} tickers "
          f"({int(spells['truncated'].sum())} truncated at the warm-up boundary)")

    print(f"Downloading OHLC for {len(tickers)} tickers ({download_start} to {panel_end})...")
    opens, closes = load_universe_data_cached(tickers, download_start, panel_end)
    if closes.empty:
        raise RuntimeError("No price data downloaded -- cannot build panel.")
    print(f"  got data for {len(closes.columns)} of {len(tickers)} tickers")

    calendar = trading_calendar(closes)
    opens = opens.reindex(index=calendar, columns=closes.columns)

    # Cut recycled symbols before anything downstream can rank on spliced prices.
    recycled = _truncate_recycled(opens, closes, spells)
    if recycled:
        total = sum(r["obs_dropped"] for r in recycled)
        print(f"  truncated {len(recycled)} recycled ticker(s), {total:,} observations dropped "
              f"({', '.join(r['ticker'] for r in recycled[:8])}"
              f"{'...' if len(recycled) > 8 else ''})")
        closes = closes.dropna(axis=1, how="all")
        opens = opens[closes.columns]

    in_index = membership_matrix(spells, calendar, list(closes.columns))

    if write_outputs:
        _write_panel_outputs(opens, closes, in_index, spells, tickers)

    return opens, closes, in_index, spells


def _write_panel_outputs(opens, closes, in_index, spells, requested_tickers) -> None:
    """Write the tidy panel, the spell frame and the coverage report."""
    tidy = pd.DataFrame({
        "open": opens.stack(future_stack=True),
        "close": closes.stack(future_stack=True),
        "in_index": in_index.stack(future_stack=True),
    })
    tidy.index.names = ["date", "ticker"]
    tidy = tidy.reset_index()
    tidy = tidy[tidy["open"].notna() | tidy["close"].notna()]     # drop all-empty rows
    tidy.to_parquet(PANEL_PATH, index=False)

    spells.to_csv(SPELLS_PATH, index=False)

    downloaded = set(closes.columns)
    # Final spell per ticker via tail(1) -- see the note in panel_diagnostics on .last().
    final = (spells.sort_values(["ticker", "spell_id"])
                   .groupby("ticker").tail(1).set_index("ticker"))
    coverage = pd.DataFrame({
        "n_spells": spells.groupby("ticker")["spell_id"].size(),
        "still_member": final["end"].isna(),
        "exit_date": final["end"],
        "exit_reason": final["exit_reason"],
    }).reindex(requested_tickers)
    coverage["outcome"] = ["ok" if t in downloaded else "missing" for t in coverage.index]
    coverage["first_obs"] = [closes[t].first_valid_index() if t in downloaded else pd.NaT
                             for t in coverage.index]
    coverage["last_obs"] = [closes[t].last_valid_index() if t in downloaded else pd.NaT
                            for t in coverage.index]
    coverage["n_obs"] = [int(closes[t].notna().sum()) if t in downloaded else 0
                         for t in coverage.index]
    coverage.to_csv(COVERAGE_PATH)

    print(f"  wrote {os.path.basename(PANEL_PATH)} ({len(tidy):,} rows), "
          f"{os.path.basename(SPELLS_PATH)}, {os.path.basename(COVERAGE_PATH)}")


def _truncate_recycled(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    spells: pd.DataFrame,
    grace_days: int = 30,
) -> list[dict]:
    """
    Cut price series that continue long past a ceased-trading exit.

    A delisted symbol can be reassigned, and Yahoo then serves the new company's history
    under the old ticker -- BEAM (Beam Inc, acquired 2014) later became Beam Therapeutics;
    ADT and DELL both re-IPO'd under their old symbols. This succeeds silently and splices
    two unrelated companies into one series.

    Rather than dropping the ticker outright, which would discard its legitimate in-index
    history too, the series is cut a short way past the exit. The grace window keeps the
    price needed to book the removal exit against.
    """
    final = (spells.sort_values(["ticker", "spell_id"])
                   .groupby("ticker").tail(1).set_index("ticker"))
    ceased = final["exit_reason"].fillna("").str.contains(_CEASED_TRADING, case=False)

    flagged = []
    for ticker in closes.columns:
        if ticker not in final.index or not bool(ceased.get(ticker, False)):
            continue
        end = final.loc[ticker, "end"]
        if pd.isna(end):
            continue
        series = closes[ticker].dropna()
        if series.empty or (series.index[-1] - end).days <= 365:
            continue

        cutoff = end + pd.Timedelta(days=grace_days)
        dropped = int((closes.index > cutoff).sum() and closes.loc[closes.index > cutoff, ticker].notna().sum())
        closes.loc[closes.index > cutoff, ticker] = float("nan")
        opens.loc[opens.index > cutoff, ticker] = float("nan")
        flagged.append({"ticker": ticker, "exit": end.date(),
                        "ran_to": series.index[-1].date(), "obs_dropped": dropped})

    return flagged


# ---------------------------------------------------------------- diagnostics

# Names that failed hard while plausibly inside a momentum screen. Yahoo deletes
# delisted companies, so most return nothing -- which is the point of checking.
KNOWN_FAILURES = ["SIVB", "SIVBQ", "SBNY", "FRC", "FRCB",       # 2023 regional banks
                  "SUNE",                                       # 2016, a genuine momentum name
                  "BTU", "CHK", "HTZ", "FTR", "WLL"]            # 2016 / 2020 energy and travel

_CEASED_TRADING = "acquir|merg|bankrupt|purchase|bought|taken private"


def panel_diagnostics(opens, closes, in_index, spells, verbose: bool = True) -> dict:
    """
    Build-time checks. These are meant to be impossible to miss.

    Returns a dict of results so callers can assert on them.
    """
    results = {}
    say = print if verbose else (lambda *a, **k: None)

    say("\n" + "=" * 62)
    say("PANEL DIAGNOSTICS")
    say("=" * 62)

    # --- universe size: the primary validation -------------------------------
    # This must be measured on the FULL spell set, not on `in_index`, which is
    # restricted to tickers Yahoo actually returned. Mixing the two conflates a broken
    # reconstruction with an unavoidable data gap, and only the first is a bug.
    # A reconstruction losing entries drifts away from ~500 monotonically. Oscillation
    # of a few names is log noise (multi-class listings, unlogged renames), so the band
    # is deliberately wider than 500-505.
    full = membership_matrix(spells, in_index.index)
    counts = full.sum(axis=1)
    lo, hi = int(counts.min()), int(counts.max())
    in_band = 495 <= lo and hi <= 510
    results["universe_min"], results["universe_max"] = lo, hi
    results["universe_in_band"] = in_band
    say(f"\nReconstructed universe size:  min {lo}  max {hi}  "
        f"[{'OK' if in_band else 'OUT OF BAND -- investigate before running strategy code'}]")
    yearly = counts.groupby(counts.index.year).agg(["min", "max"])
    say("  " + "  ".join(f"{y}:{r['min']}-{r['max']}" for y, r in yearly.iterrows()))

    # --- how much of that universe is actually priceable ----------------------
    # Not a pass/fail check -- this IS the survivorship gap, and it necessarily widens
    # the further back you look, because Yahoo deletes companies as they disappear.
    priceable = in_index.sum(axis=1)
    share = (priceable / counts.reindex(priceable.index)).dropna()
    results["priceable_min"], results["priceable_max"] = int(priceable.min()), int(priceable.max())
    say(f"\nPriceable subset:  {priceable.min()}-{priceable.max()} names "
        f"({share.min():.0%}-{share.max():.0%} of the index)")
    ys = share.groupby(share.index.year).mean()
    say("  " + "  ".join(f"{y}:{v:.0%}" for y, v in ys.items()))

    # --- coverage by exit reason ---------------------------------------------
    # Low coverage in the ceased-trading bucket is survivorship bias made visible.
    # This measures the gap; it cannot fix it.
    downloaded = set(closes.columns)
    # tail(1), not .last() -- GroupBy.last() skips NaN, so a ticker whose final spell is
    # still open (end = NaT) would report the end date of an EARLIER spell and be
    # misclassified as having left the index.
    last = (spells.sort_values(["ticker", "spell_id"])
                  .groupby("ticker").tail(1).set_index("ticker"))
    ceased = last["exit_reason"].fillna("").str.contains(_CEASED_TRADING, case=False)
    bucket = pd.Series("still in index", index=last.index)
    bucket[last["end"].notna() & ~ceased] = "left index, kept trading"
    bucket[last["end"].notna() & ceased] = "ceased trading"
    got = pd.Series([t in downloaded for t in last.index], index=last.index)

    table = pd.crosstab(bucket, got.map({True: "ok", False: "missing"}))
    results["coverage"] = table
    say("\nCoverage by exit reason:")
    for reason, row in table.iterrows():
        ok, missing = int(row.get("ok", 0)), int(row.get("missing", 0))
        pct = 100 * ok / max(ok + missing, 1)
        say(f"  {reason:<26} ok {ok:4d}   missing {missing:4d}   ({pct:.0f}% covered)")

    # --- known-failure check --------------------------------------------------
    say("\nKnown-failure check:")
    failures = {}
    for ticker in KNOWN_FAILURES:
        if ticker not in downloaded:
            failures[ticker] = None
            say(f"  {ticker:<6} no data")
            continue
        series = closes[ticker].dropna()
        if series.empty:
            failures[ticker] = None
            say(f"  {ticker:<6} no data")
            continue
        # Was it above its 200d MA when it became selectable? A slow collapse breaks the
        # MA months ahead and the signal drops it; a fast one (SVB, over a weekend) could
        # not have been exited under Friday-signal/Monday-open and is the real exposure.
        ma200 = series.rolling(200).mean()
        above = (series > ma200).dropna()
        share = float(above.tail(250).mean()) if len(above) else float("nan")
        failures[ticker] = share
        say(f"  {ticker:<6} {len(series):5d} obs, last {series.index[-1].date()}, "
            f"above 200d MA on {share:.0%} of final year")
    results["known_failures"] = failures

    # --- ticker recycling -----------------------------------------------------
    # A delisted symbol can be reassigned and Yahoo serves the new company's history
    # under it. This succeeds silently and splices two unrelated companies together.
    recycled = []
    for ticker, row in last.iterrows():
        if ticker not in downloaded or pd.isna(row["end"]) or not ceased.get(ticker, False):
            continue
        series = closes[ticker].dropna()
        if series.empty:
            continue
        overrun = (series.index[-1] - row["end"]).days
        if overrun > 365:
            recycled.append((ticker, row["end"].date(), series.index[-1].date(), overrun))
    results["recycled"] = recycled
    say(f"\nTicker recycling: {len(recycled)} flagged "
        f"(data continues >1y past a ceased-trading exit)")
    for ticker, end, last_obs, days in recycled[:10]:
        say(f"  {ticker:<6} exited {end}, data runs to {last_obs}  (+{days}d)")
    if len(recycled) > 10:
        say(f"  ... and {len(recycled) - 10} more")

    say("=" * 62 + "\n")
    return results

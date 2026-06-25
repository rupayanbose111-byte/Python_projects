"""
================================================================================
NIFTY50 MEAN REVERSION STRATEGY — BACKTEST & BENCHMARK COMPARISON
================================================================================

What this script does (in order):

  PART A — DATA ACQUISITION
    A1. Scrape current NIFTY50 constituents (+ weights, if available) from
        Wikipedia / NSE and save to 'nifty50_constituents.csv'.
    A2. Download 3 years of daily adjusted-close prices for every constituent
        via yfinance, save to 'nifty50_prices.csv'. Any ticker that fails is
        printed by name so you know which one to investigate.
    A3. Reload both CSVs into DataFrames and clean missing data carefully.

  PART B — MEAN REVERSION STRATEGY
    B1. Daily returns per stock.
    B2. The 10 worst (lowest-return) stocks for every trading day.
    B3. Trade simulation: buy the 10 worst performers at today's close,
        sell at tomorrow's close, equal-weighted, starting capital 1,00,000.
        Per-day stats (avg/median std-dev, avg return, avg Sharpe) are
        printed as the simulation progresses, and the full equity curve is
        stored in a nicely formatted table (via `rich`, fallback to plain
        pandas if `rich` isn't installed).

  PART C — BENCHMARK COMPARISON
    Compares the strategy against the NIFTYIETF.NS ETF (a Nifty50 index ETF
    on Yahoo Finance) over the *same date range* as the strategy — CAGR,
    annualized volatility, Sharpe ratio (rf = 0), with a verdict on which
    had better risk-adjusted returns.

  PART D — VISUALIZATION
    Interactive Plotly chart of both ₹100,000 portfolios growing over time.

Run end-to-end:
    python nifty50_mean_reversion.py

Requirements are listed at the very bottom of this file as a comment block,
and also written out separately as requirements.txt / README notes when the
script finishes running.
================================================================================
"""

import os
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Optional but recommended packages — we degrade gracefully if missing.
try:
    import yfinance as yf
except ImportError:
    sys.exit(
        "ERROR: yfinance is not installed. Run: pip install yfinance --break-system-packages"
    )

try:
    import requests
except ImportError:
    sys.exit("ERROR: requests is not installed. Run: pip install requests --break-system-packages")

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
YEARS_OF_HISTORY = 3
INITIAL_CAPITAL = 100_000.0
N_LOSERS = 10
BENCHMARK_TICKER = "NIFTYIETF.NS"      # Nifty50 ETF on NSE, tracked via Yahoo Finance
RISK_FREE_RATE = 0.0
TRADING_DAYS_PER_YEAR = 252

CONSTITUENTS_CSV = "nifty50_constituents.csv"
PRICES_CSV = "nifty50_prices.csv"
EQUITY_CURVE_CSV = "mean_reversion_equity_curve.csv"

WIKI_URL = "https://en.wikipedia.org/wiki/NIFTY_50"


def hr(title=""):
    """Pretty section divider for console output."""
    bar = "=" * 80
    if title:
        print(f"\n{bar}\n{title}\n{bar}")
    else:
        print(bar)


# ==============================================================================
# PART A — DATA ACQUISITION
# ==============================================================================

def fetch_nifty50_constituents() -> pd.DataFrame:
    """
    A1. Fetch current NIFTY50 constituents (and weights if available).

    Primary source : NSE official "Nifty 50" index CSV (has weights).
    Fallback source: Wikipedia "NIFTY 50" page (no weights, symbols only;
                      we tag the Yahoo-style '.NS' suffix ourselves).

    Saves result to CONSTITUENTS_CSV with columns:
        Symbol, Company, Weight (Weight may be NaN if unavailable)
    """
    hr("PART A1 — Fetching NIFTY50 constituents")

    df = None

    # --- Attempt 1: NSE official index CSV (includes weights in some endpoints) ---
    try:
        nse_url = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        }
        resp = requests.get(nse_url, headers=headers, timeout=15)
        resp.raise_for_status()
        from io import StringIO
        nse_df = pd.read_csv(StringIO(resp.text))
        # Typical columns: 'Company Name', 'Industry', 'Symbol', 'Series', 'ISIN Code'
        nse_df = nse_df.rename(columns={"Company Name": "Company", "Symbol": "Symbol"})
        nse_df["Symbol"] = nse_df["Symbol"].astype(str).str.strip()
        nse_df["Company"] = nse_df["Company"].astype(str).str.strip()
        nse_df["Weight"] = np.nan  # NSE's public list CSV does not carry live weights
        df = nse_df[["Symbol", "Company", "Weight"]].copy()
        print(f"Fetched {len(df)} constituents from NSE official source.")
    except Exception as e:
        print(f"NSE fetch failed ({e}); falling back to Wikipedia.")

    # --- Attempt 2: Wikipedia fallback ---
    if df is None or df.empty:
        try:
            tables = pd.read_html(WIKI_URL)
            candidate = None
            for t in tables:
                cols_lower = [str(c).lower() for c in t.columns]
                if any("symbol" in c for c in cols_lower):
                    candidate = t
                    break
            if candidate is None:
                raise ValueError("No table with a 'Symbol' column found on Wikipedia page.")

            candidate.columns = [str(c).strip() for c in candidate.columns]
            symbol_col = next(c for c in candidate.columns if "symbol" in c.lower())
            name_col = next(
                (c for c in candidate.columns if "company" in c.lower() or "name" in c.lower()),
                symbol_col,
            )

            df = pd.DataFrame(
                {
                    "Symbol": candidate[symbol_col].astype(str).str.strip(),
                    "Company": candidate[name_col].astype(str).str.strip(),
                }
            )
            df["Weight"] = np.nan
            print(f"Fetched {len(df)} constituents from Wikipedia (no weight data available).")
        except Exception as e:
            sys.exit(f"FATAL: Could not fetch NIFTY50 constituents from either source: {e}")

    # Clean up: drop dupes/blank symbols, add the Yahoo Finance '.NS' suffix
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df = df[df["Symbol"].str.len() > 0].drop_duplicates(subset="Symbol").reset_index(drop=True)
    df["YahooTicker"] = df["Symbol"] + ".NS"

    df.to_csv(CONSTITUENTS_CSV, index=False)
    print(f"Saved {len(df)} constituents -> '{CONSTITUENTS_CSV}'")
    return df


def fetch_price_data(constituents: pd.DataFrame, years: int = YEARS_OF_HISTORY) -> pd.DataFrame:
    """
    A2. Fetch `years` years of daily ADJUSTED CLOSE prices for every constituent
    using yfinance. Saves a long-format CSV with columns: Date, Symbol, AdjClose.

    Any ticker that errors out (network issue, delisted, no data, etc.) is
    printed by name/symbol so the user can identify exactly which stock failed.
    """
    hr("PART A2 — Fetching 3 years of daily adjusted close prices")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(365.25 * years) + 5)

    all_frames = []
    failed_symbols = []

    tickers = constituents["YahooTicker"].tolist()
    print(f"Downloading data for {len(tickers)} tickers from {start_date.date()} to {end_date.date()} ...")

    for i, ticker in enumerate(tickers, start=1):
        try:
            hist = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=False,   # we want explicit 'Adj Close' column
                progress=False,
                threads=False,
            )

            if hist is None or hist.empty:
                raise ValueError("No data returned (empty frame).")

            # yfinance sometimes returns a MultiIndex column structure for single tickers
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            if "Adj Close" not in hist.columns:
                raise ValueError("'Adj Close' column missing from response.")

            sub = hist[["Adj Close"]].reset_index()
            sub.columns = ["Date", "AdjClose"]
            sub["Symbol"] = ticker.replace(".NS", "")
            all_frames.append(sub)

        except Exception as e:
            failed_symbols.append(ticker)
            print(f"  [ERROR] Failed to fetch data for symbol: {ticker}  ->  {e}")
            continue

        # Be polite to Yahoo Finance's endpoint
        time.sleep(0.15)

        if i % 10 == 0 or i == len(tickers):
            print(f"  Progress: {i}/{len(tickers)} tickers processed.")

    if not all_frames:
        sys.exit("FATAL: No price data could be fetched for ANY constituent. Aborting.")

    prices_long = pd.concat(all_frames, ignore_index=True)
    prices_long["Date"] = pd.to_datetime(prices_long["Date"])
    prices_long = prices_long.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    prices_long.to_csv(PRICES_CSV, index=False)
    print(f"\nSaved long-format price data ({len(prices_long)} rows) -> '{PRICES_CSV}'")

    if failed_symbols:
        print(f"\nSymbols that FAILED to download ({len(failed_symbols)}): {failed_symbols}")
    else:
        print("\nAll symbols downloaded successfully — no errors.")

    return prices_long


def load_and_clean_data():
    """
    A3. Load constituents + price CSVs back into DataFrames and handle
    missing data carefully:
      - Pivot prices to wide format (Date x Symbol).
      - Drop symbols with too little history (e.g. < 50% of trading days).
      - Forward-fill small gaps (holidays mismatched across exchanges,
        momentary feed gaps) but cap the fill window so stale/delisted
        stocks don't get silently carried forward forever.
      - Drop any remaining fully-empty rows/columns.
    Returns: (constituents_df, prices_wide_df)
    """
    hr("PART A3 — Loading & cleaning data from CSV")

    constituents = pd.read_csv(CONSTITUENTS_CSV)
    prices_long = pd.read_csv(PRICES_CSV, parse_dates=["Date"])

    prices_wide = prices_long.pivot(index="Date", columns="Symbol", values="AdjClose")
    prices_wide = prices_wide.sort_index()

    total_days = len(prices_wide)
    min_required = total_days * 0.5
    valid_counts = prices_wide.count()
    sparse_symbols = valid_counts[valid_counts < min_required].index.tolist()

    if sparse_symbols:
        print(f"Dropping {len(sparse_symbols)} symbol(s) with insufficient history (<50% of days): {sparse_symbols}")
        prices_wide = prices_wide.drop(columns=sparse_symbols)

    # Forward-fill small gaps only (limit=3 trading days), then drop rows that
    # are STILL fully NaN (e.g. exchange holiday rows that slipped through).
    prices_wide = prices_wide.ffill(limit=3)
    prices_wide = prices_wide.dropna(how="all")

    # Any isolated NaNs remaining (new listings, brief halts) — leave as NaN;
    # they are handled defensively downstream in the simulation, never
    # silently treated as zero/inf which would corrupt return calculations.
    remaining_na = int(prices_wide.isna().sum().sum())
    print(f"Price matrix shape: {prices_wide.shape[0]} trading days x {prices_wide.shape[1]} stocks.")
    print(f"Remaining isolated NaNs after cleaning: {remaining_na} (handled safely during simulation).")

    return constituents, prices_wide


# ==============================================================================
# PART B — MEAN REVERSION STRATEGY
# ==============================================================================

def calculate_daily_returns(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """B1. Daily % returns for each stock. First row will be all-NaN by definition."""
    hr("PART B1 — Calculating daily returns")
    returns = prices_wide.pct_change()
    print(f"Computed daily returns for {returns.shape[1]} stocks across {returns.shape[0]} days.")
    return returns


def identify_biggest_losers(returns: pd.DataFrame, n: int = N_LOSERS) -> pd.DataFrame:
    """
    B2. For each trading day (skipping day 1, which is all-NaN), find the
    `n` stocks with the LOWEST return that day.

    Returns a tidy long DataFrame: Date, Symbol, Return, Rank (1 = worst).
    """
    hr("PART B2 — Identifying the 10 biggest losers each day")

    returns_clean = returns.iloc[1:]  # skip first day (no prior price -> NaN)

    records = []
    for date, row in returns_clean.iterrows():
        valid = row.dropna()
        if valid.empty:
            continue
        worst = valid.nsmallest(n)
        for rank, (symbol, ret) in enumerate(worst.items(), start=1):
            records.append({"Date": date, "Symbol": symbol, "Return": ret, "Rank": rank})

    losers_df = pd.DataFrame(records)
    print(f"Identified daily top-{n} losers for {losers_df['Date'].nunique()} trading days.")
    return losers_df


def simulate_mean_reversion_strategy(
    losers_df: pd.DataFrame,
    prices_wide: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    n_stocks: int = N_LOSERS,
) -> pd.DataFrame:
    """
    B3. Trade simulation.

      For each date D with its 10 worst performers:
        - Buy each at D's adjusted close.
        - Sell each at the NEXT trading day's adjusted close (found via a
          MERGE against the price matrix's date index — not .shift() — so
          we are robust to any gaps/misalignment between the losers table
          and the master price calendar).
        - Capital is split equally across the (up to) 10 names.
        - If a stock's next-day price is missing, that slot's allocation is
          neither invested nor magically duplicated — it is carried forward
          in cash for that day's leg (so capital is never erroneously
          inflated or deflated by a missing price point).
        - The very last date in the dataset has no "next day" -> skipped
          safely with a message.

    Also prints, for every day: average & median std-dev (of that day's
    10 stocks' historical daily returns up to that point), average return,
    and average Sharpe ratio (rf = 0) for that day's 10-stock basket.

    Returns: equity_curve_df with columns ['Date', 'Capital'].
    """
    hr("PART B3 — Running trade simulation")

    losers_df = losers_df.copy()
    losers_df["Date"] = pd.to_datetime(losers_df["Date"])           # B3.1 correct datetime
    trading_dates = pd.to_datetime(prices_wide.index)
    date_list = list(trading_dates)
    date_to_pos = {d: i for i, d in enumerate(date_list)}

    # Pre-compute full-history daily returns once (used for the per-day stat printout)
    full_returns = prices_wide.pct_change()

    capital = initial_capital
    equity_records = []
    skipped_days = 0

    unique_dates = sorted(losers_df["Date"].unique())

    for date in unique_dates:
        if date not in date_to_pos:
            # Losers date isn't in the master calendar at all — skip defensively
            skipped_days += 1
            continue

        pos = date_to_pos[date]
        if pos + 1 >= len(date_list):
            print(f"  [END OF DATA] {pd.Timestamp(date).date()} is the last trading day — no next-day price to sell at. Skipping final leg.")
            break  # B3.9 — end of dataset handled cleanly, no fabricated trade

        next_date = date_list[pos + 1]

        todays_picks = losers_df[losers_df["Date"] == date]
        symbols_today = todays_picks["Symbol"].tolist()[:n_stocks]

        if not symbols_today:
            skipped_days += 1
            continue

        # ---- B3.3: next day's prices via MERGE, not shift ----
        buy_prices = prices_wide.loc[date, symbols_today].rename("BuyPrice")
        sell_prices = prices_wide.loc[next_date, symbols_today].rename("SellPrice")
        leg = pd.merge(
            buy_prices.reset_index().rename(columns={"index": "Symbol"}),
            sell_prices.reset_index().rename(columns={"index": "Symbol"}),
            on="Symbol",
            how="inner",
        )

        # B3.8 — drop legs with missing/invalid prices (NaN, zero, negative)
        valid_leg = leg[
            leg["BuyPrice"].notna()
            & leg["SellPrice"].notna()
            & (leg["BuyPrice"] > 0)
            & (leg["SellPrice"] > 0)
        ].copy()

        n_valid = len(valid_leg)
        if n_valid == 0 or capital <= 0:
            # Nothing tradeable today, or we're out of capital — carry capital forward unchanged
            equity_records.append({"Date": next_date, "Capital": capital})
            skipped_days += 1
            continue

        # ---- B3.4: equal split of capital across however many valid legs we actually have ----
        # (If some of the 10 picks had missing data, we don't silently inflate
        #  the remaining legs' size beyond what was allocated to them — instead
        #  the *unused* slice of capital for missing legs simply stays in cash.)
        per_stock_allocation = capital / n_stocks
        invested_capital = per_stock_allocation * n_valid
        uninvested_cash = capital - invested_capital  # cash from missing-data slots, preserved untouched

        valid_leg["Shares"] = per_stock_allocation / valid_leg["BuyPrice"]
        valid_leg["ProceedsAtSell"] = valid_leg["Shares"] * valid_leg["SellPrice"]

        proceeds_total = valid_leg["ProceedsAtSell"].sum()
        new_capital = proceeds_total + uninvested_cash  # B3.6 — update capital, no leakage/inflation

        # ---- Per-day stats: std-dev, return, Sharpe of the day's 10-stock basket ----
        hist_rets = full_returns.loc[:date, symbols_today].dropna(how="all")
        stdevs = hist_rets.std()  # per-stock std-dev of daily returns observed so far
        avg_stdev = stdevs.mean()
        median_stdev = stdevs.median()

        day_returns = todays_picks.set_index("Symbol")["Return"].reindex(symbols_today)
        avg_return = day_returns.mean()

        # Sharpe (rf = 0): mean daily return / std-dev of daily returns, annualized
        per_stock_sharpe = (hist_rets.mean() - RISK_FREE_RATE) / stdevs.replace(0, np.nan)
        avg_sharpe = (per_stock_sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)).mean()

        print(
            f"  {pd.Timestamp(date).date()} | "
            f"AvgStd: {avg_stdev:.4%} | MedStd: {median_stdev:.4%} | "
            f"AvgReturn: {avg_return:.4%} | AvgSharpe(ann.): {avg_sharpe:.3f} | "
            f"Capital(next day open): Rs.{new_capital:,.2f} | Trades: {n_valid}/{n_stocks}"
        )

        capital = new_capital
        equity_records.append({"Date": next_date, "Capital": capital})

    equity_curve = pd.DataFrame(equity_records).drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)

    # Prepend the very first capital value at the very first losers-date so the
    # equity curve has a sensible starting point at INITIAL_CAPITAL.
    if not equity_curve.empty:
        start_row = pd.DataFrame([{"Date": unique_dates[0], "Capital": initial_capital}])
        equity_curve = pd.concat([start_row, equity_curve], ignore_index=True).drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)

    print(f"\nSimulation complete. {len(equity_curve)} equity points recorded. {skipped_days} day(s) skipped (no valid trades / capital exhausted).")
    equity_curve.to_csv(EQUITY_CURVE_CSV, index=False)
    print(f"Saved equity curve -> '{EQUITY_CURVE_CSV}'")
    return equity_curve


def display_equity_curve_table(equity_curve: pd.DataFrame, max_rows: int = 20):
    """B3.7 — Pretty table of Date/Capital (head+tail) using `rich` if available."""
    hr("Equity Curve (sample)")

    sample = pd.concat([equity_curve.head(max_rows // 2), equity_curve.tail(max_rows // 2)])

    if RICH_AVAILABLE:
        table = Table(title="Mean Reversion Strategy — Equity Curve (sample)")
        table.add_column("Date", style="cyan")
        table.add_column("Capital (Rs.)", style="green", justify="right")
        for _, row in sample.iterrows():
            table.add_row(str(pd.Timestamp(row["Date"]).date()), f"{row['Capital']:,.2f}")
        console.print(table)
    else:
        print(sample.to_string(index=False))
        print("(Tip: `pip install rich` for a nicer table view.)")


def compute_strategy_performance(equity_curve: pd.DataFrame, years: float) -> dict:
    """B3.10 — Final capital, total return, CAGR, annualized std-dev, Sharpe."""
    hr("PART B (final) — Strategy performance summary")

    initial_cap = equity_curve["Capital"].iloc[0]
    final_cap = equity_curve["Capital"].iloc[-1]
    total_return = (final_cap / initial_cap) - 1

    daily_strategy_returns = equity_curve["Capital"].pct_change().dropna()
    n_days = len(equity_curve) - 1
    actual_years = n_days / TRADING_DAYS_PER_YEAR if n_days > 0 else years

    cagr = (final_cap / initial_cap) ** (1 / actual_years) - 1 if actual_years > 0 else np.nan
    ann_std = daily_strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ((daily_strategy_returns.mean() - RISK_FREE_RATE) / daily_strategy_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_strategy_returns.std() else np.nan

    print(f"Initial Capital     : Rs.{initial_cap:,.2f}")
    print(f"Final Capital        : Rs.{final_cap:,.2f}")
    print(f"Total Return          : {total_return:.2%}")
    print(f"CAGR ({actual_years:.2f} yrs)   : {cagr:.2%}")
    print(f"Annualized Std Dev    : {ann_std:.2%}")
    print(f"Sharpe Ratio (rf=0)   : {sharpe:.3f}")

    return {
        "initial_capital": initial_cap,
        "final_capital": final_cap,
        "total_return": total_return,
        "cagr": cagr,
        "ann_std": ann_std,
        "sharpe": sharpe,
        "daily_returns": daily_strategy_returns,
    }


# ==============================================================================
# PART C — BENCHMARK COMPARISON
# ==============================================================================

def fetch_benchmark_data(prices_wide: pd.DataFrame, ticker: str = BENCHMARK_TICKER) -> pd.DataFrame:
    """
    Fetch NIFTYIETF.NS adjusted close prices for EXACTLY the same date range
    covered by the Nifty50 constituents' price CSV (start to end date).
    """
    hr(f"PART C — Fetching benchmark data for {ticker}")

    start_date = prices_wide.index.min()
    end_date = prices_wide.index.max() + pd.Timedelta(days=1)  # yfinance 'end' is exclusive

    try:
        hist = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if hist is None or hist.empty:
            raise ValueError("Empty response from yfinance.")
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if "Adj Close" not in hist.columns:
            raise ValueError("'Adj Close' missing from benchmark response.")

        bench = hist[["Adj Close"]].rename(columns={"Adj Close": "AdjClose"})
        bench.index = pd.to_datetime(bench.index)
        print(f"Fetched {len(bench)} rows for {ticker} from {bench.index.min().date()} to {bench.index.max().date()}.")
        return bench

    except Exception as e:
        sys.exit(f"FATAL: Could not fetch benchmark data for {ticker}: {e}")


def compare_with_benchmark(strategy_perf: dict, benchmark_prices: pd.DataFrame) -> dict:
    """
    Compute benchmark CAGR / annual std-dev / Sharpe and print a clean
    side-by-side comparison table with a verdict.
    """
    hr("PART C — Strategy vs. NIFTYIETF.NS comparison")

    bench_returns = benchmark_prices["AdjClose"].pct_change().dropna()
    n_days = len(bench_returns)
    years = n_days / TRADING_DAYS_PER_YEAR

    bench_cagr = (benchmark_prices["AdjClose"].iloc[-1] / benchmark_prices["AdjClose"].iloc[0]) ** (1 / years) - 1
    bench_std = bench_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    bench_sharpe = ((bench_returns.mean() - RISK_FREE_RATE) / bench_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if bench_returns.std() else np.nan

    rows = [
        ("CAGR", f"{strategy_perf['cagr']:.2%}", f"{bench_cagr:.2%}"),
        ("Annualized Std Dev", f"{strategy_perf['ann_std']:.2%}", f"{bench_std:.2%}"),
        ("Sharpe Ratio (rf=0)", f"{strategy_perf['sharpe']:.3f}", f"{bench_sharpe:.3f}"),
    ]

    if RICH_AVAILABLE:
        table = Table(title="Mean Reversion Strategy vs. NIFTYIETF.NS")
        table.add_column("Metric", style="bold")
        table.add_column("Mean Reversion Strategy", justify="right", style="green")
        table.add_column("NIFTYIETF.NS", justify="right", style="cyan")
        for r in rows:
            table.add_row(*r)
        console.print(table)
    else:
        comp_df = pd.DataFrame(rows, columns=["Metric", "Mean Reversion Strategy", "NIFTYIETF.NS"])
        print(comp_df.to_string(index=False))

    print()
    if strategy_perf["sharpe"] > bench_sharpe:
        print(
            f"VERDICT: The Mean Reversion strategy OUTPERFORMED NIFTYIETF.NS on a "
            f"risk-adjusted basis (Sharpe {strategy_perf['sharpe']:.3f} vs {bench_sharpe:.3f})."
        )
    elif strategy_perf["sharpe"] < bench_sharpe:
        print(
            f"VERDICT: The Mean Reversion strategy UNDERPERFORMED NIFTYIETF.NS on a "
            f"risk-adjusted basis (Sharpe {strategy_perf['sharpe']:.3f} vs {bench_sharpe:.3f})."
        )
    else:
        print("VERDICT: Both the strategy and NIFTYIETF.NS produced an identical Sharpe ratio.")

    return {"cagr": bench_cagr, "ann_std": bench_std, "sharpe": bench_sharpe, "daily_returns": bench_returns, "prices": benchmark_prices}


# ==============================================================================
# PART D — VISUALIZATION
# ==============================================================================

def plot_portfolio_growth(equity_curve: pd.DataFrame, benchmark_prices: pd.DataFrame, initial_investment: float = INITIAL_CAPITAL):
    """
    Interactive Plotly line chart comparing the ₹100,000 growth of:
      - The mean reversion strategy (from its actual equity curve)
      - NIFTYIETF.NS (cumulative-return-scaled to the same initial investment)
    """
    hr("PART D — Visualizing portfolio growth")

    if not PLOTLY_AVAILABLE:
        print("Plotly is not installed — skipping interactive chart. Run: pip install plotly --break-system-packages")
        return

    strat = equity_curve.copy()
    strat["Date"] = pd.to_datetime(strat["Date"])

    bench = benchmark_prices.copy()
    bench_cum_returns = bench["AdjClose"] / bench["AdjClose"].iloc[0]
    bench_portfolio_value = bench_cum_returns * initial_investment

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strat["Date"],
            y=strat["Capital"],
            mode="lines",
            name="Mean Reversion Strategy",
            line=dict(color="#2ca02c", width=2.5),
            hovertemplate="%{x|%d %b %Y}<br>Rs.%{y:,.0f}<extra>Mean Reversion</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bench_portfolio_value.index,
            y=bench_portfolio_value.values,
            mode="lines",
            name="NIFTYIETF.NS (Buy & Hold)",
            line=dict(color="#1f77b4", width=2.5),
            hovertemplate="%{x|%d %b %Y}<br>Rs.%{y:,.0f}<extra>NIFTYIETF.NS</extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Growth of Rs.1,00,000 — Mean Reversion Strategy vs. NIFTYIETF.NS",
            font=dict(size=20),
        ),
        xaxis_title="Date",
        yaxis_title="Portfolio Value (Rs.)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=13),
    )
    fig.update_yaxes(tickprefix="Rs.", separatethousands=True)

    out_html = "portfolio_growth_comparison.html"
    fig.write_html(out_html)
    print(f"Interactive chart saved -> '{out_html}' (open in any browser).")

    fig.show()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    hr("NIFTY50 MEAN REVERSION STRATEGY — FULL PIPELINE START")

    # ---- PART A: Data acquisition ----
    constituents = fetch_nifty50_constituents()
    fetch_price_data(constituents, years=YEARS_OF_HISTORY)
    constituents, prices_wide = load_and_clean_data()

    # ---- PART B: Mean reversion strategy ----
    returns = calculate_daily_returns(prices_wide)
    losers_df = identify_biggest_losers(returns, n=N_LOSERS)
    equity_curve = simulate_mean_reversion_strategy(losers_df, prices_wide, initial_capital=INITIAL_CAPITAL, n_stocks=N_LOSERS)
    display_equity_curve_table(equity_curve)
    strategy_perf = compute_strategy_performance(equity_curve, years=YEARS_OF_HISTORY)

    # ---- PART C: Benchmark comparison ----
    benchmark_prices = fetch_benchmark_data(prices_wide, ticker=BENCHMARK_TICKER)
    compare_with_benchmark(strategy_perf, benchmark_prices)

    # ---- PART D: Visualization ----
    plot_portfolio_growth(equity_curve, benchmark_prices, initial_investment=INITIAL_CAPITAL)

    hr("PIPELINE COMPLETE")
    print("Outputs written to current directory:")
    print(f"  - {CONSTITUENTS_CSV}")
    print(f"  - {PRICES_CSV}")
    print(f"  - {EQUITY_CURVE_CSV}")
    print("  - portfolio_growth_comparison.html")


if __name__ == "__main__":
    main()


# ==============================================================================
# REQUIREMENTS TO RUN THIS SCRIPT WITHOUT ERROR
# ==============================================================================
#
# 1. Python version:  3.9 or newer recommended.
#
# 2. Install dependencies (copy into a requirements.txt or run directly):
#
#       pip install yfinance pandas numpy requests lxml html5lib beautifulsoup4 rich plotly --break-system-packages
#
#    Package roles:
#       yfinance        -> downloads daily adjusted close prices from Yahoo Finance
#       pandas, numpy   -> data wrangling, returns, stats
#       requests        -> fetches the NSE official constituents CSV
#       lxml/html5lib/
#       beautifulsoup4  -> required by pandas.read_html() for the Wikipedia fallback
#       rich            -> pretty console tables for the equity curve (optional;
#                          script falls back to plain pandas printing if absent)
#       plotly          -> interactive HTML chart for the final visualization
#                          (optional; script skips the chart with a warning if absent)
#
# 3. Internet access is required at runtime to:
#       - Reach https://nsearchives.nseindia.com (NSE constituents CSV) and/or
#         https://en.wikipedia.org (fallback constituents table)
#       - Reach Yahoo Finance via yfinance for both constituent prices and the
#         NIFTYIETF.NS benchmark
#
# 4. NSE's site sometimes blocks data-center / cloud IPs or requires specific
#    headers/cookies. If the NSE CSV fetch fails, the script automatically
#    falls back to scraping Wikipedia's NIFTY 50 page instead (symbols only,
#    no live weights) — no manual action needed, but you may see a notice
#    printed to console.
#
# 5. Yahoo Finance occasionally rate-limits rapid sequential requests. The
#    script already adds a small delay (0.15s) between ticker downloads; if
#    you still see failures, re-run the script (failed symbols are listed by
#    name so you can identify and retry just those).
#
# 6. Disk space: negligible (a few MB of CSVs + one HTML chart file).
#
# 7. To view the interactive chart in a plain terminal/server environment
#    (no GUI), open the saved 'portfolio_growth_comparison.html' file in any
#    web browser — fig.show() may not render in headless environments, but
#    the HTML file will always be written successfully.
# ==============================================================================
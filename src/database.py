import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import numpy as np
import yfinance as yf

# Constants for robust downloading
MAX_RETRIES = 5
SLEEP_BASE = 2.0

class MarketDB:
    """
    Manages the local SQLite database for market data.
    Handles incremental updates, data retrieval, and robust ingestion from yfinance.
    """

    def __init__(self, db_name: str = "market.db"):
        # Automatically places the DB inside the 'data' folder
        # Path(__file__).parent is 'src', so .parent.parent is root, then /data
        self.db_path = Path(__file__).parent.parent / "data" / db_name
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")  # Write-Ahead Logging for performance
        return conn

    def _init_db(self):
        """Creates tables if they do not exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT NOT NULL,
            date   DATE NOT NULL,
            adj_close REAL NOT NULL,
            PRIMARY KEY(symbol, date)
        );
        CREATE INDEX IF NOT EXISTS ix_prices_date ON prices(date);
        CREATE INDEX IF NOT EXISTS ix_prices_symbol ON prices(symbol);
        """
        with self.get_connection() as conn:
            conn.executescript(ddl)

    def _download_robust(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """
        Downloads data from yfinance with retry logic and backoff.
        Returns a DataFrame with 'Adj Close' prices.
        """
        attempt = 0
        while attempt <= MAX_RETRIES:
            try:
                # auto_adjust=False ensures we get 'Adj Close' explicitly
                df = yf.download(tickers, start=start, end=end, progress=False, 
                                 auto_adjust=False, group_by="column", threads=True)
                
                if df.empty:
                    return pd.DataFrame()

                # Handle MultiIndex columns (common with yfinance)
                if isinstance(df.columns, pd.MultiIndex):
                    if "Adj Close" in df.columns.get_level_values(0):
                        df = df["Adj Close"]
                    elif "Close" in df.columns.get_level_values(0):
                        # Fallback if Adj Close is missing (rare with auto_adjust=False)
                        df = df["Close"]
                
                # If single ticker, ensure it's a DataFrame
                if isinstance(df, pd.Series):
                    df = df.to_frame(name=tickers[0])

                # Clean index
                df.index = pd.to_datetime(df.index)
                return df

            except Exception as e:
                attempt += 1
                sleep_time = SLEEP_BASE * (2 ** (attempt - 1))
                print(f"[WARN] Download failed for {tickers}. Retrying in {sleep_time:.1f}s... Error: {e}")
                time.sleep(sleep_time)
        
        print(f"[ERROR] Max retries reached for {tickers}.")
        return pd.DataFrame()

    def save_prices(self, df: pd.DataFrame):
        """
        Saves a DataFrame of prices to the database (UPSERT).
        df: Index=Date, Columns=Tickers
        """
        if df.empty:
            return

        # Flatten data for SQL insertion: (symbol, date, price)
        records = []
        for symbol in df.columns:
            subset = df[symbol].dropna()
            for date, price in subset.items():
                records.append((symbol, date.strftime("%Y-%m-%d"), float(price)))

        if not records:
            return

        sql = """
        INSERT INTO prices (symbol, date, adj_close)
        VALUES (?, ?, ?)
        ON CONFLICT(symbol, date) DO UPDATE SET adj_close=excluded.adj_close;
        """
        
        with self.get_connection() as conn:
            conn.executemany(sql, records)
            conn.commit()
        
        print(f"[INFO] Saved {len(records)} records to database.")

    def get_last_date(self, ticker: str) -> Optional[pd.Timestamp]:
        """Returns the most recent date available in the DB for a ticker."""
        sql = "SELECT MAX(date) as last_date FROM prices WHERE symbol = ?"
        with self.get_connection() as conn:
            cursor = conn.execute(sql, (ticker,))
            result = cursor.fetchone()
            if result and result[0]:
                return pd.to_datetime(result[0])
        return None

    def update_tickers(self, tickers: List[str], start_date: str = "2000-01-01"):
        """
        Smart update: checks the DB for existing data and only downloads new data.
        """
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        
        for ticker in tickers:
            last_date = self.get_last_date(ticker)
            
            if last_date is None:
                # No data, download full history
                current_start = start_date
                print(f"[INFO] {ticker}: No data found. Downloading from {current_start}...")
            else:
                # Incremental update
                current_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                print(f"[INFO] {ticker}: Data found up to {last_date.date()}. Checking updates from {current_start}...")

            if pd.to_datetime(current_start) >= pd.to_datetime(today):
                print(f"[INFO] {ticker} is already up to date.")
                continue

            # Download and Save
            df = self._download_robust([ticker], start=current_start, end=today)
            self.save_prices(df)

    def get_prices(self, tickers: List[str], start_date: str = "2000-01-01") -> pd.DataFrame:
        """
        Retrieves prices from the database for analysis.
        Returns: DataFrame (Index=Date, Columns=Tickers)
        """
        placeholders = ",".join(["?"] * len(tickers))
        sql = f"""
        SELECT date, symbol, adj_close 
        FROM prices 
        WHERE symbol IN ({placeholders}) AND date >= ?
        ORDER BY date ASC
        """
        
        params = tickers + [start_date]
        
        with self.get_connection() as conn:
            df = pd.read_sql(sql, conn, params=params, parse_dates=["date"])
        
        if df.empty:
            return pd.DataFrame()

        # Pivot to format: Index=Date, Columns=Tickers
        return df.pivot(index="date", columns="symbol", values="adj_close")
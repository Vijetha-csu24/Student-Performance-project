"""Fetch news + sentiment from Alpha Vantage's NEWS_SENTIMENT endpoint.

Alpha Vantage returns up to 1000 articles per call. We walk month-by-month per
ticker and cache raw JSON under data/raw/news/{ticker}/{YYYY-MM}.json so reruns
are idempotent and resumable after rate-limit exhaustion (free tier: 25 req/day).

Usage:
    python -m src.data.ingest_news --tickers NVDA,AMD --start 2022-04-01 --end 2022-06-30
    python -m src.data.ingest_news --config configs/tickers.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

API_URL = "https://www.alphavantage.co/query"
RAW_DIR = Path("data/raw/news")
MAX_LIMIT = 1000
REQUEST_TIMEOUT = 30
# Free tier: 25 req/day, 5 req/min. Sleep between calls to stay under the burst cap.
DEFAULT_SLEEP_SEC = 13


@dataclass
class MonthWindow:
    ticker: str
    year: int
    month: int

    @property
    def time_from(self) -> str:
        return f"{self.year:04d}{self.month:02d}01T0000"

    @property
    def time_to(self) -> str:
        if self.month == 12:
            nxt = datetime(self.year + 1, 1, 1)
        else:
            nxt = datetime(self.year, self.month + 1, 1)
        last = nxt - timedelta(seconds=1)
        return last.strftime("%Y%m%dT%H%M")

    @property
    def cache_path(self) -> Path:
        return RAW_DIR / self.ticker / f"{self.year:04d}-{self.month:02d}.json"


def month_windows(ticker: str, start: datetime, end: datetime) -> list[MonthWindow]:
    windows: list[MonthWindow] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        windows.append(MonthWindow(ticker=ticker, year=y, month=m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return windows


def fetch_window(window: MonthWindow, api_key: str) -> dict:
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": window.ticker,
        "time_from": window.time_from,
        "time_to": window.time_to,
        "limit": MAX_LIMIT,
        "sort": "EARLIEST",
        "apikey": api_key,
    }
    resp = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    # Alpha Vantage signals throttling via an "Information" or "Note" key (HTTP 200).
    if "Information" in payload or "Note" in payload:
        msg = payload.get("Information") or payload.get("Note")
        raise RuntimeError(f"Alpha Vantage rate-limited / info response: {msg}")
    return payload


def save_window(window: MonthWindow, payload: dict) -> None:
    window.cache_path.parent.mkdir(parents=True, exist_ok=True)
    window.cache_path.write_text(json.dumps(payload, indent=2))


def load_tickers_from_config(path: Path) -> tuple[list[str], datetime, datetime]:
    cfg = yaml.safe_load(path.read_text())
    tickers = [t["ticker"] for t in cfg.get("core", []) + cfg.get("extended", [])]
    dr = cfg["date_range"]
    return tickers, datetime.fromisoformat(dr["start"]), datetime.fromisoformat(dr["end"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="YAML with tickers + date_range")
    p.add_argument("--tickers", type=str, default=None, help="Comma-separated override, e.g. NVDA,AMD")
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD override")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD override")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SEC, help="Seconds between API calls")
    p.add_argument("--force", action="store_true", help="Re-fetch even if cache exists")
    return p.parse_args()


def resolve_plan(args: argparse.Namespace) -> tuple[list[str], datetime, datetime]:
    if args.config:
        tickers, start, end = load_tickers_from_config(args.config)
    else:
        tickers, start, end = [], datetime(2022, 4, 1), datetime.utcnow()
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.start:
        start = datetime.fromisoformat(args.start)
    if args.end:
        end = datetime.fromisoformat(args.end)
    if not tickers:
        raise SystemExit("No tickers resolved. Pass --config or --tickers.")
    return tickers, start, end


def main() -> int:
    args = parse_args()
    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print("ALPHA_VANTAGE_API_KEY not set. Copy .env.example to .env and fill it in.", file=sys.stderr)
        return 2

    tickers, start, end = resolve_plan(args)
    windows = [w for t in tickers for w in month_windows(t, start, end)]
    print(f"Planned {len(windows)} ticker-month windows across {len(tickers)} tickers.")

    fetched = skipped = 0
    for w in windows:
        if w.cache_path.exists() and not args.force:
            skipped += 1
            continue
        print(f"  fetching {w.ticker} {w.year}-{w.month:02d} ...", flush=True)
        try:
            payload = fetch_window(w, api_key)
        except RuntimeError as e:
            # Rate limit hit -- stop cleanly so caller can resume tomorrow.
            print(f"Stopping: {e}", file=sys.stderr)
            break
        except requests.HTTPError as e:
            print(f"HTTP error on {w.ticker} {w.year}-{w.month:02d}: {e}", file=sys.stderr)
            break
        save_window(w, payload)
        fetched += 1
        time.sleep(args.sleep)

    print(f"Done. fetched={fetched} skipped(existing)={skipped} remaining={len(windows) - fetched - skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

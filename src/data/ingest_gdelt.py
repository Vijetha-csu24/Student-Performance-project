"""Fetch daily news tone + article volume from GDELT DOC 2.0 API.

No API key required. GDELT DOC 2.0 covers 2017-present. We hit two modes per
ticker-window:

    * TimelineTone      -> daily average article tone (-100 .. +100, usually -10..+10)
    * TimelineVolRaw    -> raw article count matching the query per day

GDELT caps a single call at ~366 days of daily resolution, so we walk the
configured date range in 6-month windows per ticker and cache JSON under
data/raw/gdelt/{ticker}/{MODE}_{start}_{end}.json.

Usage:
    python -m src.data.ingest_gdelt --config configs/tickers.yaml
    python -m src.data.ingest_gdelt --tickers NVDA --start 2022-04-01 --end 2022-09-30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import requests
import yaml

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
RAW_DIR = Path("data/raw/gdelt")
REQUEST_TIMEOUT = 60
# GDELT 403s the default python-requests UA; identify ourselves like a browser client.
USER_AGENT = "semiconductor-ml-research/0.1 (+https://github.com/vijetha-csu24/student-performance-project)"
# GDELT requests polite ~1 req/sec max; we use 2s to be safe.
DEFAULT_SLEEP_SEC = 2.0
WINDOW_DAYS = 180  # Keep under GDELT's ~366-day cap for daily-resolution timelines.
MODES = ("timelinetone", "timelinevolraw")


@dataclass
class TickerSpec:
    ticker: str
    query: str


@dataclass
class GdeltWindow:
    ticker: str
    query: str
    start: datetime
    end: datetime
    mode: str

    @property
    def start_str(self) -> str:
        return self.start.strftime("%Y%m%d%H%M%S")

    @property
    def end_str(self) -> str:
        return self.end.strftime("%Y%m%d%H%M%S")

    @property
    def cache_path(self) -> Path:
        slug = f"{self.mode}_{self.start:%Y%m%d}_{self.end:%Y%m%d}.json"
        return RAW_DIR / self.ticker / slug


def build_windows(spec: TickerSpec, start: datetime, end: datetime) -> list[GdeltWindow]:
    windows: list[GdeltWindow] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=WINDOW_DAYS), end)
        for mode in MODES:
            windows.append(
                GdeltWindow(
                    ticker=spec.ticker,
                    query=spec.query,
                    start=cursor,
                    end=chunk_end,
                    mode=mode,
                )
            )
        cursor = chunk_end
    return windows


def fetch_window(win: GdeltWindow) -> dict:
    params = {
        "query": win.query,
        "mode": win.mode,
        "format": "json",
        "timelinesmooth": 0,
        "startdatetime": win.start_str,
        "enddatetime": win.end_str,
    }
    url = f"{API_URL}?{urlencode(params)}"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    # GDELT sometimes returns HTML error pages with 200; guard against it.
    ctype = resp.headers.get("Content-Type", "")
    if "json" not in ctype.lower() and not resp.text.lstrip().startswith("{"):
        raise RuntimeError(f"Non-JSON response from GDELT ({ctype}): {resp.text[:200]}")
    return resp.json()


def save_window(win: GdeltWindow, payload: dict) -> None:
    win.cache_path.parent.mkdir(parents=True, exist_ok=True)
    win.cache_path.write_text(json.dumps(payload, indent=2))


def load_specs(path: Path) -> tuple[list[TickerSpec], datetime, datetime]:
    cfg = yaml.safe_load(path.read_text())
    entries = cfg.get("core", []) + cfg.get("extended", [])
    specs = []
    for e in entries:
        q = e.get("gdelt_query")
        if not q:
            print(f"  skip {e['ticker']}: no gdelt_query in config", file=sys.stderr)
            continue
        specs.append(TickerSpec(ticker=e["ticker"], query=q))
    dr = cfg["date_range"]
    return specs, datetime.fromisoformat(dr["start"]), datetime.fromisoformat(dr["end"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/tickers.yaml"))
    p.add_argument("--tickers", type=str, default=None, help="Comma-separated override")
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD override")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD override")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SEC)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def filter_specs(all_specs: list[TickerSpec], subset: str | None) -> list[TickerSpec]:
    if not subset:
        return all_specs
    wanted = {t.strip().upper() for t in subset.split(",") if t.strip()}
    return [s for s in all_specs if s.ticker in wanted]


def main() -> int:
    args = parse_args()
    specs, start, end = load_specs(args.config)
    specs = filter_specs(specs, args.tickers)
    if args.start:
        start = datetime.fromisoformat(args.start)
    if args.end:
        end = datetime.fromisoformat(args.end)
    if not specs:
        raise SystemExit("No tickers resolved. Check --tickers / config.")

    windows = [w for spec in specs for w in build_windows(spec, start, end)]
    print(f"Planned {len(windows)} GDELT windows across {len(specs)} tickers.")

    fetched = skipped = failed = 0
    for w in windows:
        if w.cache_path.exists() and not args.force:
            skipped += 1
            continue
        print(f"  {w.ticker} {w.mode} {w.start:%Y-%m-%d}..{w.end:%Y-%m-%d}", flush=True)
        try:
            payload = fetch_window(w)
            save_window(w, payload)
            fetched += 1
        except (requests.HTTPError, requests.Timeout, RuntimeError) as e:
            print(f"    failed: {e}", file=sys.stderr)
            failed += 1
        time.sleep(args.sleep)

    print(f"Done. fetched={fetched} skipped(existing)={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

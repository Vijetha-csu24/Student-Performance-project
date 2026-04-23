"""Roll cached GDELT JSON into a daily per-ticker sentiment parquet.

Merges TimelineTone (average article tone) and TimelineVolRaw (matching article
count) into a single daily frame:

    ticker, date, gdelt_tone, gdelt_article_count

Writes to data/processed/gdelt_sentiment_daily.parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/gdelt")
OUT_PATH = Path("data/processed/gdelt_sentiment_daily.parquet")


def _flatten_timeline(payload: dict) -> pd.DataFrame:
    """GDELT TimelineTone / TimelineVolRaw payloads nest a 'timeline' list
    whose first entry holds 'data' = [{date, value}, ...]."""
    timeline = payload.get("timeline") or []
    if not timeline:
        return pd.DataFrame(columns=["date", "value"])
    data = timeline[0].get("data") or []
    if not data:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(data)
    # GDELT dates are "YYYYMMDDHHMMSS"; truncate to daily.
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S", errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]]


def _collect_mode(ticker_dir: Path, mode_prefix: str) -> pd.DataFrame:
    frames = []
    for path in sorted(ticker_dir.glob(f"{mode_prefix}_*.json")):
        payload = json.loads(path.read_text())
        frames.append(_flatten_timeline(payload))
    if not frames:
        return pd.DataFrame(columns=["date", "value"])
    out = pd.concat(frames, ignore_index=True)
    # Multiple overlapping windows could double-count; take mean per date.
    return out.groupby("date", as_index=False)["value"].mean()


def build_ticker_frame(ticker: str) -> pd.DataFrame:
    ticker_dir = RAW_DIR / ticker
    if not ticker_dir.exists():
        return pd.DataFrame()
    tone = _collect_mode(ticker_dir, "timelinetone").rename(columns={"value": "gdelt_tone"})
    vol = _collect_mode(ticker_dir, "timelinevolraw").rename(columns={"value": "gdelt_article_count"})
    if tone.empty and vol.empty:
        return pd.DataFrame()
    merged = tone.merge(vol, on="date", how="outer")
    merged.insert(0, "ticker", ticker)
    return merged.sort_values("date").reset_index(drop=True)


def discover_tickers() -> list[str]:
    if not RAW_DIR.exists():
        return []
    return sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", type=str, default=None)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tickers = (
        [t.strip().upper() for t in args.tickers.split(",")]
        if args.tickers
        else discover_tickers()
    )
    if not tickers:
        print(f"No cached tickers under {RAW_DIR}. Run ingest_gdelt first.")
        return 1

    frames = []
    for t in tickers:
        df = build_ticker_frame(t)
        print(f"  {t}: {len(df)} daily rows")
        if not df.empty:
            frames.append(df)

    if not frames:
        print("No data; nothing written.")
        return 1

    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

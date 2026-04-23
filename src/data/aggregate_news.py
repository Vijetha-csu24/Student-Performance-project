"""Aggregate cached Alpha Vantage news JSON into daily per-ticker sentiment features.

Reads data/raw/news/{ticker}/*.json, filters ticker_sentiment entries for the
target ticker, and emits a tidy parquet at data/processed/news_sentiment_daily.parquet
with one row per (ticker, date).

Columns:
    ticker, date, article_count, mean_sentiment, sentiment_std,
    mean_relevance, pos_share, neg_share

Run:
    python -m src.data.aggregate_news
    python -m src.data.aggregate_news --tickers NVDA,AMD
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/news")
OUT_PATH = Path("data/processed/news_sentiment_daily.parquet")

# Alpha Vantage sentiment label thresholds (per their docs).
POS_THRESHOLD = 0.15
NEG_THRESHOLD = -0.15


def load_articles(ticker: str) -> pd.DataFrame:
    """Return one row per (article, target-ticker-mention) for this ticker."""
    rows: list[dict] = []
    ticker_dir = RAW_DIR / ticker
    if not ticker_dir.exists():
        return pd.DataFrame(rows)

    for path in sorted(ticker_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        feed = payload.get("feed") or []
        for art in feed:
            time_published = art.get("time_published")
            if not time_published:
                continue
            # Pick the ticker_sentiment record matching our target ticker.
            ts_match = next(
                (ts for ts in art.get("ticker_sentiment", []) if ts.get("ticker") == ticker),
                None,
            )
            if ts_match is None:
                continue
            try:
                score = float(ts_match["ticker_sentiment_score"])
                relevance = float(ts_match["relevance_score"])
            except (KeyError, TypeError, ValueError):
                continue
            # time_published is "YYYYMMDDTHHMMSS" (UTC).
            ts = pd.to_datetime(time_published, format="%Y%m%dT%H%M%S", utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "datetime": ts,
                    "date": ts.date(),
                    "sentiment": score,
                    "relevance": relevance,
                }
            )
    return pd.DataFrame(rows)


def daily_aggregate(articles: pd.DataFrame) -> pd.DataFrame:
    if articles.empty:
        return articles
    grouped = articles.groupby(["ticker", "date"], as_index=False).agg(
        article_count=("sentiment", "size"),
        mean_sentiment=("sentiment", "mean"),
        sentiment_std=("sentiment", "std"),
        mean_relevance=("relevance", "mean"),
        pos_share=("sentiment", lambda s: (s > POS_THRESHOLD).mean()),
        neg_share=("sentiment", lambda s: (s < NEG_THRESHOLD).mean()),
    )
    grouped["date"] = pd.to_datetime(grouped["date"])
    return grouped.sort_values(["ticker", "date"]).reset_index(drop=True)


def discover_tickers() -> list[str]:
    if not RAW_DIR.exists():
        return []
    return sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", type=str, default=None, help="Comma-separated; default = all cached")
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
        print(f"No cached tickers under {RAW_DIR}. Run ingest_news first.")
        return 1

    frames = []
    for t in tickers:
        articles = load_articles(t)
        daily = daily_aggregate(articles)
        print(f"  {t}: {len(articles)} article-mentions -> {len(daily)} daily rows")
        if not daily.empty:
            frames.append(daily)

    if not frames:
        print("No daily rows produced; nothing written.")
        return 1

    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

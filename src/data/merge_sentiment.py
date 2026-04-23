"""Merge GDELT + Alpha Vantage daily sentiment into a single feature table.

Outer-joins the two sources per (ticker, date). Either source may be missing
(e.g. AV not yet ingested, or GDELT has no matching articles that day).
Downstream feature builders decide how to impute.

Output: data/processed/sentiment_daily.parquet

Columns:
    ticker, date,
    av_article_count, av_mean_sentiment, av_sentiment_std,
    av_mean_relevance, av_pos_share, av_neg_share,
    gdelt_tone, gdelt_article_count
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

AV_PATH = Path("data/processed/news_sentiment_daily.parquet")
GDELT_PATH = Path("data/processed/gdelt_sentiment_daily.parquet")
OUT_PATH = Path("data/processed/sentiment_daily.parquet")

AV_RENAME = {
    "article_count": "av_article_count",
    "mean_sentiment": "av_mean_sentiment",
    "sentiment_std": "av_sentiment_std",
    "mean_relevance": "av_mean_relevance",
    "pos_share": "av_pos_share",
    "neg_share": "av_neg_share",
}


def _load(path: Path, rename: dict | None = None) -> pd.DataFrame:
    if not path.exists():
        print(f"  (missing) {path} -- skipping")
        return pd.DataFrame(columns=["ticker", "date"])
    df = pd.read_parquet(path)
    if rename:
        df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"])
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--av", type=Path, default=AV_PATH)
    p.add_argument("--gdelt", type=Path, default=GDELT_PATH)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    av = _load(args.av, AV_RENAME)
    gd = _load(args.gdelt)

    if av.empty and gd.empty:
        print("Both sources missing; nothing to merge.")
        return 1

    merged = av.merge(gd, on=["ticker", "date"], how="outer").sort_values(["ticker", "date"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)

    coverage = merged.groupby("ticker").agg(
        rows=("date", "size"),
        has_av=("av_mean_sentiment", lambda s: s.notna().sum()),
        has_gdelt=("gdelt_tone", lambda s: s.notna().sum()) if "gdelt_tone" in merged else ("date", "size"),
    )
    print(f"Wrote {len(merged)} rows to {args.out}")
    print(coverage.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

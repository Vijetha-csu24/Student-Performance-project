import pandas as pd

SRC = "semiconductor_project_merged_dataset.csv"
CLEANED = "semiconductor_project_cleaned.csv"
TRAIN = "semiconductor_project_train.csv"
TEST = "semiconductor_project_test.csv"

df = pd.read_csv(SRC)
df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

before = len(df)
df = df.dropna(subset=["target_next_close"])
dropped_target = before - len(df)

tech_cols = ["return_1d", "ma_7", "ma_30", "volatility_7", "rsi_14"]
before = len(df)
df = df.dropna(subset=tech_cols)
dropped_tech = before - len(df)

df.to_csv(CLEANED, index=False)

train_parts, test_parts = [], []
for tkr, g in df.groupby("ticker"):
    g = g.sort_values("date")
    cut = int(len(g) * 0.8)
    train_parts.append(g.iloc[:cut])
    test_parts.append(g.iloc[cut:])

train = pd.concat(train_parts).sort_values(["ticker", "date"]).reset_index(drop=True)
test = pd.concat(test_parts).sort_values(["ticker", "date"]).reset_index(drop=True)
train.to_csv(TRAIN, index=False)
test.to_csv(TEST, index=False)

print(f"dropped {dropped_target} rows for NaN target_next_close")
print(f"dropped {dropped_tech} rows for NaN technicals (warm-up period)")
print(f"cleaned: {len(df):,} rows  ->  {CLEANED}")
print(f"train:   {len(train):,} rows  ->  {TRAIN}")
print(f"test:    {len(test):,} rows   ->  {TEST}")
print("\nper-ticker split boundaries:")
for tkr in sorted(df["ticker"].unique()):
    tr = train[train["ticker"] == tkr]
    te = test[test["ticker"] == tkr]
    print(f"  {tkr}: train {tr['date'].min().date()} -> {tr['date'].max().date()} "
          f"({len(tr)})  |  test {te['date'].min().date()} -> {te['date'].max().date()} ({len(te)})")
print(f"\nremaining NaNs in cleaned dataset: {df.isna().sum().sum()}")

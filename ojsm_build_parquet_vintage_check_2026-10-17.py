"""Combine the daily Savant CSVs into one parquet per season, one season at a time (memory-light)."""
import glob, os, hashlib, json
import pandas as pd
D = "/home/claude/work/statcast_pull/daily"; O = "/home/claude/work/statcast_pull/season"
os.makedirs(O, exist_ok=True)
info = {}
for y in (2024, 2025, 2026):
    files = sorted(f for f in glob.glob(f"{D}/{y}-*.csv") if os.path.getsize(f) > 200)
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True); del dfs
    df = df[df["game_type"] == "R"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    for c in ("batter", "pitcher", "game_pk", "game_year", "at_bat_number", "pitch_number"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    n0 = len(df)
    df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    p = f"{O}/statcast_{y}.parquet"
    df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).to_parquet(p, index=False)
    info[str(y)] = {"daily_files": len(files), "rows_before_dedupe": n0, "rows": len(df), "games": int(df["game_pk"].nunique()),
                    "first": str(df["game_date"].min().date()), "last": str(df["game_date"].max().date()),
                    "sha16": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]}
    print(y, info[str(y)], flush=True)
    del df
json.dump(info, open(f"{O}/build_info.json", "w"), indent=1)

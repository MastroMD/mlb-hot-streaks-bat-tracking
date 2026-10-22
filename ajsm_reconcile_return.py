"""Post hoc reconciliation (not prespecified): why does F4b find a bat-speed deficit after oblique and hand/wrist
returns when a companion analysis of the same public data (Paper 2, within-season recovery trajectory) found none?

The companion design: raw Statcast bat speed on every tracked swing, the first 30 days after activation, compared
with the hitter's own games in the 150 days before placement (which can reach into the previous season), with no
concurrent controls. F4b: park-season and week-adjusted bat speed on competitive swings, games 1-10 after return,
compared with the same-season baseline (games more than 30 days before placement, else the previous season) and with
up to 5 control hitters given the same pseudo-placement and pseudo-return dates.

This script moves from the companion design to F4b one element at a time on a common set of stints.

python3 ajsm_reconcile_return.py DATA_DIR IL_FILE OUT_JSON
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common3 as C            # noqa: E402
import hot_streak_study as H   # noqa: E402

DATA, ILF, OUT = sys.argv[1:4]
SEED = int(os.environ.get("RECON_SEED", "606"))
FAM = {"O": "oblique", "H": "hamstring", "W": "hand_wrist", "X": "other"}
COLS = ["game_date", "game_pk", "game_type", "game_year", "at_bat_number", "pitch_number", "batter", "home_team",
        "description", "events", "type", "zone", "release_speed", "launch_speed", "launch_angle", "launch_speed_angle",
        "estimated_woba_using_speedangle", "woba_value", "woba_denom", "bat_speed", "swing_length", "stand", "p_throws"]
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def load(data_dir):
    import glob
    import re

    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    files = [f for f in files if not re.findall(r"20\d\d", os.path.basename(f))
             or any(y in ("2024", "2025", "2026") for y in re.findall(r"20\d\d", os.path.basename(f)))]
    dfs = []
    for f in files:
        have = set(pq.ParquetFile(f).schema_arrow.names)
        dfs.append(pd.read_parquet(f, columns=[c for c in COLS if c in have]))
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["game_type"] == "R") & pd.to_numeric(df["game_year"], errors="coerce").between(2024, 2026)].copy()
    for c in df.columns:
        dt_ = str(df[c].dtype)
        if dt_ in ("Int8", "Int16", "Int32", "Int64", "Float32", "Float64", "UInt8", "UInt16", "UInt32", "UInt64"):
            df[c] = df[c].astype("float64")
        elif dt_ in ("string", "str") or dt_.startswith("string"):
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    for c in ("batter", "game_pk", "game_year"):
        df[c] = df[c].astype("int64")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    return df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def il_events(path):
    P, A = [], []
    for line in open(path):
        b, ev = line.strip().split(":")
        for e in ev.split():
            d_ = pd.Timestamp("20" + e[:6])
            (P if e[6] == "P" else A).append({"batter": int(b), "t": d_, "fam": FAM.get(e[7]) if e[6] == "P" else None})
    P = pd.DataFrame(P).rename(columns={"t": "start"}).sort_values(["batter", "start"]).reset_index(drop=True)
    A = pd.DataFrame(A).rename(columns={"t": "end"}).sort_values(["batter", "end"]).reset_index(drop=True)
    ret = []
    for b, g in P.groupby("batter", sort=False):
        a = A.loc[A["batter"] == b, "end"].tolist()
        st = g["start"].tolist()
        for k, s in enumerate(st):
            lim = st[k + 1] if k + 1 < len(st) else pd.Timestamp("2100-01-01")
            c = [x for x in a if s <= x <= lim]
            ret.append(c[0] if c else pd.NaT)
    P["ret"] = ret
    return P, A


raw = load(DATA)
raw["bs_raw"] = pd.to_numeric(raw["bat_speed"], errors="coerce")
d = C.adjust(raw)
del raw
d = H.pitch_features(d)
tr = pd.to_numeric(d["bat_speed"], errors="coerce").notna() & pd.to_numeric(d["swing_length"], errors="coerce").notna()
d["compall"] = tr.astype(np.int8)
d["rawall_num"] = np.where(tr, d["bs_raw"], 0.0)
d["adjall_num"] = np.where(tr, pd.to_numeric(d["bat_speed"], errors="coerce"), 0.0)
d["rawcomp_num"] = np.where(d["comp"] == 1, d["bs_raw"], 0.0)
KEYS = ["batter", "game_year", "game_date", "game_pk"]
S_ = ["pa", "comp", "bs_num", "compall", "rawall_num", "adjall_num", "rawcomp_num"]
bg = d.groupby(KEYS, sort=False)[S_].sum().reset_index()
del d
bg = bg[bg["pa"] > 0].sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)
log("batter-games", len(bg))

# per-batter arrays over all seasons
BAT = {}
for b, idx in bg.groupby("batter").indices.items():
    x = bg.iloc[idx]
    BAT[b] = {"t": x["game_date"].to_numpy().astype("datetime64[D]"), "y": x["game_year"].to_numpy(),
              **{c: x[c].to_numpy(float) for c in S_}}
MET = {"raw_all": ("rawall_num", "compall"), "adj_comp": ("bs_num", "comp"), "raw_comp": ("rawcomp_num", "comp"),
       "adj_all": ("adjall_num", "compall")}
D64 = lambda ts: np.datetime64(pd.Timestamp(ts), "D")


def sums(g, m):
    return {c: g[c][m].sum() for c in S_}


def base150(g, S):
    m = (g["t"] >= D64(S - pd.Timedelta(days=150))) & (g["t"] < D64(S))
    s = sums(g, m)
    return s if s["compall"] >= 50 else None


def base_ours(g, y, S):
    m = (g["y"] == y) & (g["t"] < D64(S - pd.Timedelta(days=30)))
    s = sums(g, m)
    if s["comp"] >= 90:
        return s
    m = g["y"] == y - 1
    s = sums(g, m)
    return s if s["comp"] >= 90 else None


def post30(g, R, stop):
    m = (g["t"] >= D64(R)) & (g["t"] < D64(min(R + pd.Timedelta(days=30), stop)))
    return sums(g, m)


def post10(g, R, stop):
    idx = np.flatnonzero((g["t"] >= D64(R)) & (g["t"] < D64(stop)))[:10]
    return {c: g[c][idx].sum() for c in S_}


def value(post, base, met):
    n, dn = MET[met]
    if base is None or post[dn] < 5 or base[dn] <= 0:
        return np.nan
    return post[n] / post[dn] - base[n] / base[dn]


def all_values(b, y, S, R, stop):
    g = BAT.get(b)
    if g is None:
        return None
    b1, b2 = base150(g, S), base_ours(g, y, S)
    p30, p10 = post30(g, R, stop), post10(g, R, stop)
    out = {}
    for met in MET:
        for bn, bb in (("b150", b1), ("bours", b2)):
            for pn, pp in (("p30", p30), ("g10", p10)):
                out[f"{met}|{bn}|{pn}"] = value(pp, bb, met)
    return out


P, A = il_events(ILF)
Pb = {b: g["start"].to_numpy() for b, g in P.groupby("batter")}
Ab = {b: g["end"].to_numpy() for b, g in A.groupby("batter")}
COVERED = set(bg.groupby(["batter", "game_year"])["pa"].sum().loc[lambda x: x >= 300].index.get_level_values(0))
hitters_by_year = {y: [b for b in v if b in COVERED] for y, v in bg.groupby("game_year")["batter"].unique().to_dict().items()}
rng = np.random.default_rng(SEED)
rows = []
for k, p in enumerate(P.itertuples()):
    y = p.start.year
    if pd.isna(p.ret) or p.ret.year != y:
        continue
    later = P[(P["batter"] == p.batter) & (P["start"] > p.start)]["start"]
    stop = later.min() if len(later) else pd.Timestamp(f"{y}-12-31")
    v = all_values(p.batter, y, p.start, p.ret, stop)
    if v is None:
        continue
    elig = []
    for b in hitters_by_year.get(y, []):
        if b == p.batter:
            continue
        pl = Pb.get(b, np.array([], dtype="datetime64[ns]"))
        if ((pl >= np.datetime64(p.start - pd.Timedelta(days=60))) & (pl <= np.datetime64(p.ret + pd.Timedelta(days=60)))).any():
            continue
        ac = Ab.get(b, np.array([], dtype="datetime64[ns]"))
        if ((ac <= np.datetime64(p.start)) & (ac >= np.datetime64(p.start - pd.Timedelta(days=45)))).any():
            continue
        elig.append(b)
    # every eligible control (the F4b sensitivity), so that no step carries the Monte Carlo error of a 5-control draw
    ctrl = []
    for b in elig:
        cv = all_values(b, y, p.start, p.ret, pd.Timestamp(f"{y}-12-31"))
        if cv is not None and any(np.isfinite(v_) for v_ in cv.values()):
            ctrl.append(cv)
    row = {"batter": p.batter, "fam": p.fam, "y": y, "stint": k, **{f"inj|{key}": val for key, val in v.items()}}
    for key in v:
        cvals = [c[key] for c in ctrl if np.isfinite(c[key])]
        row[f"ctl|{key}"] = float(np.mean(cvals)) if cvals else np.nan
    rows.append(row)
R = pd.DataFrame(rows)
log("stints", len(R))


def est(y, g):
    y, g = np.asarray(y, float), np.asarray(g)
    ok = np.isfinite(y)
    f = sm.OLS(y[ok], np.ones(ok.sum())).fit(cov_type="cluster", cov_kwds={"groups": g[ok]})
    return {"mean": float(f.params[0]), "lo": float(f.conf_int()[0][0]), "hi": float(f.conf_int()[0][1]),
            "p": float(f.pvalues[0]), "n": int(ok.sum())}


STEPS = [("1 companion design: unadjusted, every tracked swing, 150-day baseline, first 30 days, no controls", "raw_all|b150|p30", False),
         ("2 step 1 with matched controls", "raw_all|b150|p30", True),
         ("3 step 1 with games 1-10 in place of 30 days", "raw_all|b150|g10", False),
         ("4 step 3 with this study's baseline", "raw_all|bours|g10", False),
         ("5 step 4 with ballpark-week adjustment and competitive swings (injured hitters only)", "adj_comp|bours|g10", False),
         ("6 this study's design: step 5 with matched controls", "adj_comp|bours|g10", True),
         ("7 this study's design over the first 30 days", "adj_comp|bours|p30", True),
         ("8 companion metric (unadjusted, every tracked swing) in this study's design", "raw_all|bours|g10", True)]
cols = [f"inj|{s[1]}" for s in STEPS] + [f"ctl|{s[1]}" for s in STEPS if s[2]]
common = R.dropna(subset=list(dict.fromkeys(cols)))
res = {"note": "post hoc reconciliation, not prespecified; corrected IL file; every eligible control; common sample of stints with every variant",
       "n_stints_any": int(len(R)), "n_stints_common": int(len(common)), "by_family_common": common["fam"].value_counts().to_dict(),
       "steps": {}}
for grp in ("all", "oblique", "hand_wrist", "hamstring", "other"):
    D = common if grp == "all" else common[common["fam"] == grp]
    res["steps"][grp] = {}
    for lab, key, ctl in STEPS:
        yv = D[f"inj|{key}"] - (D[f"ctl|{key}"] if ctl else 0)
        res["steps"][grp][lab] = est(yv, D["batter"])
    # companion cohort window: activations in 2024-2025 only
    D2 = D[D["y"] <= 2025]
    res["steps"][grp]["1b companion design, 2024-2025 activations only"] = est(D2["inj|raw_all|b150|p30"], D2["batter"])
    # controls' own change under the companion design (drift that the no-control design attributes to recovery)
    res["steps"][grp]["control change under the companion design"] = est(D["ctl|raw_all|b150|p30"], D["batter"])
# each step on its own maximal sample (stints for which that variant, and its controls if used, exist)
res["steps_max_sample"] = {}
for grp in ("all", "oblique", "hand_wrist", "hamstring", "other"):
    D = R if grp == "all" else R[R["fam"] == grp]
    res["steps_max_sample"][grp] = {}
    for lab, key, ctl in STEPS:
        yv = D[f"inj|{key}"] - (D[f"ctl|{key}"] if ctl else 0)
        res["steps_max_sample"][grp][lab] = est(yv, D["batter"])
# why stints drop out of the common sample
miss = {c: int(R[c].isna().sum()) for c in list(dict.fromkeys(cols))}
res["missing_by_variant"] = miss
R.to_csv(OUT.replace(".json", "_stints.csv"), index=False)
json.dump(res, open(OUT, "w"), indent=1)
for grp in ("all", "oblique", "hand_wrist", "hamstring"):
    print("==", grp)
    for lab, r in res["steps"][grp].items():
        m = res["steps_max_sample"][grp].get(lab)
        mx = f" | max sample {m['mean']:+.2f} ({m['lo']:+.2f} to {m['hi']:+.2f}) n={m['n']:4d}" if m else ""
        print(f"  {r['mean']:+.2f} ({r['lo']:+.2f} to {r['hi']:+.2f}) n={r['n']:4d}{mx}  {lab}")
print(json.dumps(miss, indent=0))
log("wrote", OUT)

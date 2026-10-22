"""Season-ending placements and stint lengths of Table 1, with the end of the regular season and the games played
(post hoc, third independent check of 2026-10-20).

python3 ojsm_season_ending.py IL_FILE CODED_CSV FG_POSITION_XLSX FG_2026_CSV STATCAST_DIR AJSM_SEASON_JSON OUT_JSON

ajsm_season.py takes the first activation after a placement and before the hitter's next placement; the placement is
season-ending when there is none or it is on or after October 1, and otherwise its stint lasts until that activation.
This misclassifies three kinds of placement. (1) Clubs activate injured hitters in the days after the last regular-season
game, before October 1; those placements ended the season, and their stints ran only to an end-of-season roster move.
(2) The public transaction records omit some activations (recorded as from the All-Star reserve list, without naming a
list, or as a recall from a minor-league club, or not recorded at all), so a hitter who played again was counted as out
for the season. (3) A placement for a different injury made with no activation since the previous one leaves that previous
placement without an activation, although the hitter was activated later in the season.
Here a placement is season-ending when no activation is recorded from the placement through the last regular-season game
of its season (any activation, also after a later placement) and the hitter played no further regular-season game that
season (Statcast pitch files); its stint lasts until the first activation before the next placement when that activation
is on or before the last regular-season game, and has no length otherwise. Everything else follows ajsm_season.py: the
cohort of hitters with a season of 330 or more PA in 2024-2026, and region and mechanism matched on hitter and date. The
script stops unless the values it recomputes with ajsm_season.py's rule equal those of AJSM_SEASON_JSON (ajsm_season.py
run on the same IL file with CODED_CSV).
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

IL, CODED, XL, F26, SC, AJ, OUT = sys.argv[1:8]
FAM = {"O": "oblique", "H": "hamstring", "W": "hand_wrist", "X": "other"}
sha16 = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]

# regular-season games played (Statcast) and the last game of each season
files = [f for f in sorted(glob.glob(os.path.join(SC, "*.parquet"))) if any(y in os.path.basename(f) for y in ("2024", "2025", "2026"))]
g = pd.concat([pd.read_parquet(f, columns=["game_date", "game_type", "batter", "game_pk"]) for f in files], ignore_index=True)
g = g[g["game_type"] == "R"].drop_duplicates(["batter", "game_pk"])
g["game_date"] = pd.to_datetime(g["game_date"])
g["batter"] = g["batter"].astype(int)
GD = {b: np.sort(x["game_date"].to_numpy()) for b, x in g.groupby("batter")}
LAST = {int(y): d for y, d in g.groupby(g["game_date"].dt.year)["game_date"].max().items()}

# IL events
P, A = [], []
for line in open(IL):
    b, ev = line.strip().split(":")
    for e in ev.split():
        d = pd.Timestamp("20" + e[:6])
        if e[6] == "P":
            P.append({"batter": int(b), "start": d, "fam": FAM[e[7]], "code": e})
        else:
            A.append({"batter": int(b), "end": d})
P = pd.DataFrame(P).sort_values(["batter", "start"]).reset_index(drop=True)
A = pd.DataFrame(A).sort_values(["batter", "end"]).reset_index(drop=True)
P["year"] = P["start"].dt.year
act, act_season, nxt = [], [], []
for b, x in P.groupby("batter", sort=False):
    a = A.loc[A["batter"] == b, "end"].tolist()
    st = x["start"].tolist()
    for k, s in enumerate(st):
        lim = st[k + 1] if k + 1 < len(st) else pd.Timestamp("2100-01-01")
        c = [y for y in a if s <= y <= lim]
        act.append(c[0] if c else pd.NaT)
        act_season.append(any(s <= y <= LAST[s.year] for y in a))
        nxt.append(lim)
P["activation"], P["activated_in_season"], P["next_placement"] = act, act_season, nxt
# ajsm_season.py's rule
P["se_old"] = P["activation"].isna() | (P["activation"] >= pd.to_datetime(P["year"].astype(str) + "-10-01"))
P["stint_old"] = (P["activation"] - P["start"]).dt.days
# this rule
P["games_after"] = [int(((t := GD.get(b, np.array([], dtype="datetime64[ns]"))) > np.datetime64(s)).sum()
                        - (t > np.datetime64(LAST[s.year])).sum()) for b, s in zip(P["batter"], P["start"])]
P["season_ending"] = (~P["activated_in_season"]) & (P["games_after"] == 0)
# games played after the placement and before the next placement record, in the same season
P["games_before_next"] = [int(((t := GD.get(b, np.array([], dtype="datetime64[ns]"))) > np.datetime64(s)).sum()
                              - (t >= np.datetime64(min(n_, LAST[s.year] + pd.Timedelta(days=1)))).sum())
                          for b, s, n_ in zip(P["batter"], P["start"], P["next_placement"])]
P["stint_days"] = np.where(P["activation"].notna() & (P["activation"] <= P["year"].map(LAST)), P["stint_old"], np.nan)

# cohort (ajsm_season.py): hitters with a season of 330 or more PA in 2024-2026
X = pd.read_excel(XL)
X = X[X["Season"].between(2021, 2025)].dropna(subset=["MLBAMID"]).copy()
X["MLBAMID"] = X["MLBAMID"].astype(int)
Z = pd.read_csv(F26).dropna(subset=["MLBAMID"])
Z["MLBAMID"] = Z["MLBAMID"].astype(int)
S = pd.concat([X[["Season", "MLBAMID", "PA"]], Z[["Season", "MLBAMID", "PA"]]], ignore_index=True)
maxpa = S[S["Season"].between(2024, 2026)].groupby("MLBAMID")["PA"].max()
U = set(maxpa.index[maxpa >= 330])
PU = P[P["batter"].isin(U)].copy()
cd = pd.read_csv(CODED, parse_dates=["day"])
PU = PU.merge(cd[["batter", "day", "region", "mechanism", "impact"]].drop_duplicates(["batter", "day"]),
              left_on=["batter", "start"], right_on=["batter", "day"], how="left")


def stats(x, se, days):
    s = x.loc[x[days].notna(), days] if days == "stint_days" else x.loc[~x["se_old"], days]
    return {"placements": int(len(x)), "season_ending_n": int(x[se].sum()), "season_ending_share": float(x[se].mean()),
            "stint_median": float(s.median()) if len(s) else None, "stint_q1": float(s.quantile(.25)) if len(s) else None,
            "stint_q3": float(s.quantile(.75)) if len(s) else None, "stint_n": int(len(s))}


# the old rule must reproduce ajsm_season.py (AJSM_SEASON_JSON)
F3 = json.load(open(AJ))["F3"]
o = stats(PU, "se_old", "stint_old")
assert len(PU) == F3["placements"] and abs(o["season_ending_share"] - F3["season_ending_share_all"]) < 1e-12
assert (o["stint_median"], o["stint_q1"], o["stint_q3"], o["stint_n"]) == tuple(F3["stint_days_all"][k] for k in ("median", "q1", "q3", "n"))
for col in ("region", "mechanism", "impact"):
    for k, x in PU.groupby(col):
        r, f = stats(x, "se_old", "stint_old"), F3[f"by_{col}"][str(k)]
        assert abs(r["season_ending_share"] - f["season_ending_share"]) < 1e-12 and r["stint_median"] == f["stint_median"], (col, k)
        assert r["stint_q1"] == f["stint_q1"] and r["stint_q3"] == f["stint_q3"], (col, k)


def why(r):
    if r.se_old and not r.season_ending:
        if pd.isna(r.activation) and r.games_before_next > 0:
            return "played again before the next placement record, no activation recorded"
        return "no activation before the next placement record (the hitter did not play); activated later that season"
    if r.season_ending and not r.se_old:
        return "activated after the last regular-season game, before October 1"
    return None


PU["changed"] = [why(r) for r in PU.itertuples()]
PU["stint_changed"] = PU["stint_old"].where(~PU["se_old"]).notna() & PU["stint_days"].isna()
out = {"note": " ".join(__doc__.split("\n\n")[1].split()),
       "inputs": {"il": sha16(IL), "coded": sha16(CODED), "statcast_files": [os.path.basename(f) for f in files]},
       "last_regular_season_game": {y: str(d.date()) for y, d in LAST.items()},
       "placements": int(len(PU)),
       "all": stats(PU, "season_ending", "stint_days"), "all_ajsm_season_rule": o,
       "changed_by_reason": {k: int(v) for k, v in PU["changed"].value_counts().items()},
       "stints_no_longer_counted": int(PU["stint_changed"].sum()),
       "mid_season_activation_without_further_games": int((PU["activated_in_season"] & (PU["games_after"] == 0) & ~PU["se_old"]).sum()),
       "changed": PU.loc[PU["changed"].notna(), ["batter", "code", "fam", "region", "mechanism", "games_after", "games_before_next", "activation", "changed"]].astype(str).to_dict(orient="records"),
       "by_family": {f: stats(x, "season_ending", "stint_days") for f, x in PU.groupby("fam")}}
for col in ("region", "mechanism", "impact"):
    out[f"by_{col}"] = {str(k): stats(x, "season_ending", "stint_days") for k, x in PU.groupby(col)}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("placements", "all", "all_ajsm_season_rule", "changed_by_reason", "stints_no_longer_counted",
                                      "mid_season_activation_without_further_games", "last_regular_season_game")}, indent=1))

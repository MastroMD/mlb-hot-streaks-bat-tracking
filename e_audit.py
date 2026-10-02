"""Checks added after the pre-submission accuracy audit of the paper (2026-10-02; descriptive, exploratory).

python3 e_audit.py DATA_DIR PUBLIC_DIR OUT3_DIR    -> OUT3_DIR/e_audit.json
  OUT3_DIR must hold the analysis3.py caches (_cache/bg_adj1_comp1.pkl, _cache/lineup.pkl).

F1  Lineup inputs (A7). For each hitter in a decision, A7 takes talent, swing form, last-50-PA wOBA and the forecast
    residual from his most recent forecast row on or before the decision date. This records how old that row is,
    how often it comes from an earlier season, and the rule values on decisions where both hitters' rows are from the
    decision's season (and, separately, at most 7 days old). The A7 summary values are re-checked against
    results/a7.json first, so the decisions are the published ones.
F2  Simulator. Expected wOBA per standard deviation of the injected state in simulate.py's plate-appearance model
    (latent quality +0.012 per SD, bat speed +0.8 mph per SD, both feeding exit velocity), by common random numbers.
F3  Plate-appearance count in the batter-game panel, against paper_counts.n_pa.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
data, pub, out3 = sys.argv[1:4]
sys.argv = ["analysis3.py", data, pub, out3, "NONE"]
import analysis3 as A  # noqa: E402

OUT = {"meta": {"script": "e_audit.py", "status": "exploratory, added after the accuracy audit"}}

# ------------------------------------------------------------------------------------------------ F1 lineup inputs
CAP = []
_merge_asof = pd.merge_asof


def _capture(left, right, *a, **k):
    if "talent" in right.columns:                      # the forecast-row lookup in A7_core
        right = right.assign(fc_date=right["game_date"])
        res = _merge_asof(left, right, *a, **k)
        CAP.append(res)
        return res
    return _merge_asof(left, right, *a, **k)


pd.merge_asof = _capture
ref = json.load(open(os.path.join(HERE, "results", "a7.json")))
tor = lambda x: x / A.WOBA_SCALE * 50 / 1000
F1 = {}
for rule, key in (("plan", "prespecified"), ("mutual", "refined_exploratory")):
    CAP.clear()
    res = json.loads(json.dumps(A.A7_core(rule), default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    for v in ("b", "c0", "c"):
        r0 = ref[key][f"option_{v}_vs_strict_platoon_runs_per50"]["mean"]
        r1 = res[f"option_{v}_vs_strict_platoon_runs_per50"]["mean"]
        if abs(r0 - r1) > 1e-9:
            sys.exit(f"A7 re-run does not reproduce a7.json ({key} {v}: {r0} vs {r1})")
    Dd = A.A7_ROWS[rule].copy().reset_index(drop=True)
    rows = []
    for i in range(0, len(CAP), 2):
        L, R = CAP[i], CAP[i + 1]
        ok = (L["talent"].notna() & R["talent"].notna()).to_numpy()
        for j in np.flatnonzero(ok):
            rows.append((L["game_date"].iloc[j], L["fc_date"].iloc[j], R["fc_date"].iloc[j]))
    if len(rows) != len(Dd):
        sys.exit(f"captured {len(rows)} decisions, A7 has {len(Dd)}")
    G = pd.DataFrame(rows, columns=["date", "fc_L", "fc_R"])
    if not (pd.to_datetime(G["date"]).to_numpy() == pd.to_datetime(Dd["date"]).to_numpy()).all():
        sys.exit("decision dates do not align")
    d0 = pd.to_datetime(G["date"])
    lagL, lagR = (d0 - pd.to_datetime(G["fc_L"])).dt.days, (d0 - pd.to_datetime(G["fc_R"])).dt.days
    lag_old = np.maximum(lagL, lagR)
    prev = (pd.to_datetime(G["fc_L"]).dt.year < d0.dt.year) | (pd.to_datetime(G["fc_R"]).dt.year < d0.dt.year)
    lag_young = np.minimum(lagL, lagR)

    def summ(mask):
        X = Dd[mask]
        s = {"n_decisions": int(mask.sum()), "share_of_all": float(mask.mean())}
        for v in ("b", "c0", "c"):
            s[f"runs_{v}"] = float(tor(X[f"val_{v}"]).mean())
            s[f"share_{v}_ne_a"] = float((X[v] != X["a"]).mean())
        s["share_c_ne_c0"] = float((X["c"] != X["c0"]).mean())
        s["runs_c_minus_c0"] = float(tor(X["val_c"] - X["val_c0"]).mean())
        return s

    same = ~prev
    fresh = same & (lag_old <= 7)
    F1[key] = {"n_decisions": int(len(Dd)),
               "share_either_row_previous_season": float(prev.mean()),
               "n_either_row_previous_season": int(prev.sum()),
               "lag_days_older_row_median_p90": [float(np.median(lag_old)), float(np.percentile(lag_old, 90))],
               "lag_days_younger_row_median": float(np.median(lag_young)),
               "share_older_row_over_7_days": float((lag_old > 7).mean()),
               "all": summ(np.ones(len(Dd), bool)),
               "same_season_rows": summ(same.to_numpy()),
               "same_season_rows_within_7_days": summ(fresh.to_numpy())}
pd.merge_asof = _merge_asof
OUT["F1_lineup_inputs"] = F1

# ------------------------------------------------------------------------------------------------ F2 simulator state
rng = np.random.default_rng(20261002)
N = 4_000_000
tal_w = rng.normal(0.320, 0.03, N)
tal_bs = rng.normal(72, 2.5, N)
stand_R = rng.random(N) < 0.6
p_R = rng.random(N) < 0.7
pq = rng.normal(0, 0.02, N)
same = stand_R == p_R
z_bs = rng.normal(0, 3, N)
z_ev = rng.normal(0, 13, N)
la = rng.normal(12, 25, N)


def expected(s):
    q = tal_w + 0.012 * s - 0.012 * same + pq
    pk = 0.22 - 0.4 * (q - 0.32)
    pbb = 0.085 + 0.2 * (q - 0.32)
    bs = tal_bs + 0.8 * s + z_bs
    ev = 88 + 1.0 * (bs - 72) + 150 * (q - 0.32) + z_ev
    xw = np.clip(0.37 + 0.02 * (ev - 88) - 0.0002 * (la - 18) ** 2, 0, 2)
    p_hit = np.minimum(xw / 1.4, 0.95)
    pbip = 1 - pk - pbb
    return float(np.mean(pbb * 0.69 + pbip * 1.2 * p_hit)), float(np.mean(pbb * 0.69 + pbip * xw))


E = {s: expected(s) for s in (-1.0, 0.0, 1.0)}
OUT["F2_simulator"] = {"n_pa_draws": N, "seed": 20261002,
                       "expected_woba": {str(s): v[0] for s, v in E.items()},
                       "expected_xwoba": {str(s): v[1] for s, v in E.items()},
                       "woba_pts_per_sd": 1000 * (E[1.0][0] - E[-1.0][0]) / 2,
                       "xwoba_pts_per_sd": 1000 * (E[1.0][1] - E[-1.0][1]) / 2,
                       "latent_quality_pts_per_sd": 12.0, "bat_speed_mph_per_sd": 0.8, "chase_pp_per_sd": 2.0}

# ------------------------------------------------------------------------------------------------ F3 PA count
bg, _ = pickle.load(open(os.path.join(out3, "_cache", "bg_adj1_comp1.pkl"), "rb"))
pc = json.load(open(os.path.join(HERE, "results", "paper_counts.json")))
OUT["F3_pa_count"] = {"panel_pa": int(bg["pa"].sum()), "paper_counts_n_pa": int(pc["n_pa"]),
                      "equal": int(bg["pa"].sum()) == int(pc["n_pa"])}

json.dump(OUT, open(os.path.join(out3, "e_audit.json"), "w"), indent=1)
print(json.dumps(OUT, indent=1))

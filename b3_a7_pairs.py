"""B3 (plan addendum D, PLAN_ADDENDUM_2026-10-02d.md): A7 per-pair decision table, spread of the player-specific
platoon splits, and the prespecified secondary comparison of the hitter's own regressed split with the league split.

python3 b3_a7_pairs.py DATA_DIR PUBLIC_DIR OUT3_DIR
  OUT3_DIR must hold the analysis3.py panel cache (_cache/bg_adj1_comp1.pkl); the lineup cache is built from DATA_DIR
  on first use. Writes OUT3_DIR/a7_pairs.csv and OUT3_DIR/b3.json. Exits 1 unless the re-run reproduces every summary
  value in results/a7.json to 1e-9.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
data, pub, out3 = sys.argv[1:4]
sys.argv = ["analysis3.py", data, pub, out3, "NONE"]
import analysis3 as A  # noqa: E402

ref = json.load(open(os.path.join(HERE, "results", "a7.json")))
tor = lambda x: x / A.WOBA_SCALE * 50 / 1000


def compare(a, b, path=""):
    """every number in the reference must be reproduced to 1e-9; lists and strings exactly"""
    bad = []
    if isinstance(a, dict):
        for k in a:
            bad += compare(a[k], b.get(k) if isinstance(b, dict) else None, f"{path}.{k}")
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return [path]
        for i, (x, y) in enumerate(zip(a, b)):
            bad += compare(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if b is None or abs(float(a) - float(b)) > 1e-9:
            bad.append(f"{path}: {a} vs {b}")
    elif a != b:
        bad.append(f"{path}: {a!r} vs {b!r}")
    return bad


B3 = {"plan": "PLAN_ADDENDUM_2026-10-02d.md", "rules": {}}
pair_rows = []
for rule, key in (("plan", "prespecified"), ("mutual", "refined_exploratory")):
    out = json.loads(json.dumps(A.A7_core(rule), default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    bad = compare(ref[key], out)
    if bad:
        print("A7 re-run does not reproduce a7.json:", bad[:10])
        sys.exit(1)
    Dd = A.A7_ROWS[rule].copy()
    for v in ("b", "c0", "c", "c0L"):
        Dd[f"runs_{v}"] = tor(Dd[f"val_{v}"])
    g = Dd.groupby(["team", "yr", "L", "R"], sort=True)
    P = pd.DataFrame({
        "n_decisions": g.size(),
        "split_L_pts": g["split_L"].mean(), "split_R_pts": g["split_R"].mean(),
        "share_b_ne_a": g.apply(lambda x: (x["b"] != x["a"]).mean()),
        "share_c0_ne_a": g.apply(lambda x: (x["c0"] != x["a"]).mean()),
        "share_c_ne_c0": g.apply(lambda x: (x["c"] != x["c0"]).mean()),
        "share_c0_ne_c0L": g.apply(lambda x: (x["c0"] != x["c0L"]).mean()),
        "runs_b": g["runs_b"].mean(), "runs_c0": g["runs_c0"].mean(), "runs_c": g["runs_c"].mean(),
        "runs_c0L": g["runs_c0L"].mean(),
    }).reset_index()
    P.insert(0, "rule", key)
    pair_rows.append(P)
    q = lambda s: [float(s.quantile(x)) for x in (0.1, 0.5, 0.9)]
    # spread of each paired hitter's own regressed split at decision time (one value per hitter-decision)
    spl = {"L": Dd["split_L"], "R": Dd["split_R"]}
    plat = json.load(open(os.path.join(HERE, "results", "results_adjusted.json")))["platoon"]
    lg = {"L": plat["LHB_league_split_pts"], "R": plat["RHB_league_split_pts"]}
    hitter_split = {}
    for h, col, idc in (("L", "split_L", "L"), ("R", "split_R", "R")):
        per_hitter = Dd.groupby(idc)[col].mean()
        hitter_split[h] = {"league_pts": lg[h], "n_hitters": int(per_hitter.size), "mean": float(per_hitter.mean()),
                           "sd": float(per_hitter.std(ddof=1)), "p10_p50_p90": q(per_hitter),
                           "decision_level_p10_p50_p90": q(spl[h])}
    B3["rules"][key] = {
        "reproduces_a7_json": True,
        "n_pairs": int(len(P)), "n_decisions": int(len(Dd)),
        "per_pair_p10_p50_p90": {c: q(P[c]) for c in ("n_decisions", "share_b_ne_a", "share_c0_ne_a", "share_c_ne_c0",
                                                        "runs_b", "runs_c0", "runs_c")},
        "share_pairs_ride_hotter_costs_runs": float((P["runs_b"] < 0).mean()),
        "share_pairs_any_form_flip": float((P["share_c_ne_c0"] > 0).mean()),
        "hitter_split_pts": hitter_split,
        "secondary_own_vs_league_split": {"share_calls_differ": float((Dd["c0"] != Dd["c0L"]).mean()),
                                          "runs_per50_own_minus_league": float((Dd["runs_c0"] - Dd["runs_c0L"]).mean()),
                                          "runs_per50_league_vs_platoon": float(Dd["runs_c0L"].mean())},
    }
    print(key, json.dumps(B3["rules"][key], indent=1))
pd.concat(pair_rows).to_csv(os.path.join(out3, "a7_pairs.csv"), index=False, float_format="%.6g")
json.dump(B3, open(os.path.join(out3, "b3.json"), "w"), indent=1)
print("wrote a7_pairs.csv and b3.json")

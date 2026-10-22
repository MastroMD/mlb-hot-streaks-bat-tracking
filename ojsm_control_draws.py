"""Post hoc, added after the cold read of 2026-10-17 (PLAN_ADDENDUM_2026-10-17h.md, item 10): how much the return estimates
with 5 matched controls depend on which 5 controls were drawn.

python3 ojsm_control_draws.py DATA_DIR IL_FILE CODED_CSV IL_ORIGINAL OUT_JSON [--draws 1000]

Runs ajsm_game_ojsm.py (same inputs as for repo/results/ajsm_game_ojsm.json) up to the end of F4b with 2 added lines that keep
each stint's usable control hitters in the prespecified random order (seed 506, fixed in the code before the analysis ran);
the prespecified estimate uses the first 5. It checks that the stint rows reproduce repo/results/ojsm_f4b_stint_rows.csv,
then redraws the 5 controls of every stint D times (the first 5 of a random permutation of the same usable controls, seed
20261017) and recomputes, for each draw, the family estimates of games 1 to 10, 21 to 40 and 41 to 60 (hitter-clustered mean,
as in F4b), TOST against +/-0.5 mph, and the specificity contrasts of the family model (as in ojsm_new_analyses.py H3).
Nothing in the published files is changed; the stint rows of the run are written to a scratch folder beside OUT_JSON.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
DATA, ILF, CODED, IL_ORIG, OUT = args[:5]
N_DRAWS = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 1000
RESD = os.path.dirname(os.path.abspath(OUT))   # 2026-10-20: the results checked against are those beside OUT_JSON (repo/results, or a rerun folder)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

# ------------------------------------------------------------------ run ajsm_game_ojsm.py to the end of F4b, keeping the controls
src = open(os.path.join(HERE, "ajsm_game_ojsm.py"), encoding="utf-8").read()
cut = src.index("# ------------------------------------------------------------------ F4c reliability")
src = src[:cut]
a1, a2 = "rng = np.random.default_rng(SEED + 1)\n", "    ctrl = ctrl_all[:5]\n"
assert src.count(a1) == 1 and src.count(a2) == 1
src = src.replace(a1, "_CTRL = {}\n" + a1).replace(a2, a2 + "    _CTRL[k] = ctrl_all\n")
scratch = os.path.join(os.path.dirname(os.path.abspath(OUT)), "control_draws_run")
os.makedirs(scratch, exist_ok=True)
sys.argv = [os.path.join(HERE, "ajsm_game_ojsm.py"), DATA, ILF, os.path.join(scratch, "ajsm_game_ojsm_partial.json"),
            "--shuffles", "5", "--coded", CODED, "--il-original", IL_ORIG]
NS = {"__name__": "__main__", "__file__": os.path.join(HERE, "ajsm_game_ojsm.py")}
exec(compile(src, "ajsm_game_ojsm.py (to the end of F4b)", "exec"), NS)
B, stints, CTRL = NS["B"], NS["stints"], NS["_CTRL"]
log("run done", len(B), "rows", len(stints), "stints")

# gate: the rows of this run equal the published stint rows
pub = pd.read_csv(os.path.join(RESD, "ojsm_f4b_stint_rows.csv"))
key = ["stint", "win", "met"]
m = B.merge(pub, on=key, suffixes=("", "_pub"), how="outer", indicator=True)
assert (m["_merge"] == "both").all(), m["_merge"].value_counts()
for c in ("dev", "ctrl_mean", "ctrl_mean_all", "diff", "diff_all"):
    assert np.nanmax(np.abs(m[c] - m[c + "_pub"])) < 1e-9, c
assert (m["n_ctrl"] == m["n_ctrl_pub"]).all()
log("gate passed: stint rows reproduce", os.path.join(RESD, "ojsm_f4b_stint_rows.csv"))

# ------------------------------------------------------------------ redraws
FAMS = ["oblique", "hamstring", "hand_wrist", "other"]
WINS = ["1-10", "21-40", "41-60"]
MET = "bat_speed"
Z90 = norm.ppf(0.95)


def cl_mean(y, g):
    f = sm.OLS(np.asarray(y, float), np.ones(len(y))).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(g)})
    return float(f.params[0]), float(f.bse[0])


def family_contrasts(D):
    X = pd.get_dummies(D["fam"]).astype(float)[FAMS]
    f = sm.OLS(D["diff"].to_numpy(float), X).fit(cov_type="cluster", cov_kwds={"groups": D["batter"].to_numpy()})
    n_f = D["fam"].value_counts(); n_oth = sum(n_f[x] for x in FAMS if x != "hand_wrist")
    out = {}
    for name, w in (("hand_wrist_minus_all_other", [1.0 if x == "hand_wrist" else -n_f[x] / n_oth for x in FAMS]),
                    ("hand_wrist_minus_oblique", [-1, 0, 1, 0]), ("hand_wrist_minus_hamstring", [0, -1, 1, 0])):
        t = f.t_test(np.asarray(w, float))
        out[name] = (float(np.squeeze(t.effect)), float(np.squeeze(t.sd)))
    return out


# the stint-level injured deviations and the usable controls' values for each (stint, window)
meta = {k: s for k, s in enumerate(stints)}
cells = []   # (stint, batter, fam, win, dev, [control values in prespecified order])
for k, s in meta.items():
    if k not in CTRL:
        continue
    for w_ in WINS:
        if w_ in s["res"] and MET in s["res"][w_]:
            vals = [(i, c[w_][MET][0]) for i, c in enumerate(CTRL[k]) if w_ in c and MET in c[w_]]
            cells.append((k, s["batter"], s["fam"], w_, s["res"][w_][MET][0], vals, len(CTRL[k])))


def estimates(choose):
    """choose(stint k, n usable) -> indices of the 5 controls; returns the estimates of one draw."""
    picks = {}
    rows = []
    for k, b, fam, w_, v, vals, n_use in cells:
        if k not in picks:
            picks[k] = set(choose(k, n_use))
        cv = [x for i, x in vals if i in picks[k]]
        if cv:
            rows.append((k, b, fam, w_, v - float(np.mean(cv))))
    D = pd.DataFrame(rows, columns=["stint", "batter", "fam", "win", "diff"])
    r = {}
    for fam in ("hand_wrist", "oblique"):
        for w_ in WINS:
            d = D[(D["fam"] == fam) & (D["win"] == w_)]
            e, se = cl_mean(d["diff"], d["batter"])
            r[f"{fam}_{w_}"] = {"est": e, "se": se, "lo": e - 1.959964 * se, "hi": e + 1.959964 * se, "n": int(len(d)),
                                "tost_p": float(max(norm.sf((e + 0.5) / se), norm.sf((0.5 - e) / se))),
                                "equivalent": bool(e - Z90 * se > -0.5 and e + Z90 * se < 0.5)}
    for name, (e, se) in family_contrasts(D[D["win"] == "1-10"]).items():
        r[name] = {"est": e, "se": se, "lo": e - 1.959964 * se, "hi": e + 1.959964 * se}
    return r


orig = estimates(lambda k, n: range(min(5, n)))
# the prespecified draw reproduces the published F4b estimates
G = json.load(open(os.path.join(RESD, "ajsm_game_ojsm.json")))
for fam in ("hand_wrist", "oblique"):
    for w_ in WINS:
        assert abs(orig[f"{fam}_{w_}"]["est"] - G["F4b"]["effects"][fam][MET][w_]["mean"]) < 1e-9, (fam, w_)
NA = json.load(open(os.path.join(RESD, "ojsm_new_analyses.json")))
for name in ("hand_wrist_minus_all_other", "hand_wrist_minus_oblique", "hand_wrist_minus_hamstring"):
    assert abs(orig[name]["est"] - NA["H3"]["diff"]["contrasts"][name]["est"]) < 1e-9, name
log("prespecified draw reproduces F4b and the H3 contrasts")

rng = np.random.default_rng(20261017)
draws = []
for d in range(N_DRAWS):
    perms = {}

    def choose(k, n):
        if k not in perms:
            perms[k] = rng.permutation(n)[:5]
        return perms[k]
    draws.append(estimates(choose))
    if (d + 1) % 100 == 0:
        log("draws", d + 1)


def summ(key, flag=None):
    v = np.array([x[key]["est"] for x in draws])
    s = {"prespecified": orig[key]["est"], "median": float(np.median(v)), "p2.5": float(np.percentile(v, 2.5)),
         "p97.5": float(np.percentile(v, 97.5)), "min": float(v.min()), "max": float(v.max()),
         "share_below_prespecified": float(np.mean(v < orig[key]["est"]))}
    if flag == "ci_below_0":
        s["share_ci_below_0"] = float(np.mean([x[key]["hi"] < 0 for x in draws]))
    if flag == "equivalent":
        s["share_equivalent"] = float(np.mean([x[key]["equivalent"] for x in draws]))
        s["prespecified_equivalent"] = orig[key]["equivalent"]
    return s


res = {"note": __doc__.split("\n\n")[0], "n_draws": N_DRAWS, "seed_redraws": 20261017, "seed_prespecified": 506,
       "usable_controls_per_stint_median": float(np.median([len(v) for v in CTRL.values()])),
       "stints_with_5_or_fewer_usable_controls": int(sum(len(v) <= 5 for v in CTRL.values())),
       "hand_wrist_1-10": summ("hand_wrist_1-10", "ci_below_0"), "oblique_1-10": summ("oblique_1-10", "ci_below_0"),
       "hand_wrist_21-40": summ("hand_wrist_21-40", "equivalent"), "oblique_21-40": summ("oblique_21-40", "equivalent"),
       "hand_wrist_41-60": summ("hand_wrist_41-60", "ci_below_0"),
       "hand_wrist_minus_all_other": summ("hand_wrist_minus_all_other", "ci_below_0"),
       "hand_wrist_minus_oblique": summ("hand_wrist_minus_oblique", "ci_below_0"),
       "hand_wrist_minus_hamstring": summ("hand_wrist_minus_hamstring", "ci_below_0"),
       "every_eligible_control": {k: G["F4b"]["effects_all_controls"][f][MET][w]["mean"]
                                  for k, f, w in (("hand_wrist_1-10", "hand_wrist", "1-10"), ("oblique_1-10", "oblique", "1-10"),
                                                  ("hand_wrist_21-40", "hand_wrist", "21-40"), ("hand_wrist_41-60", "hand_wrist", "41-60"))}}
json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
print(json.dumps(res, indent=1))

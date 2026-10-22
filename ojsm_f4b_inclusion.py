"""Every-eligible-control return estimates without the row condition of the 5-control draw (post hoc, third independent
check of 2026-10-20).

python3 ojsm_f4b_inclusion.py STATCAST_DIR IL_FILE CODED_CSV PUBLISHED_DIR OUT_JSON

In ajsm_game_ojsm.py a stint's window (and outcome) enters the return analysis only when at least 1 of its 5 drawn control
hitters has data in that window, and the every-eligible-control estimate uses the same rows; so which rows that estimate
uses depends on the 5-control draw. This script runs ajsm_game_ojsm.py's own code (read from that file, with asserted
edits: the at-risk models, reliability and persistence parts are not run and no files are written there) and keeps every
row in which any eligible control has data. It stops unless the rows that meet the original condition, and their
every-eligible-control estimates, equal those in PUBLISHED_DIR (ojsm_f4b_stint_rows.csv and ajsm_game_ojsm.json of the
same IL file); it then reports how many rows the condition dropped and the every-eligible-control estimates with them.
"""
import json
import os
import sys

DATA, ILF, CODED, PUB, OUT_JSON = sys.argv[1:6]
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ajsm_game_ojsm.py")
src = open(SRC, encoding="utf-8").read()


def cut(s, start, end_line_prefix, repl=""):
    i = s.index(start)
    assert s.count(start) == 1, start
    j = s.index(end_line_prefix, i)
    j = s.index("\n", j) + 1
    return s[:i] + repl + s[j:]


src = cut(src, 'out["F4a_A8"] = {"complete_forward_300pa_replication"', 'log("A8 at-risk done"')
old = 'R.to_csv(os.path.join(os.path.dirname(os.path.abspath(OUT)), "ojsm_f4a_rows.csv"), index=False)\n'
assert src.count(old) == 1
src = src.replace(old, "")
old = ('            cv = [c[w_][met][0] for c in ctrl if w_ in c and met in c[w_]]\n'
       '            if not cv:\n'
       '                continue\n'
       '            cva = [c[w_][met][0] for c in ctrl_all if w_ in c and met in c[w_]]\n'
       '            rowsB.append({"stint": k, "batter": s["batter"], "fam": s["fam"], "imp": s["imp"], "win": w_, "met": met, "dev": v, "w": w,\n'
       '                          "ctrl_mean": float(np.mean(cv)), "n_ctrl": len(cv), "src": s["src"],\n'
       '                          "ctrl_mean_all": float(np.mean(cva)), "n_ctrl_all": len(cva)})\n')
new = ('            cv = [c[w_][met][0] for c in ctrl if w_ in c and met in c[w_]]\n'
       '            cva = [c[w_][met][0] for c in ctrl_all if w_ in c and met in c[w_]]\n'
       '            if not cva:\n'
       '                continue\n'
       '            rowsB.append({"stint": k, "batter": s["batter"], "fam": s["fam"], "imp": s["imp"], "win": w_, "met": met, "dev": v, "w": w,\n'
       '                          "ctrl_mean": float(np.mean(cv)) if cv else np.nan, "n_ctrl": len(cv), "src": s["src"],\n'
       '                          "ctrl_mean_all": float(np.mean(cva)), "n_ctrl_all": len(cva), "kept_prespecified": bool(cv)})\n')
assert src.count(old) == 1
src = src.replace(old, new)
i = src.index("B = pd.DataFrame(rowsB)\n")
src = src[:i] + r'''
B = pd.DataFrame(rowsB)
B["diff"] = B["dev"] - B["ctrl_mean"]
B["diff_all"] = B["dev"] - B["ctrl_mean_all"]
Bp = B[B["kept_prespecified"]].drop(columns="kept_prespecified")
ref = pd.read_csv(os.path.join(_PUB, "ojsm_f4b_stint_rows.csv"))
key = ["stint", "win", "met"]
a_, r_ = Bp[list(ref.columns)].sort_values(key).reset_index(drop=True), ref.sort_values(key).reset_index(drop=True)
assert len(a_) == len(r_), (len(a_), len(r_))
for c in ref.columns:
    if a_[c].dtype.kind in "fi" and r_[c].dtype.kind in "fi":
        assert np.allclose(a_[c].to_numpy(float), r_[c].to_numpy(float), rtol=0, atol=1e-12, equal_nan=True), c
    else:
        assert (a_[c].astype(str) == r_[c].astype(str)).all(), c
REF = json.load(open(os.path.join(_PUB, "ajsm_game_ojsm.json")))["F4b"]["effects_all_controls"]
res, dropped, maxchg, nrep = {}, {}, {}, 0
for fam, gcol, gval in GROUPS:
    for met in MET:
        for w_ in [f"{lo}-{hi}" for lo, hi in POST]:
            D = B[(B["met"] == met) & (B["win"] == w_) & ((B[gcol] == gval) if gcol else True)]
            Dp = D[D["kept_prespecified"]]
            if len(Dp) < 5:
                continue
            ep = ols_mean(Dp["diff_all"], np.ones(len(Dp)), Dp["batter"])
            ea = ols_mean(D["diff_all"], np.ones(len(D)), D["batter"])
            rf = REF[fam][met][w_]
            assert abs(ep["mean"] - rf["mean"]) < 1e-9 and abs(ep["se"] - rf["se"]) < 1e-9 and ep["n"] == rf["n"], (fam, met, w_)
            nrep += 1
            res.setdefault(fam, {}).setdefault(met, {})[w_] = {"published": {k: ep[k] for k in ("mean", "lo", "hi", "p", "n")},
                                                               "any_eligible_control": {k: ea[k] for k in ("mean", "lo", "hi", "p", "n")},
                                                               "rows_added": int(len(D) - len(Dp))}
            maxchg[met] = max(maxchg.get(met, 0.0), abs(ea["mean"] - ep["mean"]))
for (met, w_), x in B.groupby(["met", "win"]):
    dropped.setdefault(met, {})[w_] = int((~x["kept_prespecified"]).sum())
out_ = {"note": _NOTE,
        "il_sha16": sha16(ILF), "published_dir": _PUB, "estimates_reproduced": nrep,
        "rows_prespecified_condition": int(len(Bp)), "rows_any_eligible_control": int(len(B)),
        "rows_dropped_by_the_condition": dropped, "stints_with_a_dropped_row": int(B.loc[~B["kept_prespecified"], "stint"].nunique()),
        "max_abs_change_by_outcome": maxchg, "effects_all_controls": res}
json.dump(out_, open(_OUT_JSON, "w"), indent=1)
print(json.dumps({k: out_[k] for k in ("estimates_reproduced", "rows_prespecified_condition", "rows_any_eligible_control",
                                         "stints_with_a_dropped_row", "max_abs_change_by_outcome")}, indent=1))
'''
tmp_out = os.path.join(os.path.dirname(os.path.abspath(OUT_JSON)), "not_written.json")   # OUT of ajsm_game_ojsm.py: its writes are cut
sys.argv = [SRC, DATA, ILF, tmp_out, "--shuffles", "5", "--coded", CODED]
ns = {"__name__": "__main__", "__file__": SRC, "_NOTE": __doc__.split("\n\n")[1].replace("\n", " "), "_PUB": PUB, "_OUT_JSON": OUT_JSON}
exec(compile(src, SRC, "exec"), ns)

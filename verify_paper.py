"""Independent check of the headline estimates. Does NOT import hot_streak_study / analysis2 / common3.

python3 verify_paper.py DATA_DIR PAPER_RESULTS_JSON FORECAST_FRAME_PARQUET ABSTRACT_MD [PAPER_DOCX ...]
Exit 0 only if every check passes.

 1. Raw panel rebuilt from the parquet files with its own code (park/week adjustment, competitive swings,
    batter-game sums) and a per-hitter-season two-pointer window loop (no cumulative sums / searchsorted).
    Actual 50->50 slopes for bat speed, swing length, fast-swing rate and wOBA must equal the pipeline's to 1e-6.
 2. Own shuffles (5 seeds, own permutation code) -> persistence within 0.02 of the 20-shuffle value.
 3. League platoon splits recomputed from pitch rows: must equal to 0.05 wOBA points.
 4. Swing-form and results-form deciles re-estimated from the exported design matrix with numpy WLS,
    own leave-one-season-out loop and own cluster sandwich: within 0.5 wOBA points (SE within 0.5).
 5. Stale-value scan: every number in the abstract (excluding years, window sizes and citations) must appear in
    paper_results.json['abstract']; the abstract must be under 500 words (title + body).
 7. The 200-PA forward horizon in calendar weeks (median days from the decision game to the game completing the next
    200 PA, A1 decision points), own loop: must round to paper_results.json['abstract']['weeks_200'].
 6. A11 (added 2026-09-30, plan addendum C): offseason carry-over slope b (last 150 PA of Y vs first 150 of Y+1, both minus
    the rest of Y; age, age^2, pair dummy) and R_next (brute-force changepoint on Y, first 100 PA of Y+1) recomputed
    with own window, changepoint and OLS code: within 0.02 of paper_results.json['A11']. FG_BAT (env) supplies age
    where the Marcel file lacks it, as in the pipeline.
 8. B3 (added 2026-10-02, plan addendum D): the per-pair decision table results/a7_pairs.csv, aggregated with own code
    (decision-weighted means and shares, across-pair quantiles), must reproduce paper_results.json['A7'] and ['B3'].
 9. Full paper (for each PAPER_DOCX given, e.g. the SSAC27 paper, the JQAS manuscript and its online appendix): every number in the paper's text and tables, outside the reference list and
    headings, must appear in paper_results.json['paper'] (text values, table cells, design constants) or be a small
    structural integer (1-10, figure and table numbers, deciles) or a window size; no unfilled placeholder; no colon
    or semicolon in body prose (house style). Reports how many Results-text numbers also appear in a table.
10. The abstract block of paper_results.json is unchanged since submission (sorted-key sha256, first 16 hex).
"""
import glob
import json
import re
import sys

import numpy as np
import pandas as pd

data_dir, pr_path, frame_path, abs_path = sys.argv[1:5]
paper_docxs = sys.argv[5:]
PR = json.load(open(pr_path))
fails = []


def check(name, ok, detail):
    print(("PASS " if ok else "FAIL ") + name + "  " + detail)
    if not ok:
        fails.append(name)


# ------------------------------------------------------------------ 1. panel
cols = ["game_date", "game_pk", "game_type", "game_year", "at_bat_number", "pitch_number", "batter", "stand", "p_throws",
        "home_team", "description", "type", "launch_speed", "woba_value", "woba_denom", "bat_speed", "swing_length"]
df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(data_dir + "/*.parquet"))], ignore_index=True)
df = df[df["game_type"] == "R"]
for c in ["bat_speed", "swing_length", "launch_speed", "woba_value", "woba_denom"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
df["batter"] = df["batter"].astype(np.int64); df["game_pk"] = df["game_pk"].astype(np.int64)
df["game_year"] = df["game_year"].astype(np.int64); df["game_date"] = pd.to_datetime(df["game_date"])
df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
print(f"{len(df):,} pitches")
check("pitch count", len(df) == PR["meta"]["n_pitches"], f"{len(df)} vs {PR['meta']['n_pitches']}")

# platoon (unadjusted rows are fine: only wOBA)
pa = df[df["woba_denom"] == 1]
for st, key in [("L", "LHB_league_split_pts"), ("R", "RHB_league_split_pts")]:
    s = pa[pa["stand"] == st]
    same = s["p_throws"] == st
    v = 1000 * (s.loc[~same, "woba_value"].fillna(0).mean() - s.loc[same, "woba_value"].fillna(0).mean())
    check(f"platoon {st}HB", abs(v - PR["primary"]["platoon"][key]) < 0.05, f"{v:.3f} vs {PR['primary']['platoon'][key]:.3f}")

# park-season and ISO-week-of-season adjustment, written independently
week = df["game_date"].dt.isocalendar().week.astype(int).to_numpy()
for c in ["bat_speed", "swing_length", "launch_speed"]:
    x = df[c].to_numpy()
    tmp = pd.DataFrame({"x": x, "y": df["game_year"].to_numpy(), "p": df["home_team"].to_numpy(), "w": week})
    ym = tmp.groupby("y")["x"].mean()
    pm = tmp.groupby(["p", "y"])["x"].mean()
    wm = tmp.groupby(["y", "w"])["x"].mean()
    yv = ym.reindex(tmp["y"]).to_numpy()
    pv = pm.reindex(pd.MultiIndex.from_arrays([tmp["p"], tmp["y"]])).to_numpy()
    wv = wm.reindex(pd.MultiIndex.from_arrays([tmp["y"], tmp["w"]])).to_numpy()
    df[c] = x - (pv - yv) - (wv - yv)

swing = df["description"].isin(["swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"])
bs, sl, ev = df["bat_speed"], df["swing_length"], df["launch_speed"]
p10 = bs.groupby([df["batter"], df["game_year"]]).transform(lambda s: s.quantile(0.10))
comp = bs.notna() & sl.notna() & ((bs >= p10) | (ev >= 90))
df["is_pa"] = (df["woba_denom"] == 1).astype(float)
df["wnum"] = np.where(df["is_pa"] == 1, df["woba_value"].fillna(0), 0.0)
df["c"] = comp.astype(float)
df["bsn"] = np.where(comp, bs, 0.0)
df["sln"] = np.where(comp, sl, 0.0)
df["fs"] = (comp & (bs >= 75)).astype(float)
G = df.groupby(["batter", "game_year", "game_date", "game_pk"], sort=True)[["is_pa", "wnum", "c", "bsn", "sln", "fs"]].sum().reset_index()
G = G[G["is_pa"] > 0].sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)

MET = {"bat_speed": ("bsn", "c", 30), "swing_len": ("sln", "c", 30), "fast_swing": ("fs", "c", 30), "woba": ("wnum", "is_pa", 1)}


def two_pointer(G):
    """rows: (batter, season_pa, rec/fwd/orc sums) for every game with 50 PA before and 50 PA after within the season"""
    out = []
    V = G[["is_pa", "wnum", "c", "bsn", "sln", "fs"]].to_numpy()
    for (b, y), idx in G.groupby(["batter", "game_year"], sort=False).indices.items():
        X = V[idx]
        tot = X.sum(0)
        n = len(X)
        for i in range(n):
            # recent: walk back from i-1 until >= 50 PA
            s, j = np.zeros(X.shape[1]), i
            while j > 0 and s[0] < 50:
                j -= 1; s = s + X[j]
            if s[0] < 50:
                continue
            f, k = np.zeros(X.shape[1]), i
            while k < n and f[0] < 50:
                f = f + X[k]; k += 1
            if f[0] < 50:
                continue
            out.append((b, tot[0], s, f, tot - s - f))
    return out


def slopes(rows, m):
    num, den, mn = MET[m]
    ci = {"is_pa": 0, "wnum": 1, "c": 2, "bsn": 3, "sln": 4, "fs": 5}
    x, y, g = [], [], []
    for b, spa, r, f, o in rows:
        if spa < 300 or r[ci[den]] < mn or f[ci[den]] < mn or o[ci[den]] < 3 * mn:
            continue
        ob = o[ci[num]] / o[ci[den]]
        x.append(r[ci[num]] / r[ci[den]] - ob); y.append(f[ci[num]] / f[ci[den]] - ob); g.append(b)
    x, y, g = np.array(x), np.array(y), np.array(g)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[1], len(x)


rows = two_pointer(G)
for m in MET:
    b, n = slopes(rows, m)
    act = PR["primary"]["persistence"][m]["slope_actual"]
    check(f"actual slope {m}", abs(b - act) < 1e-6, f"{b:.8f} vs {act:.8f} (n={n})")

# ------------------------------------------------------------------ 2. own shuffles
rng = np.random.default_rng(424242)
nulls = {m: [] for m in MET}
for s in range(5):
    H = G.copy()
    perm = np.empty(len(H), dtype=int)
    for _, idx in H.groupby(["batter", "game_year"], sort=False).indices.items():
        perm[idx] = rng.permutation(idx)
    H[["is_pa", "wnum", "c", "bsn", "sln", "fs"]] = G[["is_pa", "wnum", "c", "bsn", "sln", "fs"]].to_numpy()[perm]
    r0 = two_pointer(H)
    for m in MET:
        nulls[m].append(slopes(r0, m)[0])
for m in MET:
    b, _ = slopes(rows, m)
    p = b - np.mean(nulls[m])
    ref = PR["primary"]["persistence"][m]["persistence"]
    check(f"persistence {m}", abs(p - ref) < 0.02, f"{p:.4f} vs {ref:.4f}")

# ------------------------------------------------------------------ 4. deciles from the exported design
D = pd.read_parquet(frame_path)
base = [c for c in D.columns if c.startswith("prior_") or c.startswith("std_") or c in ("fwd_oppq", "fwd_park", "fwd_same")]
y = D["fwd_woba"].to_numpy() * 1000
w = D["fwd_pa"].to_numpy().astype(float)


def wls(X, y, w):
    Xa = np.column_stack([np.ones(len(X)), X])
    sw = np.sqrt(w)
    return np.linalg.lstsq(Xa * sw[:, None], y * sw, rcond=None)[0]


bb = wls(D[base].to_numpy(), y, w)
resid = y - np.column_stack([np.ones(len(D)), D[base].to_numpy()]) @ bb


def cluster_wls(X, y, w, g):
    Xa = np.column_stack([np.ones(len(X)), X])
    beta = wls(X, y, w)
    e = y - Xa @ beta
    bread = np.linalg.inv(Xa.T @ (Xa * w[:, None]))
    meat = np.zeros((Xa.shape[1], Xa.shape[1]))
    order = np.argsort(g, kind="stable")
    gs = g[order]
    cuts = np.flatnonzero(np.diff(gs)) + 1
    for blk in np.split(order, cuts):
        u = (Xa[blk] * (w[blk] * e[blk])[:, None]).sum(0)
        meat += np.outer(u, u)
    n, k, G_ = len(y), Xa.shape[1], len(cuts) + 1
    V = bread @ meat @ bread * (G_ / (G_ - 1)) * ((n - 1) / (n - k))
    return beta, np.sqrt(np.diag(V))


def deciles(cols):
    comp = np.full(len(D), np.nan)
    for yr in sorted(D["game_year"].unique()):
        tr, te = (D["game_year"] != yr).to_numpy(), (D["game_year"] == yr).to_numpy()
        be = wls(D.loc[tr, cols].to_numpy(), resid[tr], w[tr])
        comp[te] = D.loc[te, cols].to_numpy() @ be[1:]
    rk = pd.Series(comp).rank(method="first").to_numpy()
    q = pd.qcut(pd.Series(rk), 10, labels=False).to_numpy()
    X = np.column_stack([(q == k).astype(float) for k in range(10) if k not in (4, 5)])
    beta, se = cluster_wls(X, resid, w, D["batter"].to_numpy())
    return beta[1], se[1], beta[-1], se[-1]


for name, cols, key in [("swing", ["dev_bat_speed", "dev_fast_swing", "dev_swing_len"], "swing_only"), ("results", ["dev_woba"], "results_only")]:
    bot, bse, top, tse = deciles(cols)
    ref = PR["primary"]["forms"][key]["deciles"]
    check(f"{name} top decile", abs(top - ref["bump"][9]) < 0.5 and abs(tse - ref["se"][9]) < 0.5, f"{top:.2f} ({tse:.2f}) vs {ref['bump'][9]:.2f} ({ref['se'][9]:.2f})")
    check(f"{name} bottom decile", abs(bot - ref["bump"][0]) < 0.5 and abs(bse - ref["se"][0]) < 0.5, f"{bot:.2f} ({bse:.2f}) vs {ref['bump'][0]:.2f} ({ref['se'][0]:.2f})")

# ------------------------------------------------------------------ 5. stale-value scan + word count
txt = re.sub(r"^!\[.*?\]\(.*?\)$", "", open(abs_path).read(), flags=re.M)  # markdown image line: file name only
vals = set()
for v in PR["abstract"].values():
    for tok in re.findall(r"\d[\d,]*\.?\d*", str(v)):
        vals.add(tok.rstrip("."))
vals |= {f"{round((float(PR['abstract']['top_swing']) + float(PR['abstract']['bot_swing'])) / 2):.0f}"}
allowed = {"1993", "2007", "2018", "2024", "2026", "50", "16", "20", "25", "200", "75", "1", "3", "30"}
stale = []
for tok in re.findall(r"\d[\d,]*\.?\d*", txt):
    t = tok.rstrip(".")
    if t not in vals and t not in allowed:
        stale.append(t)
check("stale-value scan", not stale, f"numbers in abstract not in paper_results.json: {sorted(set(stale))}")
body, _, rest = txt.partition("**Figure 1." if "**Figure 1." in txt else "**Table 1.")
_, _, concl = rest.partition("## Conclusion")
wc = lambda s: len(re.sub(r"[#*|]", " ", s).split())
n = wc(body) + wc("## Conclusion" + concl) - wc("**Track:** Baseball")
check("abstract word count < 500", n < 500, f"{n} words (title + body, table excluded)")

# ------------------------------------------------------------------ 6. A11 offseason carry-over b and R_next (own code)
A11 = PR.get("A11")
if A11 is not None:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    age = {}
    for tok in open(os.path.join(here, "data_public", "marcel_woba_2024_2026.txt")).read().split():
        f_ = tok.split(":")
        if f_[5]:
            age[(int(f_[0]), 2020 + int(f_[1]))] = float(f_[5])
    fgp = os.environ.get("FG_BAT")
    if fgp:
        fg = pd.read_excel(fgp, usecols=["Season", "MLBAMID", "Age"]).dropna()
        last = {}
        for s_, b_, a_ in sorted(fg[["Season", "MLBAMID", "Age"]].itertuples(index=False)):
            last.setdefault(int(b_), {})[int(s_)] = float(a_)
        for b_, d_ in last.items():
            for Y in (2024, 2025, 2026):
                if (b_, Y) not in age:
                    for k_ in (1, 2, 3):
                        if Y - k_ in d_:
                            age[(b_, Y)] = d_[Y - k_] + k_
                            break
    HS = {}
    V6 = G[["is_pa", "c", "bsn"]].to_numpy()
    for (b_, y_), idx in G.groupby(["batter", "game_year"], sort=False).indices.items():
        HS[(int(b_), int(y_))] = V6[np.sort(idx)]

    def wmean(rows):
        return rows[:, 2].sum() / rows[:, 1].sum()

    def take_first(X, n):
        tot, k_ = 0.0, 0
        while k_ < len(X) and tot < n:
            tot += X[k_, 0]; k_ += 1
        return k_, tot >= n

    def cp_brute(X):
        q = X[X[:, 1] >= 5]
        if len(q) < 40:
            return None
        best = None
        for k_ in range(20, len(q) - 20 + 1):
            a_, b2 = q[:k_], q[k_:]
            m1, m2 = wmean(a_), wmean(b2)
            sse = (a_[:, 1] * (a_[:, 2] / a_[:, 1] - m1) ** 2).sum() + (b2[:, 1] * (b2[:, 2] / b2[:, 1] - m2) ** 2).sum()
            if best is None or sse < best[0] - 1e-12:
                best = (sse, m1, m2)
        return best[1], best[2]

    medage = np.median(list(age.values()))
    rb, rr = [], []
    for (b_, y_), X in HS.items():
        if y_ == 2026 or X[:, 0].sum() < 300:
            continue
        N = HS.get((b_, y_ + 1))
        a_ = age.get((b_, y_), medage)
        k_, _ = take_first(X[::-1], 150)
        last_, rest_ = X[len(X) - k_:], X[:len(X) - k_]
        if N is not None and N[:, 0].sum() >= 150:
            kf, _ = take_first(N, 150)
            rb.append((wmean(last_) - wmean(rest_), wmean(N[:kf]) - wmean(rest_), a_, y_, b_))
        if N is not None and N[:, 0].sum() >= 100:
            cp = cp_brute(X)
            if cp is not None:
                kf, _ = take_first(N, 100)
                rr.append((cp[1] - cp[0], wmean(N[:kf]) - cp[0], a_, y_, b_))

    def own_ols(rows):
        R_ = np.array(rows, float)
        Xd = np.column_stack([np.ones(len(R_)), R_[:, 0], R_[:, 2] - 28, (R_[:, 2] - 28) ** 2, (R_[:, 3] == 2025).astype(float)])
        return np.linalg.lstsq(Xd, R_[:, 1], rcond=None)[0][1], len(R_)

    b6, n6 = own_ols(rb)
    r6, m6 = own_ols(rr)
    check("A11b offseason carry-over b", abs(b6 - A11["b"]) < 0.02, f"{b6:.4f} vs {A11['b']:.4f} (n={n6} vs {A11['b_n']})")
    check("A11a(ii) R_next", abs(r6 - A11["R_next"]) < 0.02, f"{r6:.4f} vs {A11['R_next']:.4f} (n={m6} vs {A11['R_next_n']})")

# ------------------------------------------------------------------ 7. PA horizon in calendar weeks (own loop)
if "weeks_200" in PR["abstract"]:
    spans = []
    Gd = G["game_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    Gp = G["is_pa"].to_numpy()
    Gc = G["c"].to_numpy()
    for (b_, y_), idx in G.groupby(["batter", "game_year"], sort=False).indices.items():
        idx = np.sort(idx)
        p_, d_, c_ = Gp[idx], Gd[idx], Gc[idx]
        if p_.sum() < 300:
            continue
        for i in range(len(idx)):
            back, j, cb = 0.0, i, 0.0
            while j > 0 and back < 50:
                j -= 1; back += p_[j]; cb += c_[j]
            if back < 50 or cb < 30:
                continue
            fwd, k, cf = 0.0, i, 0.0
            while k < len(idx) and fwd < 200:
                fwd += p_[k]; cf += c_[k]; k += 1
            if fwd < 200 or cf < 30:
                continue
            spans.append(d_[k - 1] - d_[i] + 1)
    wk = np.median(spans) / 7
    check("200-PA horizon in weeks", f"{wk:.0f}" == PR["abstract"]["weeks_200"], f"{wk:.2f} weeks (n={len(spans)}) vs {PR['abstract']['weeks_200']}")


# ------------------------------------------------------------------ 8. B3 per-pair table, own aggregation
import hashlib
import os

if "B3" in PR:
    here = os.path.dirname(os.path.abspath(__file__))
    pairs = pd.read_csv(os.path.join(here, "results", "a7_pairs.csv"))
    for key in ("prespecified", "refined_exploratory"):
        a7, b3 = PR["A7"][key], PR["B3"]["rules"][key]
        T_ = pairs[pairs["rule"] == key]
        n = T_["n_decisions"].to_numpy(float)
        wmean = lambda c: float((T_[c].to_numpy(float) * n).sum() / n.sum())
        ok = (int(n.sum()) == a7["n_decisions"] and len(T_) == b3["n_pairs"]
              and abs(wmean("runs_b") - a7["option_b_vs_strict_platoon_runs_per50"]["mean"]) < 1e-4
              and abs(wmean("runs_c0") - a7["option_c0_vs_strict_platoon_runs_per50"]["mean"]) < 1e-4
              and abs(wmean("runs_c") - a7["option_c_vs_strict_platoon_runs_per50"]["mean"]) < 1e-4
              and abs(wmean("share_b_ne_a") - a7["option_b_vs_strict_platoon_runs_per50"]["share_differs_from_platoon"]) < 1e-4
              and abs(wmean("share_c_ne_c0") - a7["share_form_flips_call"]) < 1e-4
              and abs(wmean("share_c0_ne_c0L") - b3["secondary_own_vs_league_split"]["share_calls_differ"]) < 1e-4
              and abs(wmean("runs_c0") - wmean("runs_c0L") - b3["secondary_own_vs_league_split"]["runs_per50_own_minus_league"]) < 1e-4)
        q = np.percentile(T_["runs_b"].to_numpy(float), [10, 50, 90])
        ok = ok and np.allclose(q, a7["option_b_vs_strict_platoon_runs_per50"]["per_pair_p10_p50_p90"], atol=1e-4)
        check(f"B3 per-pair table, {key}", ok,
              f"{int(n.sum())} decisions in {len(T_)} pairs; ride-hotter {wmean('runs_b'):+.4f} runs/50 PA, form flips {wmean('share_c_ne_c0'):.4f}, "
              f"own split changes {wmean('share_c0_ne_c0L'):.4f}")

# ------------------------------------------------------------------ 10. abstract block unchanged since submission
ah = hashlib.sha256(json.dumps(PR["abstract"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
check("abstract block unchanged", ah == "93fc915627b12146", f"{ah}")

# ------------------------------------------------------------------ 9. full paper: stale values, placeholders, house style
for paper_docx in paper_docxs:
    from docx import Document
    PB = PR["paper"]
    known = set()
    num_re = re.compile(r"\d[\d,]*\.?\d*")
    norm = lambda t: t.rstrip(".").replace(",", "")
    for v in list(PB["text"].values()) + [str(v) for v in PB["design"].values()]:
        known |= {norm(t) for t in num_re.findall(str(v))}
    table_nums = set()
    for T_ in PB["tables"].values():
        for row in T_["rows"] + T_.get("albright", []):
            for cell in row:
                table_nums |= {norm(t) for t in num_re.findall(str(cell))}
    known |= table_nums
    structural = {str(i) for i in range(0, 11)} | {"25", "50", "100", "150", "200", "16", "95"}
    D_ = Document(paper_docx)
    body, in_refs, section = [], False, ""
    for p_ in D_.paragraphs:
        sty = p_.style.name
        if sty.startswith("Heading") or sty == "Title":
            in_refs = p_.text.strip() == "References"
            section = p_.text.strip()
            structural |= {norm(t) for t in num_re.findall(section)}          # section numbers cited in the text
            continue
        if in_refs:
            structural |= set(re.findall(r"\((\d{4})\)", p_.text))           # citation years
        elif p_.text.strip():
            body.append((section, p_.text))
    cells = [c.text for t in D_.tables for r in t.rows for c in r.cells]
    found = {norm(t) for _, txt in body for t in num_re.findall(txt)} | {norm(t) for c in cells for t in num_re.findall(c)}
    stale = sorted(found - known - structural)
    tag = os.path.basename(paper_docx)
    check(f"paper stale-value scan, {tag}", not stale, f"numbers in the paper not in paper_results.json['paper']: {stale[:20]}")
    unfilled = [txt[:60] for _, txt in body if re.search(r"\{\w+\}", txt)]
    check(f"paper placeholders filled, {tag}", not unfilled, f"{unfilled[:3]}")
    prose = [txt for sec_, txt in body if not txt.startswith(("Figure", "Table", "Note.", "[")) and sec_ not in ("",)]
    punct = [txt[:70] for txt in prose if re.search(r"[:;]", txt)]
    check(f"no colons or semicolons in body prose, {tag}", not punct, f"{len(punct)} paragraphs: {punct[:3]}")
    res_nums = [norm(t) for sec_, txt in body if sec_.startswith("4.") and not txt.startswith(("Figure", "Table", "Note."))
                for t in num_re.findall(txt)]
    dup = [t for t in res_nums if t in table_nums and t not in structural]
    print(f"INFO {tag}: Results-text numbers that also appear in a table: {len(dup)} of {len(res_nums)} (house rule 2A reports this count)")

print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(0 if not fails else 1)

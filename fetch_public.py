"""Rebuild the three public inputs in data_public/ from the MLB Stats API (statsapi.mlb.com).

python3 fetch_public.py STATCAST_DIR OUT_DIR

  il_transactions_hitters_2024_2026.txt  /transactions, typeCode SC, "injured list", non-pitchers, restricted to
                                         hitters with a >=300-PA season in the Statcast panel.
                                         id:YYMMDD<code> ...   P<fam><days> placement (retroactive date if stated),
                                         A activation/reinstatement; fam O oblique/intercostal, H hamstring,
                                         W hand/wrist/thumb/finger/hamate/metacarpal, X other.
  marcel_woba_2024_2026.txt              Marcel wOBA for every hitter-season with >=100 PA (hitters with >=250 PA
                                         2024-26) from season hitting lines 2021-2025: 5/4/3 weights, 1,200 PA of
                                         league average, age x(1+0.006(29-age)) under 29, x(1-0.003(age-29)) over.
                                         id:Y:proj*1000:reliability*100:PA prior season:age
  postseason_woba_2024_2025.txt          gameType=P season lines: id:Y:wOBA numerator*1000:denominator:PA

The files used for the paper were pulled on 2026-09-30 with the same logic (pulled through a web browser, because
statsapi.mlb.com was not reachable from the build machine's shell); hashes are in PLAN_2026-09-30.md.
The two checksums printed at the end (31-multiplier rolling hash) match the ones recorded at pull time.
"""
import glob
import json
import os
import re
import sys
import urllib.request
from datetime import date, timedelta

import pandas as pd

API = "https://statsapi.mlb.com/api/v1"
W = dict(ubb=.696, hbp=.726, b1=.883, b2=1.244, b3=1.569, hr=2.004)
MON = {m: i + 1 for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                        "September", "October", "November", "December"])}


def get(path):
    with urllib.request.urlopen(API + path, timeout=60) as r:
        return json.load(r)


def jsround(x):
    """JavaScript Math.round (half up), used at pull time"""
    import math
    return int(math.floor(x + 0.5))


def chk(s):
    c = 0
    for ch in s:
        c = (c * 31 + ord(ch)) & 0xFFFFFFFF
    return c


def panel_ids(statcast_dir):
    d = pd.concat([pd.read_parquet(f, columns=["batter", "game_year", "woba_denom", "game_type"])
                   for f in sorted(glob.glob(os.path.join(statcast_dir, "*.parquet")))])
    d = d[d["game_type"] == "R"]
    g = d.groupby(["batter", "game_year"])["woba_denom"].sum().reset_index()
    ids300 = set(g.loc[g["woba_denom"] >= 300, "batter"].astype(int))
    tot = g.groupby("batter")["woba_denom"].sum()
    g = g[(g["woba_denom"] >= 100) & g["batter"].map(tot).ge(250)]
    pairs = g.groupby("batter")["game_year"].apply(lambda s: "".join(str(y)[-1] for y in sorted(s)))
    return ids300, pairs


def il(ids):
    out = []
    months = [(y, m) for y in (2024, 2025, 2026) for m in range(2, 12) if not (y == 2026 and m > 9)]
    for y, m in months:
        s = date(y, m, 1)
        e = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
        for x in get(f"/transactions?sportId=1&startDate={s}&endDate={e}").get("transactions", []):
            d = x.get("description", "")
            if x.get("typeCode") != "SC" or not re.search("injured list", d, re.I) or "person" not in x:
                continue
            pm = re.search(r"(?:placed|activated|reinstated|transferred|recalled)\s+(?:[A-Z][a-z]+\s)?([A-Z0-9]{1,3})\s", d)
            pos = pm.group(1) if pm else "?"
            if pos in ("RHP", "LHP", "P"):
                continue
            k = "P" if re.search("placed", d, re.I) else ("A" if re.search("activated|reinstated", d, re.I) else "T")
            if k == "T":
                continue
            dt = x["date"]
            rm = re.search(r"retroactive to ([A-Z][a-z]+) (\d+), (\d{4})", d)
            if rm and k == "P":
                dt = f"{rm.group(3)}-{MON[rm.group(1)]:02d}-{int(rm.group(2)):02d}"
            low = d.lower()
            f = "O" if re.search("oblique|intercostal", low) else "H" if "hamstring" in low else \
                "W" if re.search("hand|wrist|thumb|finger|hamate|metacarpal", low) else "X"
            days = (re.search(r"(\d+)-day", d) or [None, ""])[1]
            out.append(f"{x['person']['id']},{dt.replace('-', '')[2:]},{k}{f + days if k == 'P' else ''}")
    out = list(dict.fromkeys(out))
    g = {}
    for s_ in out:
        pid, d_, c = s_.split(",")
        if int(pid) in ids:
            g.setdefault(pid, []).append(d_ + c)
    return "\n".join(f"{k}:{' '.join(sorted(v))}" for k, v in sorted(g.items()))


def woba_parts(t):
    b1 = t["hits"] - t["doubles"] - t["triples"] - t["homeRuns"]
    ubb = t["baseOnBalls"] - t["intentionalWalks"]
    num = W["ubb"] * ubb + W["hbp"] * t["hitByPitch"] + W["b1"] * b1 + W["b2"] * t["doubles"] + W["b3"] * t["triples"] + W["hr"] * t["homeRuns"]
    den = t["atBats"] + ubb + t["sacFlies"] + t["hitByPitch"]
    return num, den


def marcel(pairs):
    S, LG = {}, {}
    for y in range(2021, 2026):
        sp = get(f"/stats?stats=season&group=hitting&season={y}&sportId=1&playerPool=ALL&limit=5000")["stats"][0]["splits"]
        S[y], tn, td = {}, 0.0, 0.0
        for x in sp:
            n, dn = woba_parts(x["stat"])
            S[y][str(x["player"]["id"])] = (n, dn, x["stat"]["plateAppearances"], x["stat"].get("age"))
            tn += n; td += dn
        LG[y] = tn / td
    out = []
    for pid, ys in pairs.items():
        pid = str(pid)
        for c in ys:
            Y = 2020 + int(c)
            wn = wd = lgw = wsum = 0.0
            age = None
            for k, w in zip((1, 2, 3), (5, 4, 3)):
                s = S.get(Y - k, {}).get(pid)
                lgw += w * LG[Y - k]; wsum += w
                if s:
                    wn += w * s[0]; wd += w * s[1]
                    if age is None and s[3]:
                        age = s[3] + k
            rate = (wn + 1200 * lgw / wsum) / (wd + 1200)
            rel = wd / (wd + 1200)
            if age is not None:
                rate *= 1 + 0.006 * (29 - age) if age < 29 else 1 - 0.003 * (age - 29)
            prev = S.get(Y - 1, {}).get(pid, (0, 0, 0, 0))[2]
            out.append(f"{pid}:{c}:{jsround(rate * 1000)}:{jsround(rel * 100)}:{prev}:{'' if age is None else age}")
    return " ".join(out)


def post():
    o = []
    for y in (2024, 2025):
        for x in get(f"/stats?stats=season&group=hitting&season={y}&sportId=1&playerPool=ALL&limit=5000&gameType=P")["stats"][0]["splits"]:
            t = x["stat"]
            if not t["plateAppearances"]:
                continue
            n, dn = woba_parts(t)
            o.append(f"{x['player']['id']}:{y % 10}:{jsround(n * 1000)}:{dn}:{t['plateAppearances']}")
    return " ".join(o)


if __name__ == "__main__":
    sc, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    ids300, pairs = panel_ids(sc)
    for name, txt in [("il_transactions_hitters_2024_2026.txt", il(ids300)), ("marcel_woba_2024_2026.txt", marcel(pairs)),
                      ("postseason_woba_2024_2025.txt", post())]:
        open(os.path.join(out, name), "w").write(txt + "\n")
        print(name, len(txt), "checksum", chk(txt))
    print("pull-time checksums: IL 516648570, Marcel 267097853, postseason 2416212279 "
          "(IL and postseason can differ if MLB edits a transaction or stat line after 2026-09-30)")

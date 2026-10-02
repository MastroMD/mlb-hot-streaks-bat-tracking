"""Synthetic Statcast-shaped data with a controllable hot-hand state, to validate the pipeline.
python3 simulate.py OUT_DIR HOT_SD   (HOT_SD=0 -> no hot hand; e.g. 1.0 -> real streaks)"""
import os
import sys

import numpy as np
import pandas as pd


# ============================================================================ A11d (offseason analysis, plan addendum C)
# Game-level generator on the REAL panel skeleton: every hitter-season's games, dates, PA and competitive-swing counts
# are kept; only bat speed (bs_num) is replaced:
#   bs_num_g = comp_g * (mu_h + L_hg + u_g) + sqrt(comp_g) * sigma_sw * z
# L_hg is the latent level: none (a), AR(1) per PA (b), one permanent step per hitter-season (c), or both (d).
# Unit-scale draws are generated once per replicate (common random numbers) and scaled, so calibration is smooth.

def a11_skeleton(bg):
    """Per-hitter arrays in (batter, date, game) order."""
    bg = bg.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)
    out = []
    for b, idx in bg.groupby("batter", sort=False).indices.items():
        idx = np.asarray(idx)
        yrs = bg["game_year"].to_numpy()[idx]
        days = bg["game_date"].to_numpy()[idx].astype("datetime64[D]").astype(np.int64)
        pa = bg["pa"].to_numpy()[idx].astype(float)
        out.append({"batter": b, "idx": idx, "yrs": yrs, "days": days, "pa": pa})
    return bg, out


def a11_draws(skel, n_rows, jumps_abs, seed):
    """Unit-scale random components for one replicate."""
    rng = np.random.default_rng(seed)
    D = {"z": rng.standard_normal(n_rows), "u": rng.standard_normal(n_rows), "ar": np.zeros(n_rows),
         "step": np.zeros(n_rows), "delta": np.zeros(n_rows), "_seed": seed}
    D["ar_eps"] = rng.standard_normal(n_rows)
    D["ar_init"] = {}
    for h in skel:
        idx, yrs = h["idx"], h["yrs"]
        D["ar_init"][h["batter"]] = rng.standard_normal()
        lvl, dl = 0.0, 0.0
        first = True
        for y in np.unique(yrs):
            m = np.flatnonzero(yrs == y)
            if not first:
                dl += rng.standard_normal()
            first = False
            D["delta"][idx[m]] = dl
            k = rng.integers(1, len(m)) if len(m) > 1 else 0     # step takes effect at game k (uniform over games 2..n)
            size = rng.choice([-1.0, 1.0]) * rng.choice(jumps_abs)
            seg = np.zeros(len(m)); seg[:k] = lvl; seg[k:] = lvl + size
            D["step"][idx[m]] = seg
            lvl += size
    return D


def a11_ar(skel, D, hl):
    """AR(1) state with half-life hl PA (unit stationary SD). Decay between games = 0.5**(PA of previous game / hl);
    across an offseason = 0.5**(gap days * hitter's PA per season-day in the previous season / hl)."""
    s_out = np.zeros(len(D["z"]))
    eps = D["ar_eps"]
    for h in skel:
        idx, yrs, days, pa = h["idx"], h["yrs"], h["days"], h["pa"]
        s = D["ar_init"][h["batter"]]
        for i in range(len(idx)):
            if i > 0:
                if yrs[i] != yrs[i - 1]:
                    prev = yrs == yrs[i - 1]
                    span = max(days[prev].max() - days[prev].min(), 1)
                    gap_pa = (days[i] - days[i - 1]) * pa[prev].sum() / span
                    rho = 0.5 ** (gap_pa / hl)
                else:
                    rho = 0.5 ** (pa[i - 1] / hl)
                s = rho * s + np.sqrt(1 - rho * rho) * eps[idx[i]]
            s_out[idx[i]] = s
    return s_out


def a11_world(bg, D, mu, sigma_sw, sigma_g, ar=None, ar_sd=0.0, step_k=0.0, sigma_delta=0.0):
    """Return a copy of bg (sorted as in a11_skeleton) with bs_num replaced."""
    comp = bg["comp"].to_numpy().astype(float)
    level = mu + sigma_delta * D["delta"] + sigma_g * D["u"]
    if ar is not None and ar_sd > 0:
        level = level + ar_sd * ar
    if step_k > 0:
        level = level + step_k * D["step"]
    out = bg.copy()
    out["bs_num"] = np.where(comp > 0, comp * level + np.sqrt(comp) * sigma_sw * D["z"], 0.0)
    return out


if __name__ == "__main__":

    out, hot = sys.argv[1], float(sys.argv[2])
    rng = np.random.default_rng(1 if hot else 2)
    os.makedirs(out, exist_ok=True)
    NB, SEAS, G = 160, [2024, 2025, 2026], 150
    teams = [f"T{i:02d}" for i in range(30)]
    pitchers = pd.DataFrame({"pitcher": np.arange(900000, 900500), "p_throws": rng.choice(["R", "L"], 500, p=[.7, .3]),
                             "pq": rng.normal(0, 0.02, 500)})
    rows = []
    pk = 700000
    for b in range(NB):
        bid = 600000 + b
        stand = rng.choice(["R", "L"], p=[.6, .4])
        tal = {"bs": rng.normal(72, 2.5), "woba": rng.normal(0.320, 0.03), "chase": rng.normal(0.28, 0.05)}
        for yr in SEAS:
            # AR(1) latent form, per game; stationary sd = 1
            rho = 0.93
            e = rng.normal(0, np.sqrt(1 - rho ** 2), G)
            s = np.zeros(G); s[0] = rng.normal()
            for g in range(1, G):
                s[g] = rho * s[g - 1] + e[g]
            s *= hot
            for g in range(G):
                if rng.random() < 0.1:
                    continue
                date = pd.Timestamp(f"{yr}-04-01") + pd.Timedelta(days=int(g * 1.2))
                gpk = yr * 1000 + g  # shared ids are fine
                for ab in range(4):
                    p = pitchers.iloc[rng.integers(500)]
                    same = (p.p_throws == stand)
                    q = tal["woba"] + 0.012 * s[g] - 0.012 * same + p.pq
                    pk_ = 0.22 - 0.4 * (q - 0.32)
                    pbb = 0.085 + 0.2 * (q - 0.32)
                    u = rng.random()
                    outcome = "K" if u < pk_ else ("BB" if u < pk_ + pbb else "BIP")
                    npitch = 1 + rng.poisson(2.8)
                    for pn in range(1, npitch + 1):
                        last = pn == npitch
                        inz = rng.random() < 0.48
                        if last:
                            desc = {"K": "swinging_strike", "BB": "ball", "BIP": "hit_into_play"}[outcome]
                            inz = inz if outcome != "BB" else False
                        else:
                            sw = rng.random() < (0.67 if inz else tal["chase"] + 0.02 * s[g])
                            desc = (rng.choice(["foul", "swinging_strike"], p=[.7, .3]) if sw
                                    else ("called_strike" if inz else "ball"))
                        swing = desc in ("foul", "swinging_strike", "hit_into_play")
                        bs = tal["bs"] + 0.8 * s[g] + rng.normal(0, 3) if swing else np.nan
                        r = dict(game_date=date, game_pk=gpk * 100 + b, game_type="R", game_year=yr,
                                 at_bat_number=ab + 1, pitch_number=pn, batter=bid, pitcher=p.pitcher,
                                 stand=stand, p_throws=p.p_throws, home_team=teams[b % 30], away_team=teams[(b + 7) % 30],
                                 zone=(5 if inz else 12), description=desc, release_speed=rng.normal(89, 5),
                                 bat_speed=bs, swing_length=(7.2 + 0.05 * (bs - 72) + rng.normal(0, .3)) if swing else np.nan,
                                 type=("X" if desc == "hit_into_play" else ("B" if desc == "ball" else "S")),
                                 events=None, woba_value=np.nan, woba_denom=np.nan, launch_speed=np.nan,
                                 launch_angle=np.nan, launch_speed_angle=np.nan, estimated_woba_using_speedangle=np.nan)
                        if last:
                            r["woba_denom"] = 1
                            if outcome == "K":
                                r.update(events="strikeout", woba_value=0.0)
                            elif outcome == "BB":
                                r.update(events="walk", woba_value=0.69)
                            else:
                                ev = 88 + 1.0 * (bs - 72) + 150 * (q - 0.32) + rng.normal(0, 13)
                                la = rng.normal(12, 25)
                                xw = float(np.clip(0.37 + 0.02 * (ev - 88) - 0.0002 * (la - 18) ** 2, 0, 2))
                                hit = rng.random() < min(xw / 1.4, 0.95)
                                r.update(launch_speed=ev, launch_angle=la, estimated_woba_using_speedangle=xw,
                                         launch_speed_angle=6 if (ev > 98 and 24 < la < 32) else 4,
                                         events="single" if hit else "field_out", woba_value=(1.2 if hit else 0.0))
                        rows.append(r)
    df = pd.DataFrame(rows)
    for yr in SEAS:
        df[df.game_year == yr].to_parquet(os.path.join(out, f"sim_{yr}.parquet"), index=False)
    print(len(df), "pitches")

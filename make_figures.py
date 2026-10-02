"""Draw the paper's figures from results/paper_results.json (no estimation here).

python3 make_figures.py RESULTS_JSON OUT_DIR [--tif]
Writes fig1_persistence.png, fig2_deciles.png, fig3_platoon.png, fig4_decay.png, fig5_pitchers.png,
figA1_simulation.png and figA2_a11.png at 300 dpi. The abstract's Figure 1 is drawn by sloan/make_figure1.py.
PNG files are written without text metadata. --tif writes 300-dpi LZW-compressed TIFF files instead (the JQAS figure
format), also without text metadata.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

P = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
TIF = "--tif" in sys.argv[3:]
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e1", "#cde2fb"
MINUS = "−"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.unicode_minus": True})


def save(fig, name):
    if TIF:
        name = name.rsplit(".", 1)[0] + ".tif"
        import io
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white", metadata={"Software": None})
        Image.open(buf).convert("RGB").save(os.path.join(OUT, name), format="TIFF", compression="tiff_lzw", dpi=(300, 300))
    else:
        fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight", facecolor="white", metadata={"Software": None})
    plt.close(fig)
    print("wrote", name)


def m(x, k=1):
    return f"{x:.{k}f}".replace("-", MINUS)


PERS = P["primary"]["persistence"]
LABEL = {"swing_len": "Swing length", "bat_speed": "Bat speed", "fast_swing": "Fast-swing rate", "z_swing": "In-zone swing rate",
         "chase": "Chase rate", "whiff": "Whiff rate", "barrel": "Barrel rate", "k_rate": "Strikeout rate", "ev": "Exit velocity",
         "xwoba": "xwOBA", "hard_hit": "Hard-hit rate", "woba": "wOBA"}
SWINGSIDE = {"swing_len", "bat_speed", "fast_swing", "z_swing", "chase"}


def persistence_bars(ax, title=True):
    order = sorted(PERS, key=lambda k: -PERS[k]["persistence"])
    y = np.arange(len(order))[::-1]
    for yi, k in zip(y, order):
        v, se = 100 * PERS[k]["persistence"], 100 * PERS[k]["se"]
        col = BLUE if k in SWINGSIDE else ORANGE
        ax.barh(yi, v, height=0.62, color=col, zorder=2)
        ax.errorbar(v, yi, xerr=se, fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
        ax.text(max(v, 0) + se + 0.6, yi, f"{m(v)}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[k] for k in order])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-3, 32)
    ax.set_xlabel("Persistence beyond shuffled null (%)")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Swing mechanics and decisions"), Patch(color=ORANGE, label="Contact quality and results")],
              loc="lower right", frameon=False, fontsize=9)
    if title:
        ax.set_title("What carries over (50 PA → next 50 PA)")


# ---------------------------------------------------------------- Figure 1
fig, ax = plt.subplots(figsize=(6.5, 4.2))
persistence_bars(ax)
save(fig, "fig1_persistence.png")

# ---------------------------------------------------------------- Figure 2 deciles
forms = P["primary"]["forms"]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
xs = np.arange(1, 11)
for key, col, lab, off, mk in (("swing_only", BLUE, "Swing-form index", -0.12, "o"), ("results_only", ORANGE, "wOBA streak index", 0.12, "s")):
    d = forms[key]["deciles"]
    b, s = np.array(d["bump"]), np.array(d["se"])
    ref = np.isin(np.arange(10), [4, 5])
    ax.errorbar(xs[~ref] + off, b[~ref], yerr=s[~ref], fmt=mk, color=col, ms=5.5, elinewidth=1, capsize=2, label=lab, zorder=3)
    ax.plot(xs[ref] + off, b[ref], mk, mfc="white", mec=col, ms=5.5, zorder=3)
    ax.plot(xs + off, b, color=col, lw=1, alpha=0.6, zorder=2)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(xs)
ax.set_xlabel("Decile of the recent-form index (out of sample)")
ax.set_ylabel("Next-50-PA wOBA vs forecast\n(points, vs deciles 5–6)")
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_title("What recent form is worth")
save(fig, "fig2_deciles.png")

# ---------------------------------------------------------------- Figure 3 platoon vs form
B3 = P["B3"]["rules"]["prespecified"]
A7 = P["A7"]["prespecified"]
pl = P["primary"]["platoon"]
sw = forms["swing_only"]["deciles"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
rows = [("Swing form, bottom decile", -sw["bump"][0], sw["se"][0], BLUE, None),
        ("Swing form, top decile", sw["bump"][9], sw["se"][9], BLUE, None),
        ("Platoon penalty, RHB", pl["RHB_league_split_pts"], None, ORANGE, B3["hitter_split_pts"]["R"]["p10_p50_p90"]),
        ("Platoon penalty, LHB", pl["LHB_league_split_pts"], None, ORANGE, B3["hitter_split_pts"]["L"]["p10_p50_p90"])]
for i, (lab, v, se, col, rng) in enumerate(rows):
    a1.barh(i, v, height=0.55, color=col, zorder=2)
    if se:
        a1.errorbar(v, i, xerr=se, fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
    if rng:
        a1.plot([rng[0], rng[2]], [i + 0.38, i + 0.38], color=INK, lw=1.4, zorder=3)
        a1.plot([rng[1]], [i + 0.38], "|", color=INK, ms=7, zorder=3)
    a1.text(v + (se or 0) + 0.8, i - 0.05, m(v), va="center", fontsize=9)
a1.set_yticks(range(len(rows)))
a1.set_yticklabels([r[0] for r in rows])
a1.tick_params(axis="y", length=0)
a1.set_xlabel("wOBA points")
a1.set_xlim(0, 44)
a1.grid(axis="x", color=GRID, lw=0.8, zorder=0)
a1.set_title("A. Form vs the platoon advantage")
opts = [("(b) Ride the better\nrecent wOBA", "option_b_vs_strict_platoon_runs_per50", "runs_b", ORANGE),
        ("(c0) Talent +\nown split", "option_c0_vs_strict_platoon_runs_per50", "runs_c0", BLUE),
        ("(c) Talent + own\nsplit + form", "option_c_vs_strict_platoon_runs_per50", "runs_c", BLUE)]
for i, (lab, key, bk, col) in enumerate(opts):
    mean = A7[key]["mean"]
    q = B3["per_pair_p10_p50_p90"][bk]
    a2.plot([q[0], q[2]], [i, i], color=col, lw=2.2, alpha=0.45, zorder=2)
    a2.plot([mean], [i], "o", color=col, ms=7, zorder=3)
    a2.text(mean, i + 0.22, f"{mean:+.2f}".replace("-", MINUS), ha="center", fontsize=8.5)
a2.axvline(0, color=INK2, lw=0.8)
a2.set_yticks(range(len(opts)))
a2.set_yticklabels([o[0] for o in opts], fontsize=8.5)
a2.tick_params(axis="y", length=0)
a2.set_ylim(-0.6, len(opts) - 0.4)
a2.invert_yaxis()
a2.set_xlabel("Expected runs per 50 PA vs strict platoon")
a2.grid(axis="x", color=GRID, lw=0.8, zorder=0)
a2.set_title("B. Lineup rules")
fig.tight_layout(w_pad=2.0)
save(fig, "fig3_platoon.png")

# ---------------------------------------------------------------- Figure 4 decay with calendar axis
fx = P["A1"]["horizons_fixed_sample"]
HW = P["horizon_weeks"]
H = ["25", "50", "100", "150", "200"]
days = np.array([HW[h]["median_days"] for h in H])
fig, ax = plt.subplots(figsize=(6.5, 3.8))
SER = [("bat_speed", "Bat speed", BLUE, "o"), ("swing_len", "Swing length", AQUA, "s"), ("fast_swing", "Fast-swing rate", MAGENTA, "^"),
       ("z_swing", "In-zone swing rate", YELLOW, "D"), ("woba", "wOBA", ORANGE, "v")]
for k, lab, col, mk in SER:
    v = np.array([100 * fx[h][k]["persistence"] for h in H])
    s = np.array([100 * fx[h][k]["se"] for h in H])
    if k == "bat_speed":
        ax.fill_between(days, v - s, v + s, color=BAND, zorder=1, lw=0)
    ax.plot(days, v, marker=mk, color=col, lw=1.8, ms=5, label=lab, zorder=3, mec="white", mew=0.6)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlim(0, days[-1] + 5)
ax.set_ylim(-3, 30)
ax.set_xticks([0, 7, 14, 21, 28, 35, 42, 49, 56, 63])
ax.set_xticklabels(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
ax.set_ylim(-3, 33)
ax.set_xlabel("Weeks ahead (median calendar span of the forward window)")
ax.set_ylabel("Persistence beyond shuffled null (%)")
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
top = ax.secondary_xaxis("top")
top.set_xticks(days)
top.set_xticklabels([f"{h} PA" for h in H], fontsize=8.5)
top.tick_params(colors=INK2)
ax.legend(frameon=False, loc="upper right", fontsize=8.5, ncol=2)
ax.set_title("How long form lasts", pad=22)
save(fig, "fig4_decay.png")

# ---------------------------------------------------------------- Figure 5 pitchers
A5 = P["A5"]["persistence"]
fig, ax = plt.subplots(figsize=(6.5, 3.9))
order = sorted(A5, key=lambda k: -A5[k]["persistence"])
y = np.arange(len(order))[::-1]
for yi, k in zip(y, order):
    v, se = 100 * A5[k]["persistence"], 100 * A5[k]["se"]
    deliv = k.startswith("ff_")
    ax.barh(yi, v, height=0.62, color=BLUE if deliv else ORANGE, zorder=2, hatch="///" if k == "ff_arm" else None,
            edgecolor="white" if k == "ff_arm" else None, lw=0)
    ax.errorbar(v, yi, xerr=se, fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
    ax.text(v + se + 0.6, yi, f"{m(v)}%", va="center", fontsize=9)
lab5 = {k: A5[k]["label"].replace("K-BB%", "K" + MINUS + "BB%") for k in A5}
ax.set_yticks(y)
ax.set_yticklabels([lab5[k] for k in order])
ax.tick_params(axis="y", length=0)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlim(-3, 44)
ax.set_xlabel("Persistence beyond shuffled null (%)")
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=BLUE, label="Delivery (four-seam)"), Patch(color=ORANGE, label="Outcomes and decisions")],
          loc="lower right", frameon=False, fontsize=9)
ax.set_title("Pitchers (200 pitches → next 200)")
save(fig, "fig5_pitchers.png")

# ---------------------------------------------------------------- Figure A1 simulation
SIM = P["simulation"]
fig, ax = plt.subplots(figsize=(5.2, 2.8))
mets = [("bat_speed", "Bat speed"), ("chase", "Chase rate"), ("woba", "wOBA")]
for j, (world, col, lab) in enumerate((("hot", BLUE, "Hot-hand state injected"), ("null", ORANGE, "No state"))):
    for i, (k, _) in enumerate(mets):
        v, se = 100 * SIM[world][k]["persistence"], 100 * SIM[world][k]["se"]
        ax.bar(i + (j - 0.5) * 0.36, v, width=0.34, color=col, zorder=2, label=lab if i == 0 else None)
        ax.errorbar(i + (j - 0.5) * 0.36, v, yerr=se, fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(len(mets)))
ax.set_xticklabels([x[1] for x in mets])
ax.set_ylabel("Recovered persistence (%)")
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.legend(frameon=False, fontsize=8.5)
ax.set_title("Pipeline check on simulated seasons")
save(fig, "figA1_simulation.png")

# ---------------------------------------------------------------- Figure A2 A11 signatures
sig = P["A11"]["signature"]
W = [("a_none", "(a) No state"), ("b_hl50", "(b) AR, half-life 50 PA"), ("b_hl150", "(b) AR, 150 PA"), ("b_hl400", "(b) AR, 400 PA"),
     ("c_step", "(c) Lasting step"), ("d_mix", "(d) Mix")]
ci = {"b": None, "R_next": None}
a11 = P["A11"]
ci["b"] = (a11["b"] - 1.96 * a11["b_se"], a11["b"] + 1.96 * a11["b_se"])
ci["R_next"] = (a11["R_next"] - 1.96 * a11["R_next_se"], a11["R_next"] + 1.96 * a11["R_next_se"])
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
for ax, (k, title) in zip(axs, (("b", "A. Offseason carry-over slope"), ("R_next", "B. Step retention into next season"))):
    ax.axvspan(*ci[k], color=BAND, zorder=0, lw=0)
    ax.axvline(sig["real"][k], color=BLUE, lw=1.8, zorder=1)
    for i, (w, lab) in enumerate(W):
        mu, sd = sig[w][k]["mean"], sig[w][k]["sd"]
        ax.errorbar(mu, i, xerr=sd, fmt="o", color=INK, ms=4.5, elinewidth=1, capsize=2.5, zorder=3)
    ax.set_yticks(range(len(W)))
    ax.set_yticklabels([x[1] for x in W])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.1, 1.15)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch as _Patch  # noqa: E402
fig.tight_layout(w_pad=1.5, rect=(0, 0.07, 1, 1))
fig.legend(handles=[Line2D([0], [0], color=BLUE, lw=1.8, label="Real estimate"), _Patch(color=BAND, label="Real 95% CI"),
                    Line2D([0], [0], marker="o", color=INK, lw=0, label="Simulated world, mean ± SD")],
           loc="lower center", ncol=3, frameon=False, fontsize=8.5)
save(fig, "figA2_a11.png")

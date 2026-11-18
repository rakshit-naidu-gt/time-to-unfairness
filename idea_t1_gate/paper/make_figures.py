"""
Generate all paper figures for Time-to-Unfairness.

Figures produced:
  fig1_degradation_trajectories.pdf  — degradation paths, sex attribute, 3 tasks;
                                        income panel also overlays the
                                        discontinuity-corrected fit (see
                                        discontinuity_correction.py)
  fig2_hr_summary.pdf                — HR forest plot w/ bootstrap CI error bars,
                                        sex and race, all tasks; income shown both
                                        naive and discontinuity-corrected
  fig3_detection_delay.pdf           — Gantt-style SSE timeline, employment/race
                                        task (genuine, uncontaminated by the
                                        income-task 2021 discontinuity)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS = Path(__file__).parent.parent / "results"
EXT     = RESULTS / "extended"
OUT     = Path(__file__).parent

TAU = 0.05   # degradation threshold
W   = 20     # deployment window (years)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

C_MIN  = "#c0392b"   # minority / Female / Non-White  (crimson)
C_MAJ  = "#2980b9"   # majority / Male   / White      (steel blue)
C_TAU  = "#27ae60"   # threshold line                 (green)
C_COMP = "#d5f5e3"   # compliant region               (light green)
C_NCOM = "#fadbd8"   # non-compliant region           (light red)
C_GRID = "#ecf0f1"   # grid                           (light grey)

# ── Data ──────────────────────────────────────────────────────────────────────

TASKS = {
    "income_compat": {
        "label":  "Income > \$50k\n(2014–2022, 8 snapshots)",
        "short":  "Income",
        "dir":    EXT / "income_compat",
        "hr_sex": 1.905, "hr_race": 0.185,  # naive (artifact, see paper S6.2)
        "hr_sex_corrected": 1.090, "hr_race_corrected": 1.103,
        "hr_sex_ci": (1.06, 2.71), "hr_race_ci": (0.03, 1.11),  # naive
        "hr_sex_corrected_ci": (1.00, 1.38), "hr_race_corrected_ci": (0.84, 1.26),
        "r_sex":  {"Male": 0.003146, "Female": 0.005994},
        "r_race": {"White": 0.011561, "NonWhite": 0.002141},
        "ttc_sex": {"Male": 15.9, "Female": 8.3},
        "ttc_race": {"White": 4.3, "NonWhite": 23.4},
        # discontinuity-corrected fit: Delta_g(t) = r*t + c + delta*1[t>=7]
        "corrected_sex": {
            "Male":   {"r": 0.004524, "c": 0.001187, "delta": -0.010195},
            "Female": {"r": 0.004931, "c": 0.003077, "delta": 0.007864},
        },
        "ttc_sex_corrected": {"Male": 10.8, "Female": 9.5},
        "ttc_race_corrected": {"White": 10.3, "NonWhite": 9.3},
    },
    "employment": {
        "label":  "Employment Status\n(2014–2018, 5 snapshots)",
        "short":  "Employment",
        "dir":    EXT / "employment",
        "hr_sex": 1.072, "hr_race": 0.493,
        "hr_sex_ci": None, "hr_race_ci": (0.12, 0.80),
        "r_sex":  {"Male": 0.001038, "Female": 0.001113},
        "r_race": {"White": 0.002601, "NonWhite": 0.001282},
        "ttc_sex": {"Male": 48.2, "Female": 44.9},
        "ttc_race": {"White": 19.2, "NonWhite": 39.0},
    },
    "coverage": {
        "label":  "Public Health Coverage\n(2014–2022, 8 snapshots)",
        "short":  "Coverage",
        "dir":    EXT / "coverage",
        "hr_sex": None, "hr_race": None,   # one-sided
        "r_sex":  {"Male": 0.0, "Female": 0.001082},
        "r_race": {"White": 0.0, "NonWhite": 0.002593},
        "ttc_sex": {"Male": None, "Female": 46.1},
        "ttc_race": {"White": None, "NonWhite": 19.3},
    },
    "poverty": {
        "label":  "Income < 250% Poverty\n(2014–2018, 5 snapshots)",
        "short":  "Poverty",
        "dir":    EXT / "poverty",
        "hr_sex": 1.249, "hr_race": 1.416,
        "hr_sex_ci": (0.84, 1.70), "hr_race_ci": (1.07, 1.62),
        "r_sex":  {"Male": 0.007137, "Female": 0.008913},
        "r_race": {"White": 0.007200, "NonWhite": 0.010198},
        "ttc_sex": {"Male": 7.0, "Female": 5.6},
        "ttc_race": {"White": 6.9, "NonWhite": 4.9},
    },
}

def load_deg(task_key: str, attr: str) -> pd.DataFrame:
    path = TASKS[task_key]["dir"] / "degradation.csv"
    df = pd.read_csv(path)
    return df[df["attr"] == attr].copy()

def fit_line(t_arr, d_arr):
    """OLS through origin: slope = sum(t*d)/sum(t^2)."""
    if len(t_arr) < 2 or t_arr.max() == 0:
        return 0.0
    return float(np.sum(t_arr * d_arr) / np.sum(t_arr ** 2))

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Degradation trajectories (sex attribute, 2×2 grid)
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig1():
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.1), sharey=False)
    task_order = ["income_compat", "employment", "poverty"]
    panel_titles = {
        "income_compat": "(a) Income, 2014–2022",
        "employment":     "(b) Employment, 2014–2018",
        "poverty":        "(c) Poverty, 2014–2018",
    }

    for ax, tk in zip(axes, task_order):
        cfg = TASKS[tk]
        df  = load_deg(tk, "sex")
        corrected = cfg.get("corrected_sex")

        for group, color, marker, lbl in [
            ("Male",   C_MAJ, "s", "Male (majority)"),
            ("Female", C_MIN, "o", "Female (minority)"),
        ]:
            sub = df[df["group"] == group].sort_values("t")
            if sub.empty:
                continue
            t_arr = sub["t"].values.astype(float)
            d_arr = sub["degradation"].values

            # Scatter
            ax.scatter(sub["year"], d_arr, color=color, marker=marker,
                       s=50, zorder=5, label=lbl)

            # Naive fitted line (through origin) from t=0 to end-of-data
            slope = fit_line(t_arr, d_arr)
            t_fit = np.linspace(0, t_arr.max(), 200)
            naive_lbl = "Naive fit" if (tk == "income_compat" and group == "Male") else None
            ax.plot(sub["year"].min() + t_fit,
                    slope * t_fit,
                    color=color, linestyle="--", linewidth=1.5, alpha=0.85,
                    label=naive_lbl)

            # Discontinuity-corrected fit overlay (income task only)
            if corrected is not None and group in corrected:
                p = corrected[group]
                t_full = np.linspace(0, t_arr.max(), 200)
                pred = p["r"] * t_full + p["c"] + np.where(t_full >= 7, p["delta"], 0.0)
                corr_lbl = "Corrected fit" if group == "Male" else None
                ax.plot(sub["year"].min() + t_full, pred,
                        color=color, linestyle=":", linewidth=2.0, alpha=0.95,
                        label=corr_lbl)

        # τ threshold
        ax.axhline(TAU, color=C_TAU, linestyle=":", linewidth=1.4,
                   label=f"$\\tau = {TAU}$" if tk == "income_compat" else None)

        # HR annotation
        if tk == "income_compat":
            hr_str = (f"naive $\\widehat{{\\mathrm{{HR}}}} = {cfg['hr_sex']:.2f}$\n"
                      f"corrected $\\widehat{{\\mathrm{{HR}}}} = {cfg['hr_sex_corrected']:.2f}$")
            ax.text(0.97, 0.80, hr_str, transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", alpha=0.9))
        else:
            hr = cfg["hr_sex"]
            hr_str = f"$\\widehat{{\\mathrm{{HR}}}}_{{\\rm sex}} = {hr:.2f}$" \
                     if hr is not None else "one-sided\n(Male stable)"
            ax.text(0.97, 0.97, hr_str, transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", alpha=0.9))

        ax.set_title(panel_titles[tk], fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel(r"Degradation $\Delta_g(t)$")
        ax.set_ylim(bottom=0)
        ax.set_xlim(df["year"].min() - 0.4, df["year"].max() + 0.4)
        ax.grid(True, color=C_GRID, linewidth=0.8)

    # Shared legend on the last (empty) axis or figure
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", ncol=5, frameon=True,
               bbox_to_anchor=(0.5, -0.08), fontsize=8.5)
    plt.tight_layout()
    path = OUT / "fig1_degradation_trajectories.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — HR forest plot (sex and race, all tasks)
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig2():
    # Data: (row_label, hr_sex, ci_sex, hr_race, ci_race)
    rows = [
        ("Income (naive)",     1.905, (1.06, 2.71), 0.185, (0.03, 1.11)),
        ("Income (corrected)", 1.090, (1.00, 1.38), 1.103, (0.84, 1.26)),
        ("Employment",         1.072, None,         0.493, (0.12, 0.80)),
        ("Poverty",            1.249, (0.84, 1.70), 1.416, (1.07, 1.62)),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)

    task_names = [r[0] for r in rows]
    y_pos = np.arange(len(task_names))[::-1]  # naive income on top

    for ax, hr_idx, ci_idx, title in [(ax1, 1, 2, "Sex (Female / Male)"),
                                        (ax2, 3, 4, "Race (Non-White / White)")]:
        for i, row in enumerate(rows):
            hr, ci = row[hr_idx], row[ci_idx]
            y = y_pos[i]
            if hr is None:
                ax.scatter([1.0], [y], marker="D", s=80,
                           facecolors="none", edgecolors="#888", linewidths=1.5,
                           zorder=4)
                ax.text(1.06, y, "n.e.", va="center", ha="left",
                        fontsize=7.5, color="#888")
                continue

            color = C_MIN if hr > 1 else C_MAJ
            faded = row[0].startswith("Income (naive)")
            alpha = 0.55 if faded else 1.0
            if ci is not None:
                lo, hi = ci
                ax.errorbar([hr], [y], xerr=[[hr - lo], [hi - hr]],
                            fmt="none", ecolor=color, elinewidth=1.6,
                            capsize=3, alpha=alpha, zorder=4)
            ax.scatter([hr], [y], color=color, s=80, zorder=5, alpha=alpha)
            ax.plot([1.0, hr], [y, y], color=color, linewidth=1.6,
                    alpha=0.5 * alpha, zorder=3)
            lbl = f"{hr:.2f}"
            ax.text(hr, y - 0.32, lbl, va="top", ha="center",
                    fontsize=8, color=color, fontweight="bold", alpha=alpha)

        ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--", zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(task_names)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Hazard ratio", labelpad=4)
        ax.set_xlim(0.0, 2.9)
        ax.set_ylim(y_pos.min() - 0.6, y_pos.max() + 0.6)
        ax.grid(True, axis="x", color=C_GRID, linewidth=0.8)
        ax.axvspan(0.0, 1.0, alpha=0.04, color=C_MAJ)
        ax.axvspan(1.0, 2.9, alpha=0.04, color=C_MIN)

    # Shared legend
    leg_handles = [
        mpatches.Patch(color=C_MIN, label="HR > 1: minority certificate expires sooner"),
        mpatches.Patch(color=C_MAJ, label="HR < 1: majority certificate expires sooner"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#888",
                   markersize=7, alpha=0.55, label="Naive (uncorrected) income estimate"),
    ]
    fig.legend(handles=leg_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.14), fontsize=8.5, frameon=False)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = OUT / "fig2_hr_summary.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Detection-delay SSE timeline (employment/race task; genuine,
# uncontaminated by the income-task 2021 composition discontinuity)
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig3():
    # Employment task, race attribute: genuine, uncontaminated by the
    # income-task 2021 composition discontinuity (window ends 2018).
    TTC_FAST = 19.2   # White (majority) -- faster-drifting here
    TTC_SLOW = 39.0   # Non-White (minority)
    W_win = 20
    T_audit = 1

    fig, ax = plt.subplots(figsize=(9, 2.8))

    groups = [
        ("White (majority)",     TTC_FAST, C_MAJ, 1.0),
        ("Non-White (minority)", TTC_SLOW, C_MIN, 0.0),
    ]

    bar_h = 0.38

    for label, ttc, color, y in groups:
        ttc_disp = min(ttc, W_win)  # clamp for drawing; TTC_SLOW=39 exceeds W=20
        # Compliant period
        ax.barh(y, ttc_disp, left=0, height=bar_h,
                color=C_COMP, edgecolor="#aaa", linewidth=0.5, zorder=3)
        # Non-compliant period (zero width if TTC exceeds the window)
        nc_width = max(0.0, W_win - ttc_disp)
        if nc_width > 0:
            ax.barh(y, nc_width, left=ttc_disp, height=bar_h,
                    color=C_NCOM, edgecolor="#aaa", linewidth=0.5, zorder=3)

        # TTC marker (only drawn if expiry falls inside the window)
        if ttc <= W_win:
            ax.axvline(ttc, ymin=y / (len(groups) + 0.5) - 0.05,
                       ymax=y / (len(groups) + 0.5) + 0.35,
                       color=color, linewidth=2.0, zorder=5)
            ax.text(ttc + 0.15, y + bar_h / 2 + 0.05,
                    f"$\\widehat{{\\mathrm{{TTC}}}}={ttc}\\,\\mathrm{{yr}}$",
                    va="bottom", ha="left", fontsize=8.5, color=color)
        else:
            ax.text(W_win - 0.3, y + bar_h / 2 + 0.05,
                    f"$\\widehat{{\\mathrm{{TTC}}}}={ttc}\\,\\mathrm{{yr}} > W$",
                    va="bottom", ha="right", fontsize=8.5, color=color)

        # SSE label
        sse = max(0.0, W_win - ttc)
        if nc_width > 0.5:
            ax.text(ttc_disp + nc_width / 2, y, f"SSE = {sse:.1f} yr",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="#c0392b")
        else:
            ax.text(W_win - 0.3, y - bar_h / 2 - 0.08, f"SSE = {sse:.1f} yr",
                    ha="right", va="top", fontsize=8, fontweight="bold",
                    color="#27ae60")

    # Annual audit markers
    for t in range(T_audit, W_win + 1, T_audit):
        ax.axvline(t, color="#bdc3c7", linewidth=0.6, linestyle=":", zorder=2)

    # Inequity brace annotation (both clamped to the visible window)
    ttc_fast_disp = min(TTC_FAST, W_win)
    ttc_slow_disp = min(TTC_SLOW, W_win)
    ax.annotate("", xy=(ttc_slow_disp, 1.57), xytext=(ttc_fast_disp, 1.57),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5))
    inequity = max(0.0, W_win - TTC_FAST) - max(0.0, W_win - TTC_SLOW)
    ax.text((ttc_fast_disp + ttc_slow_disp) / 2, 1.63,
            f"Inequity at $W={W_win}$: {inequity:.1f} yr",
            ha="center", va="bottom", fontsize=8.5, color="#555")

    # Legend patches
    comp_patch = mpatches.Patch(color=C_COMP, label="Compliant (certificate valid)")
    ncom_patch = mpatches.Patch(color=C_NCOM, label="Non-compliant (certificate expired)")
    audit_line = plt.Line2D([0], [0], linestyle=":", color="#bdc3c7",
                             linewidth=1.2, label="Annual audit (LL 144)")
    ax.legend(handles=[comp_patch, ncom_patch, audit_line],
              loc="upper left", fontsize=8, frameon=True)

    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["Non-White (minority)", "White (majority)"], fontsize=9)
    ax.set_xlabel("Years since deployment", labelpad=4)
    ax.set_xlim(0, W_win + 0.3)
    ax.set_ylim(-0.35, 1.95)
    ax.grid(True, axis="x", color=C_GRID, linewidth=0.5, zorder=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    plt.tight_layout()
    path = OUT / "fig3_detection_delay.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    make_fig1()
    make_fig2()
    make_fig3()
    print("Done.")

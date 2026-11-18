"""
T1 Extended Gate: Multi-Task × Extended-Year Analysis
======================================================

Tier 1 — Alternate ACS tasks on cached 2014-2018 data (no downloads):
  ACSEmployment        : predict employment status
  ACSPublicCoverage    : predict public health insurance coverage
  ACSIncomePovertyRatio: predict income < 2.5× poverty line

Tier 2 — Extended year range for ACSIncome (downloads 2019, 2021, 2022):
  ACSIncome 2014-2022  : 8 time points instead of 5

For each task: train on earliest year, evaluate fixed model on all subsequent
years, compute per-group degradation rates r_g and hazard ratio HR = r_min/r_maj.

Usage:
  python extended_gate.py               # all tasks, all years
  python extended_gate.py --tier 1      # alternate tasks only (no new downloads)
  python extended_gate.py --tier 2      # extended income years only
"""

import os
import sys
import json
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from core.metrics import demographic_parity_gap, equalized_odds_gap

CACHE_DIR = ROOT / "acs_income" / "_cache"
OUT_DIR   = Path(__file__).parent / "results" / "extended"

STATES    = ["CA", "TX", "NY"]
SUBSAMPLE = 50_000
SEED      = 42
EPOCHS    = 30
BATCH_SIZE = 256
LR        = 0.01
TAU       = 0.05

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Task registry ─────────────────────────────────────────────────────────────
def _build_task_config():
    from folktables import (ACSIncome, ACSEmployment,
                             ACSPublicCoverage, ACSIncomePovertyRatio,
                             BasicProblem, adult_filter)

    # ACSIncome without RELP — RELP was restructured in 2019 PUMS.
    # Dropping it lets the same model evaluate consistently on 2014-2022.
    ACSIncomeCompat = BasicProblem(
        features=['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'WKHP', 'SEX', 'RAC1P'],
        target='PINCP',
        target_transform=lambda x: x > 50000,
        group='RAC1P',
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    return {
        # Tier 2: income with extended years (compat task, no RELP)
        "income_compat": {
            "cls":   ACSIncomeCompat,
            "label": "Income >$50k (no RELP)",
            "years": [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022],
            "tier":  2,
        },
        # Tier 1: alternate tasks on cached 2014-2018
        "employment": {
            "cls":   ACSEmployment,
            "label": "Employment Status",
            "years": [2014, 2015, 2016, 2017, 2018],
            "tier":  1,
        },
        # Tier 1+2: no RELP → extend to 2022 for free
        "coverage": {
            "cls":   ACSPublicCoverage,
            "label": "Public Health Coverage",
            "years": [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022],
            "tier":  1,
        },
        # ACSIncomePovertyRatio also contains RELP — keep at 2014-2018 only
        "poverty": {
            "cls":   ACSIncomePovertyRatio,
            "label": "Income <250% Poverty",
            "years": [2014, 2015, 2016, 2017, 2018],
            "tier":  1,
        },
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_acs_task_year(task_cls, year: int, seed: int = SEED) -> dict | None:
    from folktables import ACSDataSource
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        source = ACSDataSource(
            survey_year=str(year), horizon="1-Year",
            survey="person", root_dir=str(CACHE_DIR),
        )
        raw = source.get_data(states=STATES, download=True)
        feature_names = list(task_cls.features)
        features, label, _ = task_cls.df_to_numpy(raw)

        df = pd.DataFrame(features, columns=feature_names)
        df["label"] = label.astype(int)
        df = df.dropna().reset_index(drop=True)

        rng = np.random.RandomState(seed)
        if len(df) > SUBSAMPLE:
            idx = rng.choice(len(df), SUBSAMPLE, replace=False)
            df = df.iloc[idx].reset_index(drop=True)

        protected_race = (df["RAC1P"].values != 1).astype(int) if "RAC1P" in df.columns else None
        protected_sex  = (df["SEX"].values == 2).astype(int)   if "SEX"   in df.columns else None

        drop_cols = {"SEX", "RAC1P", "label"}
        keep_cols = [c for c in feature_names if c not in drop_cols]
        X = df[keep_cols].values.astype(np.float32)
        y = df["label"].values.astype(int)

        pos = float(y.mean())
        print(f"    {year}: n={len(df):,}  pos={pos:.2%}")
        return {"X": X, "y": y, "race": protected_race, "sex": protected_sex,
                "feature_names": keep_cols, "year": year}
    except Exception as e:
        print(f"    FAILED {year}: {e}")
        return None


# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)


def train_model(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xs), torch.FloatTensor(y)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    model = MLP(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    return model, scaler


def predict_proba(model, scaler, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(scaler.transform(X))).numpy()
    return torch.sigmoid(torch.FloatTensor(logits)).numpy()


# ── Per-year audit ────────────────────────────────────────────────────────────

def audit_year(model, scaler, data: dict) -> dict:
    proba = predict_proba(model, scaler, data["X"])
    row = {"year": data["year"]}
    for attr_name, attr in [("race", data["race"]), ("sex", data["sex"])]:
        if attr is None:
            continue
        g0 = (attr == 0); g1 = (attr == 1)
        ypred = (proba >= 0.5).astype(int)
        dp, _ = demographic_parity_gap(proba, attr)
        eo, _, _ = equalized_odds_gap(proba, data["y"], attr)
        maj = "White" if attr_name == "race" else "Male"
        mn  = "NonWhite" if attr_name == "race" else "Female"
        row[f"{attr_name}_pos_{maj}"] = float(ypred[g0].mean())
        row[f"{attr_name}_pos_{mn}"]  = float(ypred[g1].mean())
        row[f"{attr_name}_dp_gap"]    = dp
        row[f"{attr_name}_eo_gap"]    = eo
    return row


# ── Degradation analysis ──────────────────────────────────────────────────────

def degradation_analysis(audit_df: pd.DataFrame) -> dict:
    t0 = audit_df["year"].min()
    t  = (audit_df["year"] - t0).values
    t0_row = audit_df[audit_df["year"] == t0].iloc[0]

    group_specs = []
    for attr in ["race", "sex"]:
        pairs = [("White", "NonWhite")] if attr == "race" else [("Male", "Female")]
        for maj, mn in pairs:
            mc = f"{attr}_pos_{maj}"; nc = f"{attr}_pos_{mn}"
            if mc in audit_df.columns:
                group_specs += [(attr, maj, mc), (attr, mn, nc)]

    records, slopes = [], {}
    for attr, label, col in group_specs:
        baseline = float(t0_row[col])
        deg = np.abs(audit_df[col].values - baseline)
        for i, (yr, d) in enumerate(zip(audit_df["year"].values, deg)):
            records.append({
                "attr": attr, "group": label,
                "year": int(yr), "t": int(t[i]),
                "pos_rate": float(audit_df[col].values[i]),
                "degradation": float(d),
            })
        slope, intercept, r, p, se = stats.linregress(t, deg)
        n_pts = len(t)
        # 95% CI on slope: slope ± t_{n-2, 0.975} * se
        t_crit = float(stats.t.ppf(0.975, df=max(n_pts - 2, 1)))
        slope_ci_lo = float(slope - t_crit * se)
        slope_ci_hi = float(slope + t_crit * se)
        slopes[(attr, label)] = {
            "slope": float(slope),
            "slope_pos": max(float(slope), 1e-8),
            "slope_se": float(se),
            "slope_ci_lo": slope_ci_lo,
            "slope_ci_hi": slope_ci_hi,
            "r2": float(r ** 2),
            "p": float(p),
            "n": n_pts,
        }

    SLOPE_FLOOR = 1e-4  # below this the slope is "effectively zero" — HR undefined

    alt_hrs, ttcs = {}, {}
    for attr in ["race", "sex"]:
        keys = [(attr, "White"), (attr, "NonWhite")] if attr == "race" \
               else [(attr, "Male"), (attr, "Female")]
        if all(k in slopes for k in keys):
            maj_k, min_k = keys
            r_maj_raw = slopes[maj_k]["slope"]
            r_min_raw = slopes[min_k]["slope"]
            r_maj = slopes[maj_k]["slope_pos"]
            r_min = slopes[min_k]["slope_pos"]

            # Flag one-sided cases where majority is stable
            both_meaningful = (abs(r_maj_raw) >= SLOPE_FLOOR and
                               abs(r_min_raw) >= SLOPE_FLOOR)
            hr_val = float(r_min / r_maj) if both_meaningful else float("nan")
            alt_hrs[attr] = hr_val
            alt_hrs[f"{attr}_one_sided"] = not both_meaningful

            # Delta-method 95% CI on HR = r_min / r_maj
            # Var(HR) ≈ HR^2 * [(se_min/r_min)^2 + (se_maj/r_maj)^2]
            if both_meaningful and hr_val > 0:
                se_min = slopes[min_k]["slope_se"]
                se_maj = slopes[maj_k]["slope_se"]
                hr_se = hr_val * np.sqrt((se_min / r_min) ** 2 +
                                         (se_maj / r_maj) ** 2)
                # Use normal approximation on log scale for better CI coverage
                log_hr = np.log(hr_val)
                log_hr_se = np.sqrt((se_min / r_min) ** 2 + (se_maj / r_maj) ** 2)
                alt_hrs[f"{attr}_ci_lo"] = float(np.exp(log_hr - 1.96 * log_hr_se))
                alt_hrs[f"{attr}_ci_hi"] = float(np.exp(log_hr + 1.96 * log_hr_se))
            else:
                alt_hrs[f"{attr}_ci_lo"] = float("nan")
                alt_hrs[f"{attr}_ci_hi"] = float("nan")

            ttcs[(attr, maj_k[1])] = float(TAU / r_maj) if abs(r_maj_raw) >= SLOPE_FLOOR else float("inf")
            ttcs[(attr, min_k[1])] = float(TAU / r_min) if abs(r_min_raw) >= SLOPE_FLOOR else float("inf")

    return {
        "deg_df": pd.DataFrame(records),
        "slopes": slopes,
        "alt_hrs": alt_hrs,
        "ttcs": ttcs,
    }


# ── Single-task runner ────────────────────────────────────────────────────────

def run_task(task_name: str, cfg: dict) -> dict | None:
    task_cls = cfg["cls"]
    years    = cfg["years"]
    label    = cfg["label"]
    out_dir  = OUT_DIR / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Task: {label}  ({task_name})  |  years: {years[0]}-{years[-1]}")
    print(f"{'='*60}")

    # Load all years
    year_data = {}
    for yr in years:
        d = load_acs_task_year(task_cls, yr)
        if d is not None:
            year_data[yr] = d

    if len(year_data) < 3:
        print(f"  Only {len(year_data)} years loaded — skipping.")
        return None

    base_year = min(year_data)
    base = year_data[base_year]
    X_tr, X_val, y_tr, y_val = train_test_split(
        base["X"], base["y"], test_size=0.2, random_state=SEED
    )
    print(f"\n  Training on {base_year}...")
    model, scaler = train_model(X_tr, y_tr)
    val_proba = predict_proba(model, scaler, X_val)
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"  Val AUC: {val_auc:.4f}")

    # Audit each year
    audit_rows = [audit_year(model, scaler, year_data[yr]) for yr in sorted(year_data)]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(out_dir / "audit_trajectory.csv", index=False)

    # Degradation analysis
    analysis = degradation_analysis(audit_df)
    analysis["deg_df"].to_csv(out_dir / "degradation.csv", index=False)

    # Print HR table
    print(f"\n  Hazard Ratios (slope_minority / slope_majority):")
    for attr, hr in analysis["alt_hrs"].items():
        if attr.endswith("_one_sided"):
            continue
        maj = "White" if attr == "race" else "Male"
        mn  = "NonWhite" if attr == "race" else "Female"
        r_maj = analysis["slopes"].get((attr, maj), {}).get("slope_pos", np.nan)
        r_min = analysis["slopes"].get((attr, mn),  {}).get("slope_pos", np.nan)
        ttc_maj = analysis["ttcs"].get((attr, maj), float("inf"))
        ttc_min = analysis["ttcs"].get((attr, mn),  float("inf"))
        one_sided = analysis["alt_hrs"].get(f"{attr}_one_sided", False)
        ci_lo = analysis["alt_hrs"].get(f"{attr}_ci_lo", float("nan"))
        ci_hi = analysis["alt_hrs"].get(f"{attr}_ci_hi", float("nan"))
        if one_sided or (isinstance(hr, float) and np.isnan(hr)):
            hr_str = "one-sided (majority stable)"
        elif not np.isnan(ci_lo):
            hr_str = f"{hr:.3f}  [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]"
        else:
            hr_str = f"{hr:.3f}"
        ttc_maj_str = "∞" if ttc_maj > 999 else f"{ttc_maj:.1f}yr"
        ttc_min_str = "∞" if ttc_min > 999 else f"{ttc_min:.1f}yr"
        print(f"    {attr:<6}: HR={hr_str:<50}  "
              f"r_{maj}={r_maj:.5f}  r_{mn}={r_min:.5f}  "
              f"TTC_{maj}={ttc_maj_str}  TTC_{mn}={ttc_min_str}")

    result = {
        "task": task_name,
        "label": label,
        "years": sorted(year_data.keys()),
        "val_auc": float(val_auc),
        "alt_hrs": analysis["alt_hrs"],
        "slopes": {f"{a}_{l}": v["slope"]
                   for (a, l), v in analysis["slopes"].items()},
        "ttcs": {f"{a}_{l}": v
                 for (a, l), v in analysis["ttcs"].items()},
        "audit_df": audit_df,
        "analysis": analysis,
    }

    summary = {k: v for k, v in result.items()
               if k not in ("audit_df", "analysis")}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return result


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(results: list[dict]) -> None:
    n = len(results)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
    axes = np.array(axes).flatten()

    colors = {"Female": "#e06c75", "Male": "#61afef",
              "NonWhite": "#e06c75", "White": "#61afef"}
    markers = {"Female": "o", "Male": "s", "NonWhite": "o", "White": "s"}

    for idx, res in enumerate(results):
        ax = axes[idx]
        analysis = res["analysis"]
        deg_df = analysis["deg_df"]
        t0 = deg_df["year"].min()
        t_max = deg_df["year"].max()

        # Plot per-group degradation and fitted lines
        for (attr, group), sdata in analysis["slopes"].items():
            if attr != "sex":   # show sex by default; skip race for clarity
                continue
            sub = deg_df[(deg_df["attr"] == attr) & (deg_df["group"] == group)]
            if sub.empty:
                continue
            t_arr = sub["t"].values
            d_arr = sub["degradation"].values
            ax.scatter(sub["year"].values, d_arr,
                       color=colors.get(group, "gray"),
                       marker=markers.get(group, "o"),
                       s=60, zorder=5, label=group)
            # Fitted line
            t_fit = np.linspace(0, t_arr.max() + 1, 100)
            d_fit = sdata["slope"] * t_fit
            ax.plot(t0 + t_fit, d_fit,
                    color=colors.get(group, "gray"),
                    linestyle="--", linewidth=1.5, alpha=0.8)

        # Threshold line
        ax.axhline(TAU, color="black", linestyle=":", linewidth=1.2, alpha=0.7,
                   label=f"τ = {TAU}")

        hr_sex = res["alt_hrs"].get("sex", None)
        hr_str = f"HR$_{{\\rm sex}}$ = {hr_sex:.2f}" if hr_sex else ""
        ax.set_title(f"{res['label']}\n{hr_str}", fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel(r"Degradation $\Delta_g(t)$")
        ax.legend(fontsize=9)
        ax.set_xlim(t0 - 0.3, t_max + 0.3)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    # Hide any unused panels
    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Group Fairness Certificate Degradation Across ACS Tasks\n"
                 "(Fixed 2014 model, drifting population; dashed = fitted linear rate)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig_path = OUT_DIR / "figure_degradation_all_tasks.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved → {fig_path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def make_summary_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for attr in ["race", "sex"]:
            hr = r["alt_hrs"].get(attr)
            if hr is not None and (isinstance(hr, float) and np.isnan(hr)):
                hr = None  # treat NaN as missing for display
            if hr is None:
                continue
            maj = "White" if attr == "race" else "Male"
            mn  = "NonWhite" if attr == "race" else "Female"
            r_maj = r["slopes"].get(f"{attr}_{maj}", np.nan)
            r_min = r["slopes"].get(f"{attr}_{mn}",  np.nan)
            ttc_maj = r["ttcs"].get(f"{attr}_{maj}", np.nan)
            ttc_min = r["ttcs"].get(f"{attr}_{mn}",  np.nan)
            ci_lo = r["alt_hrs"].get(f"{attr}_ci_lo", float("nan"))
            ci_hi = r["alt_hrs"].get(f"{attr}_ci_hi", float("nan"))
            rows.append({
                "task":       r["task"],
                "label":      r["label"],
                "attr":       attr,
                "years":      f"{r['years'][0]}-{r['years'][-1]}",
                "n_snapshots":len(r["years"]),
                "val_auc":    round(r["val_auc"], 4),
                f"r_{maj}":   round(r_maj, 6),
                f"r_{mn}":    round(r_min, 6),
                "HR":         round(hr, 3) if hr is not None else "one-sided",
                "HR_ci_lo":   round(ci_lo, 3) if not np.isnan(ci_lo) else "—",
                "HR_ci_hi":   round(ci_hi, 3) if not np.isnan(ci_hi) else "—",
                f"TTC_{maj}": round(ttc_maj, 1) if ttc_maj < 1e6 else "∞",
                f"TTC_{mn}":  round(ttc_min, 1) if ttc_min < 1e6 else "∞",
                "inequity_yrs": round(abs(ttc_maj - ttc_min), 1)
                                if ttc_maj < 1e6 and ttc_min < 1e6 else "∞",
            })
    df = pd.DataFrame(rows)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=int, choices=[1, 2], default=None,
                   help="1=alternate tasks only, 2=extended income only, "
                        "omit for both")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TASK_CONFIG = _build_task_config()

    tasks_to_run = {
        k: v for k, v in TASK_CONFIG.items()
        if args.tier is None or v["tier"] == args.tier
    }

    print(f"\nT1 Extended Gate — {len(tasks_to_run)} task(s)")
    for k, v in tasks_to_run.items():
        print(f"  {k}: {v['label']}  years={v['years']}")

    results = []
    for task_name, cfg in tasks_to_run.items():
        res = run_task(task_name, cfg)
        if res is not None:
            results.append(res)

    if not results:
        print("No results to summarize.")
        return

    # Summary table
    summary_df = make_summary_table(results)
    summary_df.to_csv(OUT_DIR / "summary_hr.csv", index=False)

    print("\n" + "=" * 72)
    print("SUMMARY: Hazard Ratios Across All Tasks")
    print("=" * 72)
    for attr in ["sex", "race"]:
        sub = summary_df[summary_df["attr"] == attr]
        if sub.empty:
            continue
        print(f"\n  Protected attribute: {attr}")
        print(f"  {'Task':<24} {'Years':<12} {'AUC':>6} "
              f"{'HR':>6} {'Inequity (yr)':>14}")
        print(f"  {'-'*64}")
        for _, row in sub.iterrows():
            print(f"  {row['label']:<24} {row['years']:<12} "
                  f"{row['val_auc']:>6.4f} {row['HR']:>6.3f} "
                  f"{str(row['inequity_yrs']):>14}")

    print(f"\nSummary CSV → {OUT_DIR / 'summary_hr.csv'}")

    # Figure
    make_figure(results)


if __name__ == "__main__":
    main()

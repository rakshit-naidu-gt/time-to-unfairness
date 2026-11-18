"""
Model-Class Robustness Check for the Income Task (sex, race)
==============================================================

Reviewer ask (FAccT revision of Time-to-Unfairness): does the headline
HR_sex = 1.91 on the ACS income task hold qualitatively under (a) a linear
classifier and (b) a gradient-boosted tree ensemble, not just the
two-hidden-layer MLP reported in the paper?

This script reuses the exact protocol in extended_gate.py (same cached
ACS PUMS data, same income_compat task definition, same years 2014-2022
minus 2020, same train-on-2014-hold-fixed design, same tau=0.05,
same OLS-with-intercept degradation-rate estimator) and swaps in
LogisticRegressionWrapper / GradientBoostingWrapper from core.models
in place of the MLP.

Adds a percentile-bootstrap CI on HR (1000 resamples of annual snapshots,
seed 42, paired across groups) matching the methodology described in the
paper's Remark on confidence intervals.

Usage (must run under the 'new_env' conda environment which has
folktables + torch + sklearn together):
  conda run -n new_env python model_class_robustness.py
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # for core.models

from core.models import LogisticRegressionWrapper, GradientBoostingWrapper
import extended_gate as eg  # reuse task registry, data loader, degradation analysis

TAU = eg.TAU
SEED = eg.SEED
OUT_DIR = HERE / "results" / "model_class_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
BOOT_SEED = 42


# ── Model backends ──────────────────────────────────────────────────────────

def make_backend(name: str):
    if name == "LogisticRegression":
        return LogisticRegressionWrapper(max_iter=1000)
    if name == "GradientBoosting":
        return GradientBoostingWrapper(n_estimators=100, max_depth=3,
                                        learning_rate=0.1, random_state=SEED)
    raise ValueError(name)


def audit_year_sklearn(wrapper, data: dict) -> dict:
    proba = wrapper.predict_proba(data["X"])
    ypred = (proba >= 0.5).astype(int)
    row = {"year": data["year"]}
    for attr_name, attr in [("race", data["race"]), ("sex", data["sex"])]:
        if attr is None:
            continue
        g0 = (attr == 0)
        g1 = (attr == 1)
        maj = "White" if attr_name == "race" else "Male"
        mn = "NonWhite" if attr_name == "race" else "Female"
        row[f"{attr_name}_pos_{maj}"] = float(ypred[g0].mean())
        row[f"{attr_name}_pos_{mn}"] = float(ypred[g1].mean())
    return row


# ── Bootstrap CI on HR (percentile bootstrap over annual snapshots) ─────────

def bootstrap_hr_ci(t: np.ndarray, deg_maj: np.ndarray, deg_min: np.ndarray,
                     n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    n = len(t)
    hrs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        t_b = t[idx]
        maj_b = deg_maj[idx]
        min_b = deg_min[idx]
        if len(np.unique(t_b)) < 2:
            continue
        slope_maj, *_ = stats.linregress(t_b, maj_b)
        slope_min, *_ = stats.linregress(t_b, min_b)
        if abs(slope_maj) < 1e-8:
            continue
        hr = slope_min / slope_maj
        if hr > 0:
            hrs.append(hr)
    if len(hrs) < n_boot * 0.5:
        return float("nan"), float("nan"), len(hrs)
    lo, hi = np.percentile(hrs, [2.5, 97.5])
    return float(lo), float(hi), len(hrs)


# ── Main ──────────────────────────────────────────────────────────────────

def run_for_backend(backend_name: str, task_name: str, cfg: dict) -> dict:
    print(f"\n{'='*70}")
    print(f"Model class: {backend_name}  |  Task: {cfg['label']}")
    print(f"{'='*70}")

    years = cfg["years"]
    year_data = {}
    for yr in years:
        d = eg.load_acs_task_year(cfg["cls"], yr)
        if d is not None:
            year_data[yr] = d

    base_year = min(year_data)
    base = year_data[base_year]
    X_tr, X_val, y_tr, y_val = train_test_split(
        base["X"], base["y"], test_size=0.2, random_state=SEED
    )

    wrapper = make_backend(backend_name)
    wrapper.fit(X_tr, y_tr)
    val_proba = wrapper.predict_proba(X_val)
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"  Val AUC: {val_auc:.4f}")

    audit_rows = [audit_year_sklearn(wrapper, year_data[yr]) for yr in sorted(year_data)]
    audit_df = pd.DataFrame(audit_rows)

    analysis = eg.degradation_analysis(audit_df)

    out = {"backend": backend_name, "task": task_name, "val_auc": float(val_auc),
           "years": sorted(year_data.keys())}

    t0 = audit_df["year"].min()
    t = (audit_df["year"] - t0).values

    for attr, maj, mn in [("race", "White", "NonWhite"), ("sex", "Male", "Female")]:
        maj_col = f"{attr}_pos_{maj}"
        mn_col = f"{attr}_pos_{mn}"
        if maj_col not in audit_df.columns:
            continue
        baseline_maj = float(audit_df[audit_df["year"] == t0][maj_col].iloc[0])
        baseline_mn = float(audit_df[audit_df["year"] == t0][mn_col].iloc[0])
        deg_maj = np.abs(audit_df[maj_col].values - baseline_maj)
        deg_mn = np.abs(audit_df[mn_col].values - baseline_mn)

        slope_maj, _, r_maj, p_maj, se_maj = stats.linregress(t, deg_maj)
        slope_mn, _, r_mn, p_mn, se_mn = stats.linregress(t, deg_mn)

        hr = slope_mn / slope_maj if abs(slope_maj) > 1e-8 else float("nan")
        ci_lo, ci_hi, n_valid = bootstrap_hr_ci(t, deg_maj, deg_mn)

        ttc_maj = TAU / slope_maj if slope_maj > 1e-8 else float("inf")
        ttc_mn = TAU / slope_mn if slope_mn > 1e-8 else float("inf")

        print(f"  [{attr}] r_{maj}={slope_maj:.6f} (R2={r_maj**2:.2f})  "
              f"r_{mn}={slope_mn:.6f} (R2={r_mn**2:.2f})  "
              f"HR={hr:.3f}  CI=[{ci_lo:.3f}, {ci_hi:.3f}]  (n_boot_valid={n_valid})")

        out[attr] = {
            "r_maj": float(slope_maj), "r_min": float(slope_mn),
            "r2_maj": float(r_maj ** 2), "r2_min": float(r_mn ** 2),
            "p_maj": float(p_maj), "p_min": float(p_mn),
            "HR": float(hr), "HR_ci_lo": ci_lo, "HR_ci_hi": ci_hi,
            "n_boot_valid": n_valid,
            "TTC_maj": float(ttc_maj) if np.isfinite(ttc_maj) else None,
            "TTC_min": float(ttc_mn) if np.isfinite(ttc_mn) else None,
        }

    return out


def main():
    task_cfg = eg._build_task_config()["income_compat"]
    results = []
    for backend_name in ["LogisticRegression", "GradientBoosting"]:
        res = run_for_backend(backend_name, "income_compat", task_cfg)
        results.append(res)
        with open(OUT_DIR / f"{backend_name}.json", "w") as f:
            json.dump(res, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY: Income-task sex HR across model classes")
    print(f"{'='*70}")
    print(f"  {'Model':<20} {'HR_sex':>8} {'CI':>20} {'HR_race':>8} {'CI':>20}")
    # MLP reference row from the paper / extended_gate summary_hr.csv
    print(f"  {'MLP (paper)':<20} {1.905:>8.3f} {'[1.06, 2.71]':>20} {0.185:>8.3f} {'[0.03, 1.11]':>20}")
    for r in results:
        sex = r.get("sex", {})
        race = r.get("race", {})
        ci_sex = f"[{sex.get('HR_ci_lo', float('nan')):.2f}, {sex.get('HR_ci_hi', float('nan')):.2f}]"
        ci_race = f"[{race.get('HR_ci_lo', float('nan')):.2f}, {race.get('HR_ci_hi', float('nan')):.2f}]"
        print(f"  {r['backend']:<20} {sex.get('HR', float('nan')):>8.3f} {ci_sex:>20} "
              f"{race.get('HR', float('nan')):>8.3f} {ci_race:>20}")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    main()

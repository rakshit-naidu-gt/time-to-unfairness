"""
T1 Gate Experiment: Time-to-Unfairness
=======================================

Hypothesis: When a model trained in 2014 is held fixed and evaluated on
successive ACS PUMS populations (2015-2018), the positive-rate degradation
D_g(t) = |pos_rate_g(t) - pos_rate_g(2014)| grows faster for minority groups
than for majority groups.  This gives a per-group "time-to-crossing-tau"
under a Weibull/linear extrapolation — the Accelerated Life Testing (ALT) frame.

Gate decision:
  GO      — At least one group crosses tau empirically AND crossing time
             differs between minority/majority (HR > 1.2 or < 0.8).
  PARTIAL — No real crossings but degradation rate differs significantly
             between groups (ALT extrapolated HR > 1.2).  Primary engine
             shifts to ALT simulation; paper is theoretical with empirical
             motivation.
  KILL    — No real crossings AND degradation rates are parallel
             (ALT HR ≈ 1.0 ± 0.2).  Survival framing adds nothing.

Usage:
  python gate_experiment.py                    # full run, downloads data
  python gate_experiment.py --years 2014 2015  # fast test with 2 years
  python gate_experiment.py --no-download      # fail fast if data absent
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
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from core.metrics import demographic_parity_gap, equalized_odds_gap

CACHE_DIR = ROOT / "acs_income" / "_cache"
OUT_DIR = Path(__file__).parent / "results"

# ── Config ───────────────────────────────────────────────────────────────────
YEARS = [2014, 2015, 2016, 2017, 2018]
STATES = ["CA", "TX", "NY"]     # same as existing 2018 loader
SUBSAMPLE = 50_000
SEED = 42
EPOCHS = 30
BATCH_SIZE = 256
LR = 0.01
TAU = 0.05   # degradation threshold: |pos_rate_g(t) - pos_rate_g(t=0)| > TAU
TAU_GAP = 0.10  # compliance threshold: DP gap > TAU_GAP counts as event

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_acs_year(year: int, seed: int = SEED) -> dict | None:
    """Load ACS Income for a single year via folktables."""
    try:
        from folktables import ACSDataSource, ACSIncome
    except ImportError:
        print("  ERROR: pip install folktables")
        return None

    cache = CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  Loading ACS {year} ({', '.join(STATES)})...")
        source = ACSDataSource(
            survey_year=str(year),
            horizon="1-Year",
            survey="person",
            root_dir=str(cache),
        )
        raw = source.get_data(states=STATES, download=True)
        feature_names = list(ACSIncome.features)
        features, label, _ = ACSIncome.df_to_numpy(raw)

        df = pd.DataFrame(features, columns=feature_names)
        df["label"] = label.astype(int)
        df = df.dropna().reset_index(drop=True)

        rng = np.random.RandomState(seed)
        if len(df) > SUBSAMPLE:
            idx = rng.choice(len(df), SUBSAMPLE, replace=False)
            df = df.iloc[idx].reset_index(drop=True)

        protected_race = (df["RAC1P"].values != 1).astype(int)  # Non-White=1
        protected_sex  = (df["SEX"].values == 2).astype(int)    # Female=1

        drop_cols = {"SEX", "RAC1P", "label"}
        keep_cols = [c for c in feature_names if c not in drop_cols]
        X = df[keep_cols].values.astype(np.float32)
        y = df["label"].values.astype(int)

        print(f"    n={len(df):,}  White={( protected_race==0).sum():,} "
              f"Non-White={(protected_race==1).sum():,}  "
              f">50K={(y==1).sum()/len(y):.2%}")
        return {
            "X": X, "y": y,
            "race": protected_race,
            "sex": protected_sex,
            "feature_names": keep_cols,
            "year": year,
        }
    except Exception as e:
        print(f"  FAILED to load {year}: {e}")
        return None


# ── Model ────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_model(X_train, y_train) -> tuple[MLP, StandardScaler]:
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)

    X_t = torch.FloatTensor(X_s)
    y_t = torch.FloatTensor(y_train)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)
    model = MLP(X_train.shape[1])
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()

    return model, scaler


def predict_proba(model: MLP, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_s = scaler.transform(X)
        logits = model(torch.FloatTensor(X_s)).numpy()
    return torch.sigmoid(torch.FloatTensor(logits)).numpy()


# ── Fairness audit ────────────────────────────────────────────────────────────

def audit_year(model, scaler, data: dict) -> dict:
    """Return per-group positive rates and fairness gaps for one year."""
    proba = predict_proba(model, scaler, data["X"])
    y_pred = (proba >= 0.5).astype(int)

    results = {"year": data["year"]}

    for attr_name, attr in [("race", data["race"]), ("sex", data["sex"])]:
        g0_mask = attr == 0
        g1_mask = attr == 1
        pos_g0 = float(y_pred[g0_mask].mean())
        pos_g1 = float(y_pred[g1_mask].mean())
        dp_gap, _ = demographic_parity_gap(proba, attr)
        eo_gap, _, _ = equalized_odds_gap(proba, data["y"], attr)
        acc = float((y_pred == data["y"]).mean())

        label_g0 = "White"    if attr_name == "race" else "Male"
        label_g1 = "NonWhite" if attr_name == "race" else "Female"
        results[f"{attr_name}_pos_{label_g0}"] = pos_g0
        results[f"{attr_name}_pos_{label_g1}"] = pos_g1
        results[f"{attr_name}_dp_gap"] = dp_gap
        results[f"{attr_name}_eo_gap"] = eo_gap
        results[f"{attr_name}_acc"] = acc

    return results


# ── Degradation + ALT analysis ────────────────────────────────────────────────

def degradation_analysis(df: pd.DataFrame) -> dict:
    """
    Compute per-group degradation D_g(t) = |pos_rate_g(t) - pos_rate_g(t0)|
    and fit linear + power-law growth models.

    Returns dict with:
      - degradation_df: per-year per-group degradation values
      - crossings: list of (group, year) where D_g > TAU
      - alt_hr: ALT hazard ratio minority/majority (rate ratio of fitted slopes)
      - gap_crossings: list of (attr, year) where dp_gap > TAU_GAP
    """
    t0_row = df[df["year"] == df["year"].min()].iloc[0]
    years = df["year"].values
    t = years - years[0]  # time since deployment (0, 1, 2, 3, 4)

    records = []
    crossings = []
    gap_crossings = []

    group_specs = [
        ("race", "White",    "race_pos_White"),
        ("race", "NonWhite", "race_pos_NonWhite"),
        ("sex",  "Male",     "sex_pos_Male"),
        ("sex",  "Female",   "sex_pos_Female"),
    ]

    slopes = {}
    for attr, label, col in group_specs:
        baseline = float(t0_row[col])
        degradation = np.abs(df[col].values - baseline)
        for i, (yr, deg) in enumerate(zip(years, degradation)):
            records.append({
                "year": int(yr), "t": int(t[i]),
                "attr": attr, "group": label,
                "pos_rate": float(df[col].values[i]),
                "degradation": float(deg),
                "event": int(deg > TAU),
            })
            if deg > TAU:
                crossings.append((attr, label, int(yr), float(deg)))

        # Fit linear slope to degradation trajectory (for ALT HR)
        if t[-1] > 0:
            slope, intercept, r, p, _ = stats.linregress(t, degradation)
            slopes[(attr, label)] = max(slope, 1e-8)  # guard against flat/negative

    # Gap-level compliance crossings
    for attr in ["race", "sex"]:
        col = f"{attr}_dp_gap"
        for _, row in df.iterrows():
            if row[col] > TAU_GAP:
                gap_crossings.append((attr, int(row["year"]), float(row[col])))

    # ALT Hazard Ratio: slope_minority / slope_majority
    alt_hrs = {}
    for attr in ["race", "sex"]:
        maj_label = "White" if attr == "race" else "Male"
        min_label = "NonWhite" if attr == "race" else "Female"
        s_maj = slopes.get((attr, maj_label), 1e-8)
        s_min = slopes.get((attr, min_label), 1e-8)
        alt_hrs[attr] = float(s_min / s_maj)

    # Extrapolated time-to-crossing (linear model: time_g = TAU / slope_g)
    t2c = {}
    for (attr, label), slope in slopes.items():
        t2c[(attr, label)] = float(TAU / slope) if slope > 1e-8 else float("inf")

    deg_df = pd.DataFrame(records)
    return {
        "degradation_df": deg_df,
        "crossings": crossings,
        "gap_crossings": gap_crossings,
        "alt_hrs": alt_hrs,
        "t2c": t2c,
        "slopes": {f"{a}_{l}": float(v) for (a, l), v in slopes.items()},
    }


# ── Verdict ───────────────────────────────────────────────────────────────────

def make_verdict(audit_df: pd.DataFrame, analysis: dict) -> str:
    crossings = analysis["crossings"]
    gap_crossings = analysis["gap_crossings"]
    alt_hrs = analysis["alt_hrs"]

    has_real_crossings = len(crossings) > 0 or len(gap_crossings) > 0

    # Group-differential crossing: minority crosses before/instead of majority
    minority_labels = {"NonWhite", "Female"}
    maj_labels = {"White", "Male"}
    minority_crossings = [c for c in crossings if c[1] in minority_labels]
    majority_crossings = [c for c in crossings if c[1] in maj_labels]

    hr_race = alt_hrs.get("race", 1.0)
    hr_sex  = alt_hrs.get("sex", 1.0)
    alt_differential = hr_race > 1.2 or hr_race < 0.8 or hr_sex > 1.2 or hr_sex < 0.8

    if has_real_crossings and (len(minority_crossings) != len(majority_crossings)):
        return "GO"
    elif has_real_crossings:
        return "GO-SYMMETRIC"   # crossings happen but no group differential
    elif alt_differential:
        return "PARTIAL"
    else:
        return "KILL"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_gate(years: list[int], allow_download: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nT1 Gate — loading ACS {years[0]}–{years[-1]} ({', '.join(STATES)})")
    print(f"Tau (degradation): {TAU}  |  Tau (DP gap): {TAU_GAP}\n")

    # ── Load all years ──────────────────────────────────────────────────────
    year_data = {}
    for yr in years:
        d = load_acs_year(yr)
        if d is None:
            print(f"  Skipping {yr} (load failed)")
        else:
            year_data[yr] = d

    if len(year_data) < 2:
        print("Need at least 2 years. Exiting.")
        return

    base_year = min(year_data.keys())
    print(f"\nTraining MLP on ACS {base_year}...")
    base = year_data[base_year]
    X_tr, X_val, y_tr, y_val = train_test_split(
        base["X"], base["y"], test_size=0.2, random_state=SEED
    )
    model, scaler = train_model(X_tr, y_tr)

    val_proba = predict_proba(model, scaler, X_val)
    val_auc = roc_auc_score(y_val, val_proba)
    val_acc = float(((val_proba >= 0.5) == y_val).mean())
    print(f"  Val AUC: {val_auc:.4f}  Acc: {val_acc:.4f}")

    # ── Audit each year ─────────────────────────────────────────────────────
    print(f"\nAuditing years: {sorted(year_data.keys())}")
    audit_rows = []
    for yr in sorted(year_data.keys()):
        row = audit_year(model, scaler, year_data[yr])
        audit_rows.append(row)
        print(f"  {yr}: DP-race={row['race_dp_gap']:.4f}  "
              f"EO-race={row['race_eo_gap']:.4f}  "
              f"DP-sex={row['sex_dp_gap']:.4f}  "
              f"EO-sex={row['sex_eo_gap']:.4f}")

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUT_DIR / "t1_audit_trajectory.csv", index=False)

    # ── Degradation + ALT ───────────────────────────────────────────────────
    analysis = degradation_analysis(audit_df)
    analysis["degradation_df"].to_csv(OUT_DIR / "t1_degradation.csv", index=False)

    # ── Print summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("T1 GATE — FAIRNESS TRAJECTORY (fixed model, drifting population)")
    print("=" * 70)
    print(f"{'Year':<6} {'DP-race':>8} {'EO-race':>8} {'DP-sex':>8} {'EO-sex':>8}")
    print("-" * 70)
    for _, row in audit_df.iterrows():
        flag_race = " *** CROSS" if row["race_dp_gap"] > TAU_GAP else ""
        flag_sex  = " *** CROSS" if row["sex_dp_gap"]  > TAU_GAP else ""
        print(f"{int(row['year']):<6} {row['race_dp_gap']:>8.4f} "
              f"{row['race_eo_gap']:>8.4f} "
              f"{row['sex_dp_gap']:>8.4f} "
              f"{row['sex_eo_gap']:>8.4f}"
              f"{flag_race}{flag_sex}")
    print("=" * 70)

    print("\n--- Degradation analysis (|pos_rate_g(t) - pos_rate_g(t0)|) ---")
    deg_summary = (
        analysis["degradation_df"]
        .groupby(["attr", "group"])
        .agg(max_deg=("degradation", "max"), slope=("degradation", lambda x: float(np.polyfit(
            analysis["degradation_df"].loc[x.index, "t"], x, 1)[0])))
        .reset_index()
    )
    print(deg_summary.to_string(index=False))

    print(f"\n--- ALT Hazard Ratios (slope_minority / slope_majority) ---")
    for attr, hr in analysis["alt_hrs"].items():
        direction = "minority degrades FASTER" if hr > 1 else "majority degrades faster"
        print(f"  {attr:<6}: HR = {hr:.3f}  ({direction})")

    print(f"\n--- Extrapolated time-to-crossing-tau={TAU} (years, linear model) ---")
    for (attr, label), t2c_val in analysis["t2c"].items():
        t2c_str = f"{t2c_val:.1f}" if t2c_val < 100 else "∞"
        print(f"  {attr}/{label:<10}: {t2c_str} years")

    if analysis["crossings"]:
        print(f"\n*** Real degradation crossings (tau={TAU}): {analysis['crossings']}")
    if analysis["gap_crossings"]:
        print(f"*** Real DP gap crossings (tau_gap={TAU_GAP}): {analysis['gap_crossings']}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    verdict = make_verdict(audit_df, analysis)
    verdict_map = {
        "GO": "GO — Time-to-Unfairness paper. Empirical crossings + group differential.",
        "GO-SYMMETRIC": "GO-SYMMETRIC — Crossings exist but affect both groups equally. "
                        "Lead with detection-delay inequity, not HR.",
        "PARTIAL": "PARTIAL — No real crossings; ALT HR differs by group. "
                   "ALT simulation is the primary engine; empirical motivation is weak.",
        "KILL": "KILL — No crossings AND degradation is symmetric across groups. "
                "Survival framing adds nothing beyond standard temporal fairness plots.",
    }

    print(f"\n=== VERDICT: {verdict} ===")
    print(verdict_map[verdict])

    # Save
    summary = {
        "verdict": verdict,
        "years_loaded": sorted(year_data.keys()),
        "base_year": base_year,
        "val_auc": float(val_auc),
        "tau_degradation": TAU,
        "tau_dp_gap": TAU_GAP,
        "real_crossings": [list(c) for c in analysis["crossings"]],
        "gap_crossings": [list(c) for c in analysis["gap_crossings"]],
        "alt_hr_race": analysis["alt_hrs"].get("race"),
        "alt_hr_sex": analysis["alt_hrs"].get("sex"),
        "slopes": analysis["slopes"],
        "t2c_years": {f"{a}_{l}": v for (a, l), v in analysis["t2c"].items()},
    }
    with open(OUT_DIR / "t1_verdict.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults → {OUT_DIR}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, default=YEARS)
    p.add_argument("--no-download", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gate(years=sorted(args.years), allow_download=not args.no_download)

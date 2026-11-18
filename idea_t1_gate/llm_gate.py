"""
LLM Path-A Gate: Time-to-Unfairness with a frozen Llama-3.2 classifier
=======================================================================
Uses Ollama (llama3.2:latest) as a frozen zero-shot binary classifier
on ACS Income text descriptions across 2014-2022.

Measures per-group positive prediction rate drift and estimates HR
using the same ALT framework as extended_gate.py.

Runtime: ~25-40 min on Apple Silicon (batches of 5, 150 records/group/year).
"""

import sys, os
sys.path.insert(0, "/tmp/fk3")       # folktables install path

import json, time, re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from folktables import ACSDataSource, BasicProblem, adult_filter

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL        = "llama3.2:latest"
STATES       = ["CA", "TX", "NY"]
YEARS        = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022]
N_PER_GROUP  = 150          # records sampled per demographic group per year
BATCH_SIZE   = 5            # records per Ollama call
TAU          = 0.05
SEED         = 42
CACHE_DIR    = Path("/tmp/acs_cache")
OUT_DIR      = Path("results/llm")
CHECKPOINT   = OUT_DIR / "checkpoint.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

# ── ACS task definition (same as income_compat in extended_gate.py) ───────────
ACSIncomeCompat = BasicProblem(
    features=["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "WKHP", "SEX", "RAC1P"],
    target="PINCP",
    target_transform=lambda x: x > 50000,
    group="RAC1P",
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

# ── Feature → text ────────────────────────────────────────────────────────────
EDU_MAP = {
    1: "no schooling", 2: "nursery school", 3: "grades 1-4",
    4: "grades 5-6",   5: "grades 7-8",    6: "grade 9",
    7: "grade 10",     8: "grade 11",      9: "grade 12 (no diploma)",
    10: "high school diploma", 11: "some college", 12: "some college",
    13: "some college", 14: "associate degree", 15: "associate degree",
    16: "bachelor degree", 17: "bachelor degree", 18: "master degree",
    19: "professional degree", 20: "professional degree",
    21: "doctorate degree", 22: "doctorate degree",
    23: "doctorate degree", 24: "doctorate degree",
}

COW_MAP = {
    1: "private sector employee", 2: "non-profit employee",
    3: "local government employee", 4: "state government employee",
    5: "federal government employee", 6: "self-employed",
    7: "self-employed (incorporated)", 8: "unpaid family worker",
    9: "unemployed",
}

MAR_MAP = {
    1: "married", 2: "widowed", 3: "divorced", 4: "separated", 5: "single",
}

def occp_to_category(code):
    """Map 4-digit ACS OCCP code to a broad occupation label."""
    c = int(code) if not np.isnan(code) else 0
    if   c <   10: return "management"
    elif c <  200: return "management or business"
    elif c <  400: return "computer, math, or engineering"
    elif c <  600: return "science or healthcare"
    elif c <  800: return "education or social services"
    elif c < 1000: return "arts or media"
    elif c < 1300: return "sales or office work"
    elif c < 1600: return "services or food"
    elif c < 2000: return "construction or maintenance"
    elif c < 2600: return "production or transportation"
    else:          return "other"

def row_to_text(row, sex_label, race_label):
    """Convert a single ACS row to a short natural-language description."""
    age  = int(row["AGEP"]) if not np.isnan(row["AGEP"]) else "unknown age"
    edu  = EDU_MAP.get(int(row["SCHL"]) if not np.isnan(row["SCHL"]) else 0, "unknown education")
    hrs  = int(row["WKHP"]) if not np.isnan(row["WKHP"]) else 0
    cow  = COW_MAP.get(int(row["COW"])  if not np.isnan(row["COW"])  else 0, "employed")
    mar  = MAR_MAP.get(int(row["MAR"])  if not np.isnan(row["MAR"])  else 5, "single")
    occp = occp_to_category(row["OCCP"])
    return (f"{sex_label}, age {age}, {edu}, {mar}, "
            f"{cow} in {occp}, {hrs} hrs/week")


# ── Ollama batch classifier ───────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a classification assistant. For each person described, "
    "decide whether their annual income exceeds $50,000. "
    "Reply with ONLY a numbered list of yes or no answers, one per line."
)

def classify_batch(texts: list[str], retries: int = 3) -> list[int]:
    """
    Send a batch of text descriptions to Ollama and return binary predictions.
    Returns list of 1 (>50k) or 0 (<=50k), same length as texts.
    -1 indicates a parsing failure for that record.
    """
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"People:\n{numbered}\n\n"
        f"Answers (one per line, just yes or no):\n"
    )
    for attempt in range(retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": len(texts) * 8},
                },
                timeout=60,
            )
            raw = resp.json()["response"].strip()
            # Parse: lines of the form "1. yes" or "1. no" or just "yes"/"no"
            results = []
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            for line in lines:
                # Strip leading number and punctuation
                cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).lower()
                if "yes" in cleaned:
                    results.append(1)
                elif "no" in cleaned:
                    results.append(0)
            # If we got the right number, return
            if len(results) == len(texts):
                return results
            # Wrong count — pad with -1 for unmatched positions
            if len(results) > len(texts):
                return results[:len(texts)]
        except Exception as e:
            if attempt == retries - 1:
                print(f"  Ollama call failed after {retries} retries: {e}")
    # Fallback: return -1 for all
    return [-1] * len(texts)


# ── Load ACS year and sample per group ────────────────────────────────────────
def load_year(year: int) -> pd.DataFrame | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        source = ACSDataSource(
            survey_year=str(year), horizon="1-Year",
            survey="person", root_dir=str(CACHE_DIR),
        )
        raw = source.get_data(states=STATES, download=True)
        features, label, _ = ACSIncomeCompat.df_to_numpy(raw)
        feat_names = list(ACSIncomeCompat.features)

        df = pd.DataFrame(features, columns=feat_names)
        df["label"] = label.astype(int)
        df = df.dropna().reset_index(drop=True)

        # Demographic group columns
        df["is_female"]   = (df["SEX"]   == 2).astype(int)
        df["is_nonwhite"] = (df["RAC1P"] != 1).astype(int)
        return df
    except Exception as e:
        print(f"  Load failed for {year}: {e}")
        return None


def sample_group(df: pd.DataFrame, sex_val: int, race_val: int,
                 n: int, seed: int) -> pd.DataFrame:
    """Sample n records from a demographic group."""
    mask = (df["is_female"] == sex_val) & (df["is_nonwhite"] == race_val)
    sub = df[mask]
    if len(sub) < n:
        return sub
    return sub.sample(n=n, random_state=seed)


# ── Main experiment ───────────────────────────────────────────────────────────
def run():
    # Load checkpoint if resuming
    checkpoint = {}
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint: {list(checkpoint.keys())}")

    groups = [
        ("Male",     "White",    0, 0),
        ("Female",   "White",    1, 0),
        ("Male",     "NonWhite", 0, 1),
        ("Female",   "NonWhite", 1, 1),
    ]

    # Structure: results[year][group_label] = {"pos_rate": float, "n": int}
    results: dict[int, dict] = {}

    for year in YEARS:
        yr_key = str(year)
        if yr_key in checkpoint:
            print(f"\n{year}: loaded from checkpoint")
            results[year] = checkpoint[yr_key]
            continue

        print(f"\n{'='*60}")
        print(f"Year {year}")
        print(f"{'='*60}")

        df = load_year(year)
        if df is None:
            continue

        yr_result = {}
        for sex_label, race_label, sex_val, race_val in groups:
            group_key = f"{sex_label}_{race_label}"
            sample = sample_group(df, sex_val, race_val,
                                  N_PER_GROUP, SEED + year)
            if len(sample) == 0:
                print(f"  {group_key}: no data")
                continue

            print(f"  {group_key}: n={len(sample)}", end="", flush=True)

            texts = [
                row_to_text(row, sex_label, race_label)
                for _, row in sample.iterrows()
            ]

            # Batch inference
            preds = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                batch_preds = classify_batch(batch)
                preds.extend(batch_preds)

            # Filter out parsing failures (-1)
            valid = [(p, l) for p, l in zip(preds, sample["label"].values)
                     if p != -1]
            if not valid:
                print(" → all parse failures, skipping")
                continue

            llm_preds   = [v[0] for v in valid]
            true_labels = [v[1] for v in valid]
            pos_rate    = float(np.mean(llm_preds))
            accuracy    = float(np.mean(
                [p == l for p, l in zip(llm_preds, true_labels)]
            ))

            print(f"  pos_rate={pos_rate:.3f}  acc={accuracy:.3f}  "
                  f"valid={len(valid)}/{len(texts)}")

            yr_result[group_key] = {
                "pos_rate": pos_rate,
                "accuracy": accuracy,
                "n_valid":  len(valid),
                "n_total":  len(texts),
            }

        results[year] = yr_result
        checkpoint[yr_key] = yr_result
        with open(CHECKPOINT, "w") as f:
            json.dump(checkpoint, f, indent=2)
        print(f"  → checkpoint saved")

    # ── ALT analysis ─────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("ALT ANALYSIS — LLM (llama3.2)")
    print("="*60)

    # Build per-group time series
    group_keys = [
        ("Male",   "White",    "sex",  "Male"),
        ("Female", "White",    "sex",  "Female"),
        ("Male",   "NonWhite", "race", "White"),    # majority for race is White
        ("Female", "NonWhite", "race", "NonWhite"),
    ]

    records = []
    for year in YEARS:
        t = year - 2014          # t=0 at baseline
        yr = results.get(year, {})
        for sex_l, race_l, attr, grp in group_keys:
            key = f"{sex_l}_{race_l}"
            if key in yr:
                records.append({
                    "year": year, "t": t,
                    "attr": attr, "group": grp,
                    "pos_rate": yr[key]["pos_rate"],
                })

    df_rec = pd.DataFrame(records)
    df_rec.to_csv(OUT_DIR / "llm_trajectory.csv", index=False)
    print(f"\nTrajectory saved → {OUT_DIR}/llm_trajectory.csv")

    # Baseline positive rates at t=0
    baseline = {}
    base_yr = df_rec[df_rec["year"] == 2014]
    for _, row in base_yr.iterrows():
        baseline[(row["attr"], row["group"])] = row["pos_rate"]

    # Degradation
    df_rec["degradation"] = df_rec.apply(
        lambda r: abs(r["pos_rate"] - baseline.get((r["attr"], r["group"]), r["pos_rate"])),
        axis=1,
    )

    SLOPE_FLOOR = 1e-4
    alt_results = {}

    for attr in ["sex", "race"]:
        pairs = {"sex":  [("Male", "Female")],
                 "race": [("White", "NonWhite")]}[attr]
        for maj, mn in pairs:
            for grp_label in [maj, mn]:
                sub = df_rec[(df_rec["attr"] == attr) &
                             (df_rec["group"] == grp_label)].sort_values("t")
                if len(sub) < 3:
                    continue
                t_arr = sub["t"].values.astype(float)
                d_arr = sub["degradation"].values.astype(float)
                slope, intercept, r, p, se = stats.linregress(t_arr, d_arr)
                alt_results[f"{attr}_{grp_label}"] = {
                    "slope": slope, "se": se, "r2": r**2, "p": p,
                }

        # HR
        maj_key = f"{attr}_{maj}"
        min_key = f"{attr}_{mn}"
        if maj_key in alt_results and min_key in alt_results:
            r_maj = alt_results[maj_key]["slope"]
            r_min = alt_results[min_key]["slope"]
            both = abs(r_maj) >= SLOPE_FLOOR and abs(r_min) >= SLOPE_FLOOR
            if both:
                hr = r_min / r_maj
                se_maj = alt_results[maj_key]["se"]
                se_min = alt_results[min_key]["se"]
                log_hr_se = np.sqrt((se_min/r_min)**2 + (se_maj/r_maj)**2)
                ci_lo = np.exp(np.log(hr) - 1.96 * log_hr_se)
                ci_hi = np.exp(np.log(hr) + 1.96 * log_hr_se)
                alt_results[f"{attr}_HR"]     = hr
                alt_results[f"{attr}_CI_lo"]  = ci_lo
                alt_results[f"{attr}_CI_hi"]  = ci_hi
            else:
                alt_results[f"{attr}_HR"]    = float("nan")
                alt_results[f"{attr}_CI_lo"] = float("nan")
                alt_results[f"{attr}_CI_hi"] = float("nan")

    # Print and save summary
    print("\nDegradation rates and HRs:")
    for attr in ["sex", "race"]:
        pairs = {"sex":  [("Male", "Female")],
                 "race": [("White", "NonWhite")]}[attr]
        maj, mn = pairs[0]
        for lbl in [maj, mn]:
            k = f"{attr}_{lbl}"
            if k in alt_results:
                d = alt_results[k]
                print(f"  {attr}/{lbl:10s}: slope={d['slope']:.4e}  "
                      f"R²={d['r2']:.3f}  p={d['p']:.3f}")
        hr    = alt_results.get(f"{attr}_HR", float("nan"))
        ci_lo = alt_results.get(f"{attr}_CI_lo", float("nan"))
        ci_hi = alt_results.get(f"{attr}_CI_hi", float("nan"))
        print(f"  {attr} HR = {hr:.3f}  delta-method 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    with open(OUT_DIR / "llm_alt_results.json", "w") as f:
        json.dump({k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                   for k, v in alt_results.items()}, f, indent=2)
    print(f"\nALT results saved → {OUT_DIR}/llm_alt_results.json")

    # ── Trajectory table for paper ───────────────────────────────────────────
    print("\n\nTrajectory for paper:")
    print(f"{'Year':>6}  {'Male_W':>8}  {'Female_W':>10}  "
          f"{'Male_NW':>9}  {'Female_NW':>11}")
    for year in YEARS:
        yr = results.get(year, {})
        mw  = yr.get("Male_White",    {}).get("pos_rate", float("nan"))
        fw  = yr.get("Female_White",  {}).get("pos_rate", float("nan"))
        mnw = yr.get("Male_NonWhite", {}).get("pos_rate", float("nan"))
        fnw = yr.get("Female_NonWhite",{}).get("pos_rate",float("nan"))
        print(f"{year:>6}  {mw:>8.3f}  {fw:>10.3f}  {mnw:>9.3f}  {fnw:>11.3f}")

    return alt_results, df_rec


if __name__ == "__main__":
    t0 = time.time()
    alt_results, df_rec = run()
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed/60:.1f} min")

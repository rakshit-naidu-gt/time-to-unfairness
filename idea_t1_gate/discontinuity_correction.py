"""
Discontinuity-corrected degradation-rate estimation for the ACS Income task.

Motivation: a naive OLS fit of Delta_g(t) = r_g * t + c_g over the full
2014-2022 window (t in {0,...,8}, 2020 omitted) attributes the entire
2019->2021 jump to ongoing drift, producing HR_sex = 1.91 and HR_race = 0.19.
Appendix L (appendix_l_intersectional.tex) already shows this window contains
a sample-composition discontinuity at 2021 visible in every demographic
cell's snapshot size, consistent with the missing 2020 ACS collection and the
2020 Census race/ethnicity coding revision, and corrects for it in one
intersectional cell (Non-White Male) with a trend-plus-level-shift fit.

This script applies the same trend-plus-level-shift model to the four
marginal race/sex series used for the paper's headline HR values, and
bootstraps a 95% CI on the corrected HR via resampling of annual snapshots
(t=0 held fixed as the anchor, matching the paper's existing bootstrap
protocol in Appendix C).

Model: Delta_g(t) = r_g * t + c_g + delta_g * 1[t >= 7]   (t=7 <-> year 2021)

Usage: python3 discontinuity_correction.py
Writes: results/extended/income_compat/discontinuity_correction.json
"""
import json
import random
from pathlib import Path

RESULTS = Path(__file__).parent / "results" / "extended" / "income_compat"
TAU = 0.05
SHIFT_T = 7  # t index of the 2021 snapshot (2020 omitted)
SEED = 42
N_BOOT = 2000


def load_series(csv_path, attr, group):
    rows = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            vals = line.strip().split(",")
            if not vals or vals[0] == "":
                continue
            rec = dict(zip(header, vals))
            if rec["attr"] == attr and rec["group"] == group:
                rows.append((int(rec["t"]), float(rec["degradation"])))
    return sorted(rows)


def solve3(rows):
    """Solve 3x3 normal equations for [t, 1, 1[t>=SHIFT_T]] design."""
    XtX = [[0.0] * 3 for _ in range(3)]
    Xty = [0.0] * 3
    for row, y in rows:
        for i in range(3):
            Xty[i] += row[i] * y
            for j in range(3):
                XtX[i][j] += row[i] * row[j]
    A = [XtX[i][:] + [Xty[i]] for i in range(3)]
    for i in range(3):
        piv = A[i][i]
        if abs(piv) < 1e-12:
            return None
        for k in range(4):
            A[i][k] /= piv
        for j in range(3):
            if j != i:
                f = A[j][i]
                for k in range(4):
                    A[j][k] -= f * A[i][k]
    return A[0][3], A[1][3], A[2][3]  # r, c, delta


def fit_naive(data):
    rows = [([t, 1.0], y) for t, y in data]
    XtX = [[0.0] * 2 for _ in range(2)]
    Xty = [0.0] * 2
    for row, y in rows:
        for i in range(2):
            Xty[i] += row[i] * y
            for j in range(2):
                XtX[i][j] += row[i] * row[j]
    det = XtX[0][0] * XtX[1][1] - XtX[0][1] * XtX[1][0]
    r = (Xty[0] * XtX[1][1] - Xty[1] * XtX[0][1]) / det
    c = (XtX[0][0] * Xty[1] - XtX[1][0] * Xty[0]) / det
    return r, c


def fit_corrected(data):
    rows = [([t, 1.0, 1.0 if t >= SHIFT_T else 0.0], y) for t, y in data]
    return solve3(rows)


def r_squared(data, pred_fn):
    ys = [y for _, y in data]
    ybar = sum(ys) / len(ys)
    sst = sum((y - ybar) ** 2 for y in ys)
    sse = sum((y - pred_fn(t)) ** 2 for t, y in data)
    return 1 - sse / sst if sst > 0 else float("nan")


def ttc(r, c, tau=TAU):
    if r <= 0:
        return float("inf")
    val = (tau - c) / r
    return val if val > 0 else 0.0


def bootstrap_hr_corrected(data_a, data_b, n=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    idxs = list(range(1, len(data_a)))  # t=0 held fixed as anchor
    ratios = []
    for _ in range(n):
        samp = [rng.choice(idxs) for _ in idxs]
        da = [data_a[0]] + [data_a[i] for i in samp]
        db = [data_b[0]] + [data_b[i] for i in samp]
        fa, fb = fit_corrected(da), fit_corrected(db)
        if fa is None or fb is None or fb[0] == 0:
            continue
        ratios.append(fa[0] / fb[0])
    ratios.sort()
    n_valid = len(ratios)
    lo = ratios[int(0.025 * n_valid)]
    hi = ratios[int(0.975 * n_valid)]
    return lo, hi, n_valid, n


def analyze_pair(csv_path, attr, group_a, group_b, label):
    data_a = load_series(csv_path, attr, group_a)
    data_b = load_series(csv_path, attr, group_b)

    r_a_naive, c_a_naive = fit_naive(data_a)
    r_b_naive, c_b_naive = fit_naive(data_b)
    hr_naive = r_a_naive / r_b_naive

    fa = fit_corrected(data_a)
    fb = fit_corrected(data_b)
    r_a, c_a, delta_a = fa
    r_b, c_b, delta_b = fb
    hr_corrected = r_a / r_b

    r2_a = r_squared(data_a, lambda t: r_a * t + c_a + (delta_a if t >= SHIFT_T else 0))
    r2_b = r_squared(data_b, lambda t: r_b * t + c_b + (delta_b if t >= SHIFT_T else 0))

    lo, hi, n_valid, n_boot = bootstrap_hr_corrected(data_a, data_b)

    return {
        "label": label,
        "group_a": group_a, "group_b": group_b,
        "naive": {
            "r_a": r_a_naive, "r_b": r_b_naive, "hr": hr_naive,
        },
        "corrected": {
            "r_a": r_a, "c_a": c_a, "delta_a": delta_a, "r2_a": r2_a,
            "r_b": r_b, "c_b": c_b, "delta_b": delta_b, "r2_b": r2_b,
            "hr": hr_corrected,
            "hr_ci95": [lo, hi],
            "bootstrap_valid": n_valid, "bootstrap_n": n_boot,
            "ttc_a": ttc(r_a, c_a), "ttc_b": ttc(r_b, c_b),
        },
    }


def main():
    csv_path = RESULTS / "degradation.csv"
    out = {
        "model": "Delta_g(t) = r_g*t + c_g + delta_g*1[t>=7]  (t=7 <-> 2021)",
        "tau": TAU,
        "bootstrap": {"n": N_BOOT, "seed": SEED, "resamples": "annual snapshots, t=0 anchor fixed"},
        "sex": analyze_pair(csv_path, "sex", "Female", "Male", "Income, sex (Female/Male)"),
        "race": analyze_pair(csv_path, "race", "NonWhite", "White", "Income, race (Non-White/White)"),
    }
    out_path = RESULTS / "discontinuity_correction.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    for key in ("sex", "race"):
        d = out[key]
        print(f"\n{d['label']}:")
        print(f"  naive HR     = {d['naive']['hr']:.3f}")
        c = d["corrected"]
        print(f"  corrected HR = {c['hr']:.3f}  95% CI [{c['hr_ci95'][0]:.2f}, {c['hr_ci95'][1]:.2f}]"
              f"  (R2={c['r2_a']:.3f}/{c['r2_b']:.3f})")
        print(f"  corrected TTC: {d['group_a']}={c['ttc_a']:.1f}yr  {d['group_b']}={c['ttc_b']:.1f}yr")


if __name__ == "__main__":
    main()

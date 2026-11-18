"""
Standard algorithmic fairness metrics.

Implements Demographic Parity, Equalized Odds, and accuracy
as baseline comparison metrics alongside the ERD.

References:
    - Dwork et al. (2012). Fairness through awareness.
    - Hardt et al. (2016). Equality of opportunity in supervised learning.
"""
import numpy as np


def demographic_parity_gap(y_pred_proba, protected_attr, threshold=0.5):
    """
    Compute the Demographic Parity (DP) gap.

    DP requires that the positive prediction rate be equal across groups:
        P(Yhat=1 | A=a) = P(Yhat=1 | A=a')  for all a, a'

    The DP gap is the max pairwise difference in positive prediction rates.

    Parameters
    ----------
    y_pred_proba : array-like of shape (n,)
        Predicted probabilities.
    protected_attr : array-like of shape (n,)
        Protected attribute values.
    threshold : float
        Classification threshold (default 0.5).

    Returns
    -------
    dp_gap : float
        Max pairwise absolute difference in positive prediction rates.
    rates : dict
        Mapping from group label to positive prediction rate.
    """
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    protected_attr = np.asarray(protected_attr)

    y_bin = (y_pred_proba >= threshold).astype(int)
    groups = np.unique(protected_attr)
    rates = {a: float(np.mean(y_bin[protected_attr == a])) for a in groups}

    vals = list(rates.values())
    if len(vals) > 1:
        dp_gap = max(
            abs(vals[i] - vals[j])
            for i in range(len(vals))
            for j in range(i + 1, len(vals))
        )
    else:
        dp_gap = 0.0

    return float(dp_gap), rates


def equalized_odds_gap(y_pred_proba, y_true, protected_attr, threshold=0.5):
    """
    Compute the Equalized Odds (EO) gap.

    Equalized Odds requires equal TPR and FPR across groups:
        P(Yhat=1 | A=a, Y=y) = P(Yhat=1 | A=a', Y=y)  for y in {0,1}

    The EO gap is max(TPR_gap, FPR_gap).

    Parameters
    ----------
    y_pred_proba : array-like of shape (n,)
        Predicted probabilities.
    y_true : array-like of shape (n,)
        True binary labels.
    protected_attr : array-like of shape (n,)
        Protected attribute values.
    threshold : float
        Classification threshold.

    Returns
    -------
    eo_gap : float
        Max of TPR gap and FPR gap.
    tpr : dict
        True positive rate per group.
    fpr : dict
        False positive rate per group.
    """
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    protected_attr = np.asarray(protected_attr)

    y_bin = (y_pred_proba >= threshold).astype(int)
    groups = np.unique(protected_attr)

    tpr = {}
    fpr = {}

    for a in groups:
        mask = protected_attr == a
        pos_mask = mask & (y_true == 1)
        neg_mask = mask & (y_true == 0)

        tpr[a] = float(np.mean(y_bin[pos_mask])) if pos_mask.sum() > 0 else 0.0
        fpr[a] = float(np.mean(y_bin[neg_mask])) if neg_mask.sum() > 0 else 0.0

    tpr_vals = list(tpr.values())
    fpr_vals = list(fpr.values())

    if len(tpr_vals) > 1:
        tpr_gap = max(
            abs(tpr_vals[i] - tpr_vals[j])
            for i in range(len(tpr_vals))
            for j in range(i + 1, len(tpr_vals))
        )
        fpr_gap = max(
            abs(fpr_vals[i] - fpr_vals[j])
            for i in range(len(fpr_vals))
            for j in range(i + 1, len(fpr_vals))
        )
    else:
        tpr_gap = 0.0
        fpr_gap = 0.0

    eo_gap = max(tpr_gap, fpr_gap)
    return float(eo_gap), tpr, fpr


def accuracy(y_pred_proba, y_true, threshold=0.5):
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_pred_proba : array-like of shape (n,)
        Predicted probabilities.
    y_true : array-like of shape (n,)
        True binary labels.
    threshold : float
        Classification threshold.

    Returns
    -------
    acc : float
        Fraction of correct predictions.
    """
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    return float(np.mean((y_pred_proba >= threshold) == y_true))


def calibration_gap(y_pred_proba, y_true, protected_attr, n_bins=10):
    """
    Compute calibration gap between groups.

    Checks whether predicted probabilities match empirical frequencies
    equally across demographic groups.

    Parameters
    ----------
    y_pred_proba : array-like of shape (n,)
        Predicted probabilities.
    y_true : array-like of shape (n,)
        True binary labels.
    protected_attr : array-like of shape (n,)
        Protected attribute values.
    n_bins : int
        Number of probability bins.

    Returns
    -------
    cal_gap : float
        Mean absolute calibration error gap across groups.
    group_cal : dict
        Per-group mean absolute calibration error.
    """
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    protected_attr = np.asarray(protected_attr)

    bins = np.linspace(0, 1, n_bins + 1)
    groups = np.unique(protected_attr)
    group_cal = {}

    for a in groups:
        mask = protected_attr == a
        preds_a = y_pred_proba[mask]
        true_a = y_true[mask]

        bin_errors = []
        for b in range(n_bins):
            bin_mask = (preds_a >= bins[b]) & (preds_a < bins[b + 1])
            if bin_mask.sum() > 0:
                mean_pred = float(np.mean(preds_a[bin_mask]))
                mean_true = float(np.mean(true_a[bin_mask]))
                bin_errors.append(abs(mean_pred - mean_true))

        group_cal[a] = float(np.mean(bin_errors)) if bin_errors else 0.0

    cal_vals = list(group_cal.values())
    if len(cal_vals) > 1:
        cal_gap = max(
            abs(cal_vals[i] - cal_vals[j])
            for i in range(len(cal_vals))
            for j in range(i + 1, len(cal_vals))
        )
    else:
        cal_gap = 0.0

    return float(cal_gap), group_cal


def individual_fairness_score(y_pred_proba, X, k=10, metric="euclidean"):
    """
    Approximate individual fairness: similar individuals should get similar predictions.

    Computes the mean absolute prediction difference for k-nearest neighbor pairs.

    Parameters
    ----------
    y_pred_proba : array-like of shape (n,)
        Predicted probabilities.
    X : array-like of shape (n, d)
        Feature matrix (used to find similar individuals).
    k : int
        Number of nearest neighbors.
    metric : str
        Distance metric.

    Returns
    -------
    if_score : float
        Mean absolute prediction difference among k-NN pairs (lower = more individually fair).
    """
    from sklearn.neighbors import NearestNeighbors

    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    X = np.asarray(X, dtype=float)

    # Use a subsample for large datasets
    n = len(X)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        X = X[idx]
        y_pred_proba = y_pred_proba[idx]

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    distances, indices = nbrs.kneighbors(X)

    diffs = []
    for i in range(len(X)):
        for j in indices[i][1:]:  # skip self
            diffs.append(abs(y_pred_proba[i] - y_pred_proba[j]))

    return float(np.mean(diffs))


def compute_all_metrics(y_pred_proba, y_true, protected_attr, q_scores, threshold=0.5):
    """
    Convenience function: compute DP gap, EO gap, accuracy together.

    Parameters
    ----------
    y_pred_proba : array-like
        Predicted probabilities.
    y_true : array-like
        True labels.
    protected_attr : array-like
        Protected attribute.
    q_scores : array-like
        Qualification scores (not used here, included for API consistency).
    threshold : float
        Classification threshold.

    Returns
    -------
    results : dict
        Keys: 'accuracy', 'dp_gap', 'eo_gap', 'dp_rates', 'tpr', 'fpr'
    """
    acc = accuracy(y_pred_proba, y_true, threshold)
    dp_gap, dp_rates = demographic_parity_gap(y_pred_proba, protected_attr, threshold)
    eo_gap, tpr, fpr = equalized_odds_gap(y_pred_proba, y_true, protected_attr, threshold)

    return {
        "accuracy": acc,
        "dp_gap": dp_gap,
        "eo_gap": eo_gap,
        "dp_rates": dp_rates,
        "tpr": tpr,
        "fpr": fpr,
    }

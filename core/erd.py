"""
Core ERD (Equity Ratio Disparity) metric implementation.

Based on Adams' equity theory: an individual's outcome should be proportional
to their inputs/qualifications. ERD measures the disparity in outcome-to-
qualification ratios across demographic groups.

Reference: Adams, J.S. (1963). Towards an understanding of inequity.
           Journal of Abnormal and Social Psychology, 67(5), 422-436.
"""
import numpy as np


def compute_equity_ratios(y_pred, q_scores, eps=1e-6):
    """
    Compute per-individual equity ratios rho_i = y_pred_i / q(X_L_i).

    The equity ratio captures how much outcome an individual receives
    per unit of their qualification. An equitable system should have
    consistent ratios across demographic groups.

    Parameters
    ----------
    y_pred : array-like of shape (n,)
        Predicted probabilities or scores in [0, 1].
    q_scores : array-like of shape (n,)
        Qualification scores in (0, 1] derived from legitimate features.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    rho : np.ndarray of shape (n,)
        Per-individual equity ratios.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    q_scores = np.asarray(q_scores, dtype=float)
    return y_pred / (q_scores + eps)


def compute_erd(y_pred, q_scores, protected_attr, eps=1e-6):
    """
    Compute the Equity Ratio Disparity (ERD).

    ERD = max_{a,a'} |E[rho | A=a] - E[rho | A=a']|

    where rho_i = y_pred_i / q(X_L_i) is the equity ratio for individual i.

    A large ERD indicates that some groups receive systematically more or
    fewer outcomes per unit of qualification, suggesting bias.

    Parameters
    ----------
    y_pred : array-like of shape (n,)
        Predicted probabilities or scores.
    q_scores : array-like of shape (n,)
        Qualification scores from legitimate features.
    protected_attr : array-like of shape (n,)
        Protected attribute values (e.g., race, gender).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    erd_value : float
        The ERD scalar (max pairwise absolute difference of group mean equity ratios).
    group_means_dict : dict
        Mapping from group label to mean equity ratio E[rho | A=a].
    """
    y_pred = np.asarray(y_pred, dtype=float)
    q_scores = np.asarray(q_scores, dtype=float)
    protected_attr = np.asarray(protected_attr)

    rho = compute_equity_ratios(y_pred, q_scores, eps)
    groups = np.unique(protected_attr)
    group_means = {a: float(np.mean(rho[protected_attr == a])) for a in groups}
    vals = list(group_means.values())

    if len(vals) > 1:
        erd = max(
            abs(vals[i] - vals[j])
            for i in range(len(vals))
            for j in range(i + 1, len(vals))
        )
    else:
        erd = 0.0

    return float(erd), group_means


def compute_erd_intersectional(y_pred, q_scores, attr_arrays, eps=1e-6):
    """
    Compute ERD across intersectional groups (e.g., race x gender).

    Intersectional analysis reveals disparities that marginal analysis
    may miss — for instance, Black women may face compounded disadvantages
    not visible when examining race or gender alone.

    Parameters
    ----------
    y_pred : array-like of shape (n,)
        Predicted probabilities or scores.
    q_scores : array-like of shape (n,)
        Qualification scores from legitimate features.
    attr_arrays : list of array-like
        Each element is a 1D array for one protected attribute.
        Intersectional groups are formed as Cartesian products.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    erd_value : float
        Max pairwise absolute difference of intersectional group mean equity ratios.
    group_means_dict : dict
        Mapping from intersectional group label (e.g., "0_1") to mean equity ratio.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    q_scores = np.asarray(q_scores, dtype=float)

    rho = compute_equity_ratios(y_pred, q_scores, eps)
    n = len(y_pred)

    # Build intersectional group labels like "0_1", "1_0", etc.
    labels = np.array(
        [
            "_".join(str(np.asarray(attr_arrays[j])[i]) for j in range(len(attr_arrays)))
            for i in range(n)
        ]
    )

    unique = np.unique(labels)
    group_means = {}
    for g in unique:
        mask = labels == g
        if mask.sum() > 0:
            group_means[g] = float(np.mean(rho[mask]))

    vals = list(group_means.values())
    if len(vals) > 1:
        erd = max(
            abs(vals[i] - vals[j])
            for i in range(len(vals))
            for j in range(i + 1, len(vals))
        )
    else:
        erd = 0.0

    return float(erd), group_means


def estimate_erd_bootstrap_ci(
    y_pred, q_scores, protected_attr, n_bootstrap=500, alpha=0.05, eps=1e-6
):
    """
    Estimate a bootstrap confidence interval for the ERD.

    Provides uncertainty quantification around the ERD point estimate,
    enabling statistical hypothesis testing for fairness audits.

    Parameters
    ----------
    y_pred : array-like of shape (n,)
        Predicted probabilities or scores.
    q_scores : array-like of shape (n,)
        Qualification scores from legitimate features.
    protected_attr : array-like of shape (n,)
        Protected attribute values.
    n_bootstrap : int
        Number of bootstrap resamples.
    alpha : float
        Significance level. CI is [alpha/2, 1-alpha/2] percentiles.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    point : float
        ERD point estimate on original data.
    ci_lower : float
        Lower bound of (1-alpha) bootstrap CI.
    ci_upper : float
        Upper bound of (1-alpha) bootstrap CI.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    q_scores = np.asarray(q_scores, dtype=float)
    protected_attr = np.asarray(protected_attr)

    n = len(y_pred)
    samples = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        e, _ = compute_erd(y_pred[idx], q_scores[idx], protected_attr[idx], eps)
        samples.append(e)

    samples = np.array(samples)
    point, _ = compute_erd(y_pred, q_scores, protected_attr, eps)

    ci_lower = float(np.percentile(samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))

    return float(point), ci_lower, ci_upper


def compute_erd_decomposition(y_pred, q_scores, protected_attr, eps=1e-6):
    """
    Decompose ERD into within-group and between-group components.

    Useful for understanding whether disparity stems from different
    qualification distributions or from differential treatment at
    the same qualification level.

    Parameters
    ----------
    y_pred : array-like of shape (n,)
        Predicted probabilities or scores.
    q_scores : array-like of shape (n,)
        Qualification scores.
    protected_attr : array-like of shape (n,)
        Protected attribute values.
    eps : float
        Small constant.

    Returns
    -------
    decomp : dict
        Keys: 'overall_mean_rho', 'group_means', 'group_variances',
              'between_group_var', 'within_group_var', 'erd'
    """
    y_pred = np.asarray(y_pred, dtype=float)
    q_scores = np.asarray(q_scores, dtype=float)
    protected_attr = np.asarray(protected_attr)

    rho = compute_equity_ratios(y_pred, q_scores, eps)
    groups = np.unique(protected_attr)

    overall_mean = float(np.mean(rho))
    group_sizes = {}
    group_means = {}
    group_vars = {}

    for a in groups:
        mask = protected_attr == a
        group_sizes[a] = int(mask.sum())
        group_means[a] = float(np.mean(rho[mask]))
        group_vars[a] = float(np.var(rho[mask]))

    n = len(rho)
    # Between-group variance (weighted)
    between_var = sum(
        group_sizes[a] / n * (group_means[a] - overall_mean) ** 2 for a in groups
    )
    # Within-group variance (weighted)
    within_var = sum(group_sizes[a] / n * group_vars[a] for a in groups)

    erd, _ = compute_erd(y_pred, q_scores, protected_attr, eps)

    return {
        "overall_mean_rho": overall_mean,
        "group_means": group_means,
        "group_variances": group_vars,
        "group_sizes": group_sizes,
        "between_group_var": float(between_var),
        "within_group_var": float(within_var),
        "erd": erd,
    }

"""
Qualification scoring functions.

A qualification function q: X_L -> [eps, 1] maps an individual's
legitimate features to a score representing their "merit" or
"deserved outcome level" independent of protected attributes.

Three strategies are provided:
    1. LinearQualificationFunction: domain-knowledge-weighted combination
    2. LearnedQualificationFunction: auxiliary logistic regression on X_L
    3. EnsembleQualificationFunction: average of multiple strategies

The choice of qualification function is a value-laden modeling decision
that should be made transparently and with domain expertise.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.calibration import CalibratedClassifierCV


class LearnedQualificationFunction:
    """
    Strategy 2: Train an auxiliary logistic regression on X_L (legitimate
    features only) to predict Y. Use predicted probabilities as q(X_L),
    scaled to [eps, 1].

    This strategy learns from data which legitimate features are predictive
    of the outcome, providing a data-driven qualification measure.

    Parameters
    ----------
    eps : float
        Minimum qualification score (avoids division by zero in ERD).
    C : float
        Regularization strength for logistic regression (inverse).
    """

    def __init__(self, eps=0.05, C=1.0):
        self.model = LogisticRegression(max_iter=1000, C=C)
        self.scaler = MinMaxScaler(feature_range=(eps, 1.0))
        self.eps = eps
        self.C = C
        self.fitted = False

    def fit(self, X_L, y):
        """
        Fit the qualification model on legitimate features.

        Parameters
        ----------
        X_L : array-like of shape (n, d_L)
            Legitimate feature matrix.
        y : array-like of shape (n,)
            Target labels.

        Returns
        -------
        self
        """
        X_L = np.asarray(X_L, dtype=float)
        y = np.asarray(y)
        self.model.fit(X_L, y)
        raw = self.model.predict_proba(X_L)[:, 1].reshape(-1, 1)
        self.scaler.fit(raw)
        self.fitted = True
        return self

    def score(self, X_L):
        """
        Compute qualification scores for new samples.

        Parameters
        ----------
        X_L : array-like of shape (n, d_L)
            Legitimate feature matrix.

        Returns
        -------
        q : np.ndarray of shape (n,)
            Qualification scores in [eps, 1].
        """
        assert self.fitted, "Must call fit() before score()"
        X_L = np.asarray(X_L, dtype=float)
        raw = self.model.predict_proba(X_L)[:, 1].reshape(-1, 1)
        return self.scaler.transform(raw).ravel()

    def __repr__(self):
        return f"LearnedQualificationFunction(eps={self.eps}, C={self.C})"


class LinearQualificationFunction:
    """
    Strategy 1: Simple weighted linear combination of legitimate features,
    scaled to [eps, 1].

    Suitable when domain knowledge suggests a linear combination of
    features appropriately measures qualification.

    Parameters
    ----------
    weights : array-like or None
        Feature weights. If None, uses equal weighting.
    eps : float
        Minimum qualification score.
    """

    def __init__(self, weights=None, eps=0.05):
        self.weights = weights
        self.eps = eps
        self.scaler = MinMaxScaler(feature_range=(eps, 1.0))
        self.fitted = False

    def fit(self, X_L, y=None):
        """
        Fit the scaler on legitimate features.

        Parameters
        ----------
        X_L : array-like of shape (n, d_L)
            Legitimate feature matrix.
        y : ignored
            Included for API consistency.

        Returns
        -------
        self
        """
        X_L = np.asarray(X_L, dtype=float)

        if self.weights is None:
            self.weights = np.ones(X_L.shape[1]) / X_L.shape[1]

        self.weights = np.asarray(self.weights, dtype=float)
        self.weights = self.weights / (self.weights.sum() + 1e-10)  # normalize

        raw = (X_L @ self.weights).reshape(-1, 1)
        self.scaler.fit(raw)
        self.fitted = True
        return self

    def score(self, X_L):
        """
        Compute qualification scores.

        Parameters
        ----------
        X_L : array-like of shape (n, d_L)
            Legitimate feature matrix.

        Returns
        -------
        q : np.ndarray of shape (n,)
            Qualification scores in [eps, 1].
        """
        assert self.fitted, "Must call fit() before score()"
        X_L = np.asarray(X_L, dtype=float)
        raw = (X_L @ self.weights).reshape(-1, 1)
        return self.scaler.transform(raw).ravel()

    def __repr__(self):
        return f"LinearQualificationFunction(weights={self.weights}, eps={self.eps})"


class GBMQualificationFunction:
    """
    Strategy 3: Gradient Boosting Machine trained on X_L to predict Y.

    More flexible than logistic regression, can capture nonlinear
    qualification relationships.

    Parameters
    ----------
    eps : float
        Minimum qualification score.
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    """

    def __init__(self, eps=0.05, n_estimators=50, max_depth=3):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        self.scaler = MinMaxScaler(feature_range=(eps, 1.0))
        self.eps = eps
        self.fitted = False

    def fit(self, X_L, y):
        X_L = np.asarray(X_L, dtype=float)
        y = np.asarray(y)
        self.model.fit(X_L, y)
        raw = self.model.predict_proba(X_L)[:, 1].reshape(-1, 1)
        self.scaler.fit(raw)
        self.fitted = True
        return self

    def score(self, X_L):
        assert self.fitted, "Must call fit() before score()"
        X_L = np.asarray(X_L, dtype=float)
        raw = self.model.predict_proba(X_L)[:, 1].reshape(-1, 1)
        return self.scaler.transform(raw).ravel()

    def __repr__(self):
        return f"GBMQualificationFunction(eps={self.eps})"


class EnsembleQualificationFunction:
    """
    Strategy 4: Ensemble of multiple qualification strategies.

    Averages scores from a learned LR and a linear combination,
    providing more robust qualification estimates.

    Parameters
    ----------
    weights_linear : array-like or None
        Weights for the linear component. None = equal weights.
    eps : float
        Minimum qualification score.
    alpha : float
        Mixing parameter in [0,1]. alpha=1 is all learned, alpha=0 is all linear.
    """

    def __init__(self, weights_linear=None, eps=0.05, alpha=0.5):
        self.learned = LearnedQualificationFunction(eps=eps)
        self.linear = LinearQualificationFunction(weights=weights_linear, eps=eps)
        self.eps = eps
        self.alpha = alpha
        self.scaler = MinMaxScaler(feature_range=(eps, 1.0))
        self.fitted = False

    def fit(self, X_L, y):
        X_L = np.asarray(X_L, dtype=float)
        y = np.asarray(y)
        self.learned.fit(X_L, y)
        self.linear.fit(X_L, y)
        # Fit final scaler on blended scores
        blended = (
            self.alpha * self.learned.score(X_L)
            + (1 - self.alpha) * self.linear.score(X_L)
        ).reshape(-1, 1)
        self.scaler.fit(blended)
        self.fitted = True
        return self

    def score(self, X_L):
        assert self.fitted, "Must call fit() before score()"
        X_L = np.asarray(X_L, dtype=float)
        blended = (
            self.alpha * self.learned.score(X_L)
            + (1 - self.alpha) * self.linear.score(X_L)
        ).reshape(-1, 1)
        return self.scaler.transform(blended).ravel()

    def __repr__(self):
        return f"EnsembleQualificationFunction(alpha={self.alpha}, eps={self.eps})"


def validate_qualification_scores(q_scores, eps=0.0, name="q"):
    """
    Validate that qualification scores are in a valid range.

    Parameters
    ----------
    q_scores : array-like
        Qualification scores to validate.
    eps : float
        Minimum acceptable value.
    name : str
        Name for error messages.

    Returns
    -------
    q : np.ndarray
        Validated scores.

    Raises
    ------
    ValueError
        If scores contain NaN/Inf or are out of range.
    """
    q = np.asarray(q_scores, dtype=float)

    if np.any(np.isnan(q)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(q)):
        raise ValueError(f"{name} contains Inf values")
    if np.any(q < eps):
        raise ValueError(f"{name} has values below minimum eps={eps}: min={q.min():.4f}")

    return q

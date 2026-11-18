"""
ML models for fairness experiments.

Provides:
    - LogisticRegressionWrapper: sklearn LR with standardization
    - GradientBoostingWrapper: sklearn GBM with standardization
    - MLPWrapper: 3-layer PyTorch MLP with optional ERD regularization

The ERD regularizer penalizes the model for producing different mean
equity ratios across demographic groups, encouraging equitable treatment
per unit qualification.
"""
import numpy as np

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class LogisticRegressionWrapper:
    """
    Logistic regression classifier with input standardization.

    Attributes
    ----------
    name : str
        Model identifier for display/logging.
    """

    name = "LogisticRegression"

    def __init__(self, C=1.0, max_iter=1000):
        self.C = C
        self.max_iter = max_iter
        self.model = SklearnLR(max_iter=max_iter, C=C)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """Fit the model."""
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict_proba(self, X):
        """Return positive class probabilities."""
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)[:, 1]

    def __repr__(self):
        return f"LogisticRegressionWrapper(C={self.C})"


# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------

class GradientBoostingWrapper:
    """
    Gradient Boosting classifier with input standardization.

    Attributes
    ----------
    name : str
        Model identifier.
    """

    name = "GradientBoosting"

    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """Fit the model."""
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict_proba(self, X):
        """Return positive class probabilities."""
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)[:, 1]

    def __repr__(self):
        return f"GradientBoostingWrapper(n_estimators={self.n_estimators})"


# ---------------------------------------------------------------------------
# MLP Architecture
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    3-layer MLP with ReLU activations and sigmoid output.

    Architecture: input -> Linear(hidden) -> ReLU -> Linear(hidden) -> ReLU -> Linear(1) -> Sigmoid

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Size of each hidden layer.
    dropout : float
        Dropout probability (0 = no dropout).
    """

    def __init__(self, input_dim, hidden_dim=64, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# MLP Wrapper with ERD Regularization
# ---------------------------------------------------------------------------

class MLPWrapper:
    """
    3-layer MLP with optional ERD regularization.

    The ERD regularizer adds a penalty term to the loss:
        L_total = BCE(y_pred, y_true) + lambda_erd * ERD(y_pred, q, A)

    where ERD = |E[y/q | A=0] - E[y/q | A=1]| computed on the current batch.

    Parameters
    ----------
    lambda_erd : float
        ERD regularization strength. 0 disables regularization.
    hidden_dim : int
        MLP hidden layer size.
    epochs : int
        Training epochs.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size.
    dropout : float
        Dropout probability.
    device : str or None
        Device ('cpu', 'cuda', or None for auto-detection).

    Attributes
    ----------
    name : str
        Model identifier (reflects lambda_erd if > 0).
    """

    name = "MLP"

    def __init__(
        self,
        lambda_erd=0.0,
        hidden_dim=64,
        epochs=50,
        lr=1e-3,
        batch_size=256,
        dropout=0.0,
        device=None,
    ):
        self.lambda_erd = lambda_erd
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None

        if lambda_erd > 0:
            self.name = f"MLP_ERD(λ={lambda_erd})"

    def fit(self, X, y, X_L=None, q_scores=None, protected_attr=None):
        """
        Train the MLP.

        Parameters
        ----------
        X : array-like of shape (n, d)
            Full feature matrix.
        y : array-like of shape (n,)
            Binary labels.
        X_L : array-like or None
            Legitimate features (unused directly, q_scores are used).
        q_scores : array-like or None
            Qualification scores. Required if lambda_erd > 0.
        protected_attr : array-like or None
            Protected attribute (binary or multi-class). Required if lambda_erd > 0.

        Returns
        -------
        self
        """
        torch.manual_seed(42)
        np.random.seed(42)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        Xs = self.scaler.fit_transform(X)
        X_t = torch.tensor(Xs, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)

        # Determine whether to use ERD regularization
        use_erd = (
            self.lambda_erd > 0
            and q_scores is not None
            and protected_attr is not None
        )

        if use_erd:
            q_scores = np.asarray(q_scores, dtype=float)
            protected_attr = np.asarray(protected_attr)
            q_t = torch.tensor(q_scores, dtype=torch.float32).to(self.device)
            a_t = torch.tensor(protected_attr, dtype=torch.long).to(self.device)
        else:
            q_t = None
            a_t = None

        self.model = MLP(X_t.shape[1], self.hidden_dim, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        bce = nn.BCELoss()
        n = len(X_t)

        self.model.train()
        for epoch in range(self.epochs):
            idx = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                xb = X_t[batch_idx]
                yb = y_t[batch_idx]

                optimizer.zero_grad()
                pred = self.model(xb)
                loss = bce(pred, yb)

                if use_erd and q_t is not None and a_t is not None:
                    qb = q_t[batch_idx]
                    ab = a_t[batch_idx]
                    rho = pred / (qb + 1e-6)
                    groups = torch.unique(ab)

                    if len(groups) >= 2:
                        group_means = []
                        for g in groups:
                            g_mask = ab == g
                            if g_mask.sum() > 0:
                                group_means.append(rho[g_mask].mean())

                        if len(group_means) >= 2:
                            # Compute max pairwise ERD loss
                            erd_loss = torch.tensor(0.0, device=self.device)
                            for i in range(len(group_means)):
                                for j in range(i + 1, len(group_means)):
                                    erd_loss = torch.max(
                                        erd_loss,
                                        torch.abs(group_means[i] - group_means[j]),
                                    )
                            loss = loss + self.lambda_erd * erd_loss

                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

        return self

    def predict_proba(self, X):
        """
        Predict positive class probabilities.

        Parameters
        ----------
        X : array-like of shape (n, d)
            Feature matrix.

        Returns
        -------
        proba : np.ndarray of shape (n,)
            Predicted probabilities in [0, 1].
        """
        assert self.model is not None, "Must call fit() before predict_proba()"
        X = np.asarray(X, dtype=float)
        Xs = self.scaler.transform(X)
        X_t = torch.tensor(Xs, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_t).cpu().numpy()

    def __repr__(self):
        return (
            f"MLPWrapper(lambda_erd={self.lambda_erd}, "
            f"hidden_dim={self.hidden_dim}, epochs={self.epochs})"
        )


# ---------------------------------------------------------------------------
# Utility: get all model variants
# ---------------------------------------------------------------------------

def get_standard_models():
    """
    Return a list of standard models for Experiment 1.

    Returns
    -------
    models : list
        [LogisticRegressionWrapper, GradientBoostingWrapper, MLPWrapper]
    """
    return [
        LogisticRegressionWrapper(),
        GradientBoostingWrapper(),
        MLPWrapper(lambda_erd=0.0, epochs=50),
    ]


def get_erd_regularized_models(lambdas=None):
    """
    Return MLP models with varying ERD regularization strengths.

    Parameters
    ----------
    lambdas : list or None
        List of lambda values. Default: [0, 0.01, 0.1, 1, 10].

    Returns
    -------
    models : list of MLPWrapper
        One model per lambda value.
    """
    if lambdas is None:
        lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
    return [MLPWrapper(lambda_erd=lam, epochs=50) for lam in lambdas]

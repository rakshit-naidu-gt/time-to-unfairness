"""
Core module for Adams Equity-Theoretic Fairness Framework.
"""
from .erd import compute_equity_ratios, compute_erd, compute_erd_intersectional, estimate_erd_bootstrap_ci
from .metrics import demographic_parity_gap, equalized_odds_gap, accuracy
from .qualification import LearnedQualificationFunction, LinearQualificationFunction
from .models import LogisticRegressionWrapper, GradientBoostingWrapper, MLPWrapper

__all__ = [
    "compute_equity_ratios",
    "compute_erd",
    "compute_erd_intersectional",
    "estimate_erd_bootstrap_ci",
    "demographic_parity_gap",
    "equalized_odds_gap",
    "accuracy",
    "LearnedQualificationFunction",
    "LinearQualificationFunction",
    "LogisticRegressionWrapper",
    "GradientBoostingWrapper",
    "MLPWrapper",
]

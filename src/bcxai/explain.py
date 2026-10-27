"""Explanations for the logistic regression model.

For a standardized logistic regression the log-odds of malignancy is

    logit(x) = b0 + sum_i w_i * (x_i - mu_i) / sigma_i

so each term ``w_i * (x_i - mu_i) / sigma_i`` is the exact, additive
contribution of feature i relative to an "average" patient in the training
set (whose standardized features are all zero, giving logit = b0). These
contributions are identical to the SHAP values of a linear model under a
feature-independence assumption, without needing the SHAP library, and the
dashboard can recompute them in the browser with the same formula.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline


def linear_parts(model: Pipeline) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (mean, scale, coef, intercept) from a scaler + logistic pipeline."""
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    return (
        scaler.mean_.astype(float),
        scaler.scale_.astype(float),
        clf.coef_.ravel().astype(float),
        float(clf.intercept_[0]),
    )


def contributions(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Per-feature log-odds contributions, shape (n_samples, n_features)."""
    mean, scale, coef, _ = linear_parts(model)
    z = (np.asarray(X, dtype=float) - mean) / scale
    return z * coef


def logit(model: Pipeline, X: np.ndarray) -> np.ndarray:
    _, _, _, intercept = linear_parts(model)
    return intercept + contributions(model, X).sum(axis=1)


def sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))


def global_importance(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Mean absolute contribution of each feature over a dataset."""
    return np.abs(contributions(model, X)).mean(axis=0)

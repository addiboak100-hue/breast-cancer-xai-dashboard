"""Tests for the data split, the explanation maths and the exported model."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from bcxai.data import RANDOM_STATE, load_dataset, split_dataset
from bcxai.explain import contributions, logit, sigmoid
from bcxai.train import tune_logistic

MODEL_JSON = Path(__file__).resolve().parents[1] / "docs" / "model.json"


@pytest.fixture(scope="module")
def data():
    X, y = load_dataset()
    return X, y, split_dataset(X, y)


@pytest.fixture(scope="module")
def fitted(data):
    _, _, split = data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return tune_logistic(split.X_train, split.y_train, cv).best_estimator_


@pytest.fixture(scope="module")
def exported():
    if not MODEL_JSON.exists():
        pytest.skip("docs/model.json missing; run `python -m bcxai.train` first")
    return json.loads(MODEL_JSON.read_text())


# ---------- data ----------

def test_dataset_shape_and_encoding(data):
    X, y, _ = data
    assert X.shape == (569, 30)
    # Malignant is the positive class.
    assert int(y.sum()) == 212


def test_split_is_stratified_and_disjoint(data):
    _, _, s = data
    assert len(s.X_train) == 455 and len(s.X_test) == 114
    assert set(s.X_train.index).isdisjoint(s.X_test.index)
    assert abs(s.y_train.mean() - s.y_test.mean()) < 0.01


# ---------- explanations ----------

def test_contributions_reproduce_decision_function(data, fitted):
    _, _, s = data
    X = s.X_test.to_numpy()
    np.testing.assert_allclose(logit(fitted, X), fitted.decision_function(s.X_test), atol=1e-9)
    np.testing.assert_allclose(
        sigmoid(logit(fitted, X)), fitted.predict_proba(s.X_test)[:, 1], atol=1e-9
    )


def test_average_case_has_zero_contributions(data, fitted):
    _, _, s = data
    mean = fitted.named_steps["scaler"].mean_
    np.testing.assert_allclose(contributions(fitted, mean[None, :]), 0, atol=1e-9)


# ---------- exported model used by the dashboard ----------

def test_export_schema(exported):
    assert exported["model"]["positive_class"] == "malignant"
    assert len(exported["features"]) == 30
    for f in exported["features"]:
        assert f["scale"] > 0
        assert f["min"] <= f["median_benign"] <= f["max"]
    assert {e["label"] for e in exported["examples"]} >= {"Clear benign", "Clear malignant"}
    assert len(exported["metrics"]["test_predictions"]) == 114


def test_export_matches_freshly_trained_model(exported, fitted):
    """Training is deterministic and the JSON is not stale."""
    clf = fitted.named_steps["clf"]
    coef, intercept = clf.coef_.ravel(), clf.intercept_[0]
    np.testing.assert_allclose([f["coef"] for f in exported["features"]], coef, rtol=1e-3, atol=1e-4)
    assert exported["model"]["intercept"] == pytest.approx(intercept, abs=1e-3)


def test_dashboard_formula_matches_exported_predictions(exported, data):
    """Re-implement the browser's maths from JSON alone and compare."""
    _, _, s = data
    f = exported["features"]
    mean = np.array([x["mean"] for x in f])
    scale = np.array([x["scale"] for x in f])
    coef = np.array([x["coef"] for x in f])
    z = (s.X_test.to_numpy() - mean) / scale
    p = sigmoid(exported["model"]["intercept"] + (z * coef).sum(axis=1))
    expected = np.array([row[0] for row in exported["metrics"]["test_predictions"]])
    np.testing.assert_allclose(p, expected, atol=1e-4)


def test_model_quality_floor(exported):
    m = exported["metrics"]
    assert m["roc_auc"] >= 0.98
    assert m["accuracy"] >= 0.94
    assert m["sensitivity"] >= 0.90
    assert exported["model"]["cv_roc_auc"] >= 0.98

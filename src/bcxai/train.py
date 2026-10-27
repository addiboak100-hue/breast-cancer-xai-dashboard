"""Train, compare and export the breast cancer classifier.

Run with:  python -m bcxai.train

Steps
1. Load WDBC and make a stratified 80/20 hold-out split.
2. Compare four model families with 5-fold stratified cross-validation on the
   training set only (the test set is never used for model selection).
3. Tune the regularization strength of a standardized logistic regression,
   chosen as the deployed model because its predictions are exactly
   decomposable into per-feature contributions.
4. Evaluate once on the untouched test set.
5. Export ``docs/model.json`` (consumed by the dashboard), plus
   ``reports/metrics.json`` and the figures used in the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import __version__
from .data import (
    CLASS_NAMES,
    RANDOM_STATE,
    class_balance,
    feature_groups,
    load_dataset,
    split_dataset,
)
from .explain import contributions, global_importance, linear_parts

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
REPORTS_DIR = ROOT / "reports"

SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}


def candidate_models() -> dict[str, Pipeline]:
    def scaled(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    return {
        "Logistic regression": scaled(
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
        ),
        "SVM (RBF kernel)": scaled(SVC(random_state=RANDOM_STATE)),
        "Random forest": scaled(
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
        ),
        "Gradient boosting": scaled(GradientBoostingClassifier(random_state=RANDOM_STATE)),
    }


def compare_models(X, y, cv) -> list[dict]:
    rows = []
    for name, model in candidate_models().items():
        scores = cross_validate(model, X, y, cv=cv, scoring=SCORING)
        row = {"model": name}
        for key in SCORING:
            vals = scores[f"test_{key}"]
            row[key] = round(float(vals.mean()), 4)
            row[f"{key}_std"] = round(float(vals.std()), 4)
        rows.append(row)
    return sorted(rows, key=lambda r: r["roc_auc"], reverse=True)


def tune_logistic(X, y, cv) -> GridSearchCV:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )
    grid = {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}
    search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", refit=True)
    search.fit(X, y)
    return search


def evaluate(model: Pipeline, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    fpr, tpr, thr = roc_curve(y, proba)
    return {
        "threshold": 0.5,
        "accuracy": round(accuracy_score(y, pred), 4),
        "sensitivity": round(recall_score(y, pred), 4),
        "specificity": round(float(tn / (tn + fp)), 4),
        "precision": round(precision_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "brier": round(brier_score_loss(y, proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        # [false positive rate, true positive rate, probability threshold]
        "roc_curve": [
            [round(float(a), 4), round(float(b), 4), round(float(min(t, 1.0)), 4)]
            for a, b, t in zip(fpr, tpr, thr)
        ],
        # [predicted probability of malignancy, true label] for every test case,
        # so the dashboard can recompute metrics at any decision threshold.
        "test_predictions": [
            [round(float(p), 5), int(t)] for p, t in zip(proba, np.asarray(y))
        ],
    }


def pick_examples(model: Pipeline, X_test, y_test) -> list[dict]:
    """Choose a few real test-set cases that tell a story in the dashboard."""
    proba = model.predict_proba(X_test)[:, 1]
    y = y_test.to_numpy()
    idx = np.arange(len(y))
    chosen = {
        "Clear benign": idx[y == 0][np.argmin(proba[y == 0])],
        "Clear malignant": idx[y == 1][np.argmax(proba[y == 1])],
        "Borderline": idx[np.argmin(np.abs(proba - 0.5))],
    }
    wrong = idx[(proba >= 0.5).astype(int) != y]
    if len(wrong):
        chosen["Misclassified"] = wrong[np.argmax(np.abs(proba[wrong] - 0.5))]
    out = []
    for label, i in chosen.items():
        out.append(
            {
                "label": label,
                "true_class": CLASS_NAMES[int(y[i])],
                "values": [round(float(v), 6) for v in X_test.iloc[i].to_numpy()],
            }
        )
    return out


def build_export(search, X_train, X_test, y_train, y_test, comparison, metrics) -> dict:
    model = search.best_estimator_
    mean, scale, coef, intercept = linear_parts(model)
    cols = list(X_train.columns)

    imp = global_importance(model, X_test.to_numpy())
    perm = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=RANDOM_STATE
    )

    features = []
    for i, name in enumerate(cols):
        x = X_train[name]
        features.append(
            {
                "name": name,
                "mean": float(mean[i]),
                "scale": float(scale[i]),
                "coef": float(coef[i]),
                "min": float(x.min()),
                "max": float(x.max()),
                "median_benign": float(x[y_train == 0].median()),
                "median_malignant": float(x[y_train == 1].median()),
                "mean_abs_contribution": round(float(imp[i]), 5),
                "permutation_importance": round(float(perm.importances_mean[i]), 5),
            }
        )

    return {
        "version": __version__,
        "dataset": {
            "name": "Wisconsin Diagnostic Breast Cancer (WDBC)",
            "n_samples": int(len(X_train) + len(X_test)),
            "n_features": len(cols),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "class_balance": class_balance(pd.concat([y_train, y_test])),
            "groups": feature_groups(cols),
        },
        "model": {
            "type": "Standardized logistic regression (L2)",
            "C": float(search.best_params_["clf__C"]),
            "intercept": intercept,
            "positive_class": "malignant",
            "cv_roc_auc": round(float(search.best_score_), 4),
        },
        "features": features,
        "metrics": metrics,
        "comparison": comparison,
        "examples": pick_examples(model, X_test, y_test),
    }


def save_figures(export: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"
    blue, red = "#2a78d6", "#e34948"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": "#52514e",
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
        }
    )

    # Global importance: signed coefficients, top 12 by |coef|.
    feats = sorted(export["features"], key=lambda f: abs(f["coef"]), reverse=True)[:12]
    feats = feats[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = [red if f["coef"] > 0 else blue for f in feats]
    ax.barh([f["name"] for f in feats], [f["coef"] for f in feats], color=colors, height=0.6)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.grid(axis="x", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Weight on standardized feature (log-odds per 1 SD)")
    ax.set_title(
        "Top features: red pushes toward malignant, blue toward benign",
        loc="left", color=ink, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_weights.png", dpi=160)
    plt.close(fig)

    # ROC curve on the test set.
    roc = np.array(export["metrics"]["roc_curve"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=muted, lw=1, ls="--")
    ax.plot(roc[:, 0], roc[:, 1], color=blue, lw=2, drawstyle="steps-post")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(
        f"ROC curve, held-out test set (AUC {export['metrics']['roc_auc']:.3f})",
        loc="left", color=ink, fontsize=12,
    )
    ax.grid(color=grid, lw=0.6)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)

    # Confusion matrix.
    cm = export["metrics"]["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() * 1.15)
    for (r, c), v in np.ndenumerate(mat):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=16,
                color="#ffffff" if v > mat.max() / 2 else ink)
    ax.set_xticks([0, 1], ["Benign", "Malignant"])
    ax.set_yticks([0, 1], ["Benign", "Malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)", loc="left", color=ink, fontsize=12)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main() -> dict:
    DOCS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    X, y = load_dataset()
    split = split_dataset(X, y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("Comparing models with 5-fold CV on the training set...")
    comparison = compare_models(split.X_train, split.y_train, cv)
    for r in comparison:
        print(f"  {r['model']:<22} AUC {r['roc_auc']:.4f}  recall {r['recall']:.4f}  "
              f"acc {r['accuracy']:.4f}")

    print("Tuning logistic regression...")
    search = tune_logistic(split.X_train, split.y_train, cv)
    print(f"  best C = {search.best_params_['clf__C']}  CV AUC = {search.best_score_:.4f}")

    metrics = evaluate(search.best_estimator_, split.X_test, split.y_test)
    shown = {k: v for k, v in metrics.items() if k not in ("roc_curve", "test_predictions")}
    print("Test set:", shown)

    export = build_export(
        search, split.X_train, split.X_test, split.y_train, split.y_test, comparison, metrics
    )

    # Sanity check: the exported linear form reproduces sklearn exactly.
    model = search.best_estimator_
    X_test = split.X_test.to_numpy()
    manual = export["model"]["intercept"] + contributions(model, X_test).sum(axis=1)
    assert np.allclose(manual, model.decision_function(split.X_test))

    (DOCS_DIR / "model.json").write_text(json.dumps(export, indent=1))
    summary = {k: v for k, v in export.items() if k not in ("features", "examples")}
    summary["metrics"] = {
        k: v for k, v in metrics.items() if k not in ("roc_curve", "test_predictions")
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    save_figures(export)
    print(f"Wrote {DOCS_DIR / 'model.json'} and figures in {REPORTS_DIR}")
    return export


if __name__ == "__main__":
    main()

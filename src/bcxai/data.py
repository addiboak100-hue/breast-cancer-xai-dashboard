"""Dataset loading.

Uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset bundled with
scikit-learn: 569 samples, 30 numeric features computed from digitized
images of fine needle aspirates (FNA) of breast masses.

scikit-learn encodes the target as 0 = malignant, 1 = benign. We flip it so
that the positive class (1) is **malignant**, which is the class we care about
detecting. That makes "recall" mean "share of malignant cases caught".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2
CLASS_NAMES = ["benign", "malignant"]


@dataclass(frozen=True)
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return features and target with malignant encoded as 1."""
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()
    y = (1 - bunch.target).rename("malignant").astype(int)
    return X, y


def split_dataset(X: pd.DataFrame, y: pd.Series) -> Split:
    """Stratified hold-out split so both sets keep the same class balance."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return Split(X_train, X_test, y_train, y_test)


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Group the 30 features by statistic: mean, standard error, worst."""
    groups: dict[str, list[str]] = {"mean": [], "error": [], "worst": []}
    for col in columns:
        if col.startswith("mean "):
            groups["mean"].append(col)
        elif col.endswith(" error"):
            groups["error"].append(col)
        elif col.startswith("worst "):
            groups["worst"].append(col)
    return groups


def class_balance(y: pd.Series) -> dict[str, int]:
    counts = np.bincount(y.to_numpy(), minlength=2)
    return {name: int(c) for name, c in zip(CLASS_NAMES, counts)}

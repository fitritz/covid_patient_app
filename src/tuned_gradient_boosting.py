"""
Tuned Gradient Boosting Pipeline for PulseAI
-------------------------------------------
Optimised for the three-vital IoT dataset. The script compares a minimal
feature set with a richer targeted feature bundle and keeps the superior model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "iot_dataset_expanded.csv"
TARGETED_FILE = PROJECT_ROOT / "iot_dataset_targeted.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

BASE_FEATURES = ["Temperature Data", "ECG Data", "Pressure Data"]
TRAIN_TEST_RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass
class ModelResult:
    name: str
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    y_true: np.ndarray
    y_pred: np.ndarray
    feature_names: Tuple[str, ...]
    model: GradientBoostingClassifier

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.name,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_macro": self.f1_macro,
            "classification_report": classification_report(
                self.y_true, self.y_pred, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(self.y_true, self.y_pred).tolist(),
            "feature_importances": self.model.feature_importances_.tolist(),
            "feature_names": list(self.feature_names),
        }


def _load_base_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    return df.drop(columns=["Sl.No", "Patient ID"], errors="ignore")


def _load_targeted_dataset() -> pd.DataFrame:
    if not TARGETED_FILE.exists():
        raise FileNotFoundError(
            "Targeted feature dataset not found. Run target feature builder first."
        )
    return pd.read_csv(TARGETED_FILE)


def _split_features(df: pd.DataFrame, feature_cols: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df[list(feature_cols)].values
    y = df["Target"].values
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=TRAIN_TEST_RANDOM_STATE,
        stratify=y,
    )


def _train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=1800,
        learning_rate=0.0125,
        max_depth=5,
        subsample=0.8,
        max_features=3,
        min_samples_leaf=2,
        random_state=TRAIN_TEST_RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def _evaluate_model(model: GradientBoostingClassifier, X_test: np.ndarray, y_test: np.ndarray, *, name: str, feature_names: Tuple[str, ...]) -> ModelResult:
    y_pred = model.predict(X_test)
    return ModelResult(
        name=name,
        accuracy=accuracy_score(y_test, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_test, y_pred),
        f1_macro=f1_score(y_test, y_pred, average="macro"),
        y_true=y_test,
        y_pred=y_pred,
        feature_names=feature_names,
        model=model,
    )


def _run_variant(df: pd.DataFrame, feature_names: Tuple[str, ...], name: str) -> ModelResult:
    X_train, X_test, y_train, y_test = _split_features(df, feature_names)
    model = _train_gradient_boosting(X_train, y_train)
    return _evaluate_model(model, X_test, y_test, name=name, feature_names=feature_names)


def _persist_result(result: ModelResult) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "tuned_gradient_boosting.pkl"
    feature_path = MODEL_DIR / "tuned_gradient_boosting_features.json"
    report_path = REPORT_DIR / "tuned_gradient_boosting_metrics.json"

    joblib.dump(result.model, model_path)

    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump({"feature_names": list(result.feature_names)}, f, indent=4)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=4)

    print("\n💾 Saved artefacts:")
    print(f"   • Model          : {model_path}")
    print(f"   • Feature schema : {feature_path}")
    print(f"   • Metrics report : {report_path}")


def main() -> None:
    print("=" * 72)
    print("PulseAI Tuned Gradient Boosting")
    print("=" * 72)

    base_df = _load_base_dataset()
    targeted_df = _load_targeted_dataset()

    print(f"\nBase feature set columns: {BASE_FEATURES}")
    print(f"Targeted feature set columns: {len(targeted_df.columns) - 1} features")

    base_result = _run_variant(base_df, tuple(BASE_FEATURES), name="GB_Base_3_Features")
    targeted_features = tuple(
        col for col in targeted_df.columns if col != "Target"
    )
    targeted_result = _run_variant(
        targeted_df, targeted_features, name="GB_Targeted_Features"
    )

    results = sorted([base_result, targeted_result], key=lambda r: r.accuracy, reverse=True)
    best = results[0]

    print("\n📊 Performance summary:")
    for res in results:
        print(
            f"   {res.name:<28} | Accuracy: {res.accuracy:0.4f} | "
            f"Balanced Acc: {res.balanced_accuracy:0.4f} | F1-macro: {res.f1_macro:0.4f}"
        )

    print(f"\n🏆 Selected model: {best.name}")
    print(f"   Accuracy        : {best.accuracy:0.4%}")
    print(f"   Balanced Acc    : {best.balanced_accuracy:0.4%}")
    print(f"   F1 (macro)      : {best.f1_macro:0.4%}")
    print(f"   Confusion matrix:\n{confusion_matrix(best.y_true, best.y_pred)}")

    _persist_result(best)

    print("\nTop feature importances:")
    for feat, score in sorted(
        zip(best.feature_names, best.model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"   {feat:<30} {score:0.4f}")


if __name__ == "__main__":
    main()

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "maternal_health_risk.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)

    feature_columns = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
    target_column = "RiskLevel"

    X = df[feature_columns].copy()
    y_raw = df[target_column].str.strip().str.lower()

    normalization_map = {
        "high risk": "High",
        "mid risk": "Medium",
        "medium risk": "Medium",
        "low risk": "Low",
    }
    y = y_raw.map(normalization_map)

    if y.isna().any():
        missing = df[target_column][y.isna()].unique().tolist()
        raise ValueError(f"Unmapped risk labels encountered: {missing}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

    model_path = models_dir / "tuned_gradient_boosting.pkl"
    scaler_path = models_dir / "enhanced_scaler.pkl"
    features_path = models_dir / "tuned_gradient_boosting_features.json"
    label_map_path = models_dir / "risk_label_mapping.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_columns}, f, indent=2)

    label_mapping = {
        int(class_index): class_label
        for class_index, class_label in enumerate(label_encoder.classes_)
    }
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    metrics_path = models_dir / "tuned_gradient_boosting_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"classification_report": report}, f, indent=2)

    print("Model, scaler, and metadata saved to:")
    print(f"  {model_path}")
    print(f"  {scaler_path}")
    print(f"  {features_path}")
    print(f"  {label_map_path}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()

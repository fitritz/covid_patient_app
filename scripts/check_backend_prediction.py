import json
from pathlib import Path

import joblib
import numpy as np


def main():
    project_root = Path(__file__).resolve().parents[1]
    model = joblib.load(project_root / "models" / "tuned_gradient_boosting.pkl")
    scaler = joblib.load(project_root / "models" / "enhanced_scaler.pkl")

    with open(project_root / "models" / "tuned_gradient_boosting_features.json", "r", encoding="utf-8") as f:
        feature_payload = json.load(f)
    feature_names = feature_payload.get("feature_names", feature_payload)

    with open(project_root / "models" / "risk_label_mapping.json", "r", encoding="utf-8") as f:
        risk_map = {int(k): v for k, v in json.load(f).items()}

    sample = {
        "Age": 32,
        "SystolicBP": 140,
        "DiastolicBP": 90,
        "BS": 8.5,
        "BodyTemp": (37.2 * 9 / 5) + 32,
        "HeartRate": 88,
    }

    row = np.array([[sample[name] for name in feature_names]])
    scaled = scaler.transform(row)
    pred = model.predict(scaled)
    risk = risk_map.get(int(pred[0]), "N/A")

    print(f"Predicted risk: {risk}")


if __name__ == "__main__":
    main()

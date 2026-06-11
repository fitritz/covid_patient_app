"""
PulseAI Targeted Feature Builder
--------------------------------
Generates lightweight, data-aware features from the three vital signals to
support classical machine learning models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "iot_dataset_expanded.csv"
OUTPUT_FILE = PROJECT_ROOT / "iot_dataset_targeted.csv"

BASE_FEATURES: List[str] = ["Temperature Data", "ECG Data", "Pressure Data"]


@dataclass
class TargetedFeatureBuilder:
    """Creates compact feature sets derived from the three raw vitals."""

    clip_quantile: float = 0.99

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe with additional statistical features.

        The transformations rely only on summary statistics so they can be
        replicated consistently on future batches of data.
        """

        feature_df = df.copy()

        for col in ["Sl.No", "Patient ID"]:
            if col in feature_df.columns:
                feature_df = feature_df.drop(columns=col)

        if not set(BASE_FEATURES).issubset(feature_df.columns):
            missing = set(BASE_FEATURES) - set(feature_df.columns)
            raise ValueError(f"Dataset missing required columns: {missing}")

        vitals = feature_df[BASE_FEATURES].copy()

        # Basic centered features
        for col in BASE_FEATURES:
            mean = vitals[col].mean()
            std = vitals[col].std() + 1e-8
            feature_df[f"{col}_Centered"] = vitals[col] - mean
            feature_df[f"{col}_ZScore"] = (vitals[col] - mean) / std

        # Log and root transforms to tame heavy tails (especially ECG)
        feature_df["ECG_Log1p"] = np.log1p(np.clip(vitals["ECG Data"], a_min=0, a_max=None))
        feature_df["Temperature_Log"] = np.log1p(vitals["Temperature Data"] - vitals["Temperature Data"].min() + 1)
        feature_df["Pressure_Log"] = np.log1p(vitals["Pressure Data"] - vitals["Pressure Data"].min() + 1)
        feature_df["ECG_Sqrt"] = np.sqrt(np.abs(vitals["ECG Data"]))

        # Pairwise ratios and aggregations
        feature_df["Temp_to_Pressure"] = vitals["Temperature Data"] / (vitals["Pressure Data"] + 1e-6)
        feature_df["ECG_to_Temperature"] = vitals["ECG Data"] / (vitals["Temperature Data"] + 1e-6)
        feature_df["Vitals_Sum"] = vitals.sum(axis=1)
        feature_df["Vitals_Mean"] = vitals.mean(axis=1)
        feature_df["Vitals_Range"] = vitals.max(axis=1) - vitals.min(axis=1)
        feature_df["Vitals_Std"] = vitals.std(axis=1)

        # Quantile clipping indicator to flag extreme ECG spikes
        upper_clip = vitals["ECG Data"].quantile(self.clip_quantile)
        feature_df["ECG_Clipped"] = np.minimum(vitals["ECG Data"], upper_clip)
        feature_df["ECG_Is_Clipped"] = (vitals["ECG Data"] > upper_clip).astype(int)

        return feature_df

    def save(self, df: pd.DataFrame, output_path: Path = OUTPUT_FILE) -> Path:
        """Persist transformed dataset to disk and return the file path."""
        transformed = self.transform(df)
        transformed.to_csv(output_path, index=False)
        return output_path


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    builder = TargetedFeatureBuilder()
    output_path = builder.save(df)
    print(f"✅ Targeted feature dataset saved to {output_path}")
    print(f"   Shape: {pd.read_csv(output_path).shape}")


if __name__ == "__main__":
    main()

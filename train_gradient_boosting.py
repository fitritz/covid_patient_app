import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

print("=" * 80)
print("TRAINING NEW GRADIENT BOOSTING MODEL")
print("=" * 80)

# Load dataset
print("\n[1/7] Loading dataset...")
df = pd.read_csv('maternal_health_risk.csv')
X = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']

print(f"  ✓ Loaded {len(df)} samples with {len(X.columns)} features")
print(f"  ✓ Features: {list(X.columns)}")

# Encode labels
print("\n[2/7] Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"  ✓ Classes: {le.classes_}")

# Split data
print("\n[3/7] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"  ✓ Training set: {len(X_train)} samples")
print(f"  ✓ Test set: {len(X_test)} samples")

# Scale features
print("\n[4/7] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  ✓ Features scaled using StandardScaler")

# Apply SMOTE
print("\n[5/7] Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"  ✓ Training samples after SMOTE: {len(X_train_resampled)}")

# Train Gradient Boosting model
print("\n[6/7] Training Gradient Boosting Classifier...")
print("  Parameters:")
print("    - n_estimators: 200")
print("    - learning_rate: 0.05")
print("    - max_depth: 5")
print("    - min_samples_split: 5")
print("    - min_samples_leaf: 2")
print("    - subsample: 0.8")

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train_resampled, y_train_resampled)
print(f"  ✓ Model trained successfully")

# Evaluate model
print("\n[7/7] Evaluating model performance...")
y_pred = gb_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n  ✓ Overall Accuracy: {accuracy*100:.2f}%")

print("\n  Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix Analysis
cm = confusion_matrix(y_test, y_pred)
print("  Confusion Matrix:")
print(cm)

# False Negatives Analysis
high_risk_idx = 0  # 'high risk' is at index 0
total_high_risk = cm[high_risk_idx].sum()
correctly_identified = cm[high_risk_idx][high_risk_idx]
false_negatives = total_high_risk - correctly_identified
high_risk_recall = (correctly_identified / total_high_risk) * 100

print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
print(f"  ║  HIGH-RISK DETECTION PERFORMANCE (Most Critical)            ║")
print(f"  ╠══════════════════════════════════════════════════════════════╣")
print(f"  ║  Total High-Risk Cases:     {total_high_risk:3d}                              ║")
print(f"  ║  Correctly Identified:      {correctly_identified:3d} (✅ Caught)                   ║")
print(f"  ║  False Negatives (Missed):  {false_negatives:3d} (❌ Missed)                    ║")
print(f"  ║  Recall (Sensitivity):      {high_risk_recall:5.1f}%                          ║")
print(f"  ╚══════════════════════════════════════════════════════════════╝")

# Save model and scaler
print("\n" + "=" * 80)
print("SAVING MODEL FILES")
print("=" * 80)

print("\n[1/2] Saving Gradient Boosting model...")
joblib.dump(gb_model, 'models/best_gradient_boosting_final.pkl')
print(f"  ✓ Saved to: models/best_gradient_boosting_final.pkl")

print("\n[2/2] Saving StandardScaler...")
joblib.dump(scaler, 'models/best_scaler_final.pkl')
print(f"  ✓ Saved to: models/best_scaler_final.pkl (overwritten)")

# Save metadata
print("\n[INFO] Saving model metadata...")
metadata = {
    'model_name': 'Gradient Boosting Classifier',
    'accuracy': f"{accuracy*100:.2f}%",
    'high_risk_recall': f"{high_risk_recall:.2f}%",
    'false_negatives': int(false_negatives),
    'total_high_risk': int(total_high_risk),
    'features': list(X.columns),
    'classes': list(le.classes_),
    'training_samples': len(X_train_resampled),
    'test_samples': len(X_test)
}

import json
with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ Saved to: models/model_metadata.json")

print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\n  New Model: Gradient Boosting Classifier")
print(f"  Accuracy: {accuracy*100:.2f}%")
print(f"  High-Risk Recall: {high_risk_recall:.2f}%")
print(f"  False Negatives: {false_negatives}/{total_high_risk}")
print(f"\n  🎯 Ready for deployment!")
print("=" * 80)

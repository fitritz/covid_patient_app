"""
PulseAI - Diagnostic Analysis
Investigating accuracy drop after feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("PulseAI Diagnostic Analysis")
print("Investigating Accuracy Drop")
print("="*70)

# Test 1: Original dataset with basic features
print("\n🔍 TEST 1: Original Dataset (Baseline)")
print("-"*70)
df_original = pd.read_csv('iot_dataset_expanded.csv')
print(f"Samples: {len(df_original)}")

# Remove duplicates
df_original = df_original.drop_duplicates()
print(f"After dedup: {len(df_original)}")

feature_cols_orig = ['Temperature Data', 'ECG Data', 'Pressure Data']
X_orig = df_original[feature_cols_orig].values
y_orig = df_original['Target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
acc_baseline = accuracy_score(y_test, rf.predict(X_test_scaled))

print(f"✅ Baseline Accuracy: {acc_baseline*100:.2f}%")

# Test 2: Augmented dataset with original features only
print("\n🔍 TEST 2: Augmented Dataset (Original Features Only)")
print("-"*70)
df_aug = pd.read_csv('dataset_augmented.csv')
print(f"Samples: {len(df_aug)}")

X_aug = df_aug[feature_cols_orig].values
y_aug = df_aug['Target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
acc_augmented = accuracy_score(y_test, rf.predict(X_test_scaled))

print(f"✅ Augmented Accuracy: {acc_augmented*100:.2f}%")

# Test 3: Engineered dataset with all features
print("\n🔍 TEST 3: Engineered Dataset (All 64 Features)")
print("-"*70)
df_eng = pd.read_csv('dataset_engineered.csv')
print(f"Samples: {len(df_eng)}")

feature_cols_eng = [col for col in df_eng.columns 
                   if col not in ['Patient ID', 'Target', 'Augmentation_Method', 'Sl.No']]
print(f"Features: {len(feature_cols_eng)}")

X_eng = df_eng[feature_cols_eng].values
y_eng = df_eng['Target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y_eng, test_size=0.2, random_state=42, stratify=y_eng
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
acc_engineered = accuracy_score(y_test, rf.predict(X_test_scaled))

print(f"✅ Engineered Accuracy: {acc_engineered*100:.2f}%")

# Test 4: Engineered with feature selection (top 20 features)
print("\n🔍 TEST 4: Top 20 Features (Feature Selection)")
print("-"*70)

# Get feature importances
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train_scaled, y_train)
importances = rf_full.feature_importances_

# Select top 20
top_indices = np.argsort(importances)[-20:]
X_train_selected = X_train_scaled[:, top_indices]
X_test_selected = X_test_scaled[:, top_indices]

rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)
acc_selected = accuracy_score(y_test, rf_selected.predict(X_test_selected))

print(f"✅ Top 20 Features Accuracy: {acc_selected*100:.2f}%")

# Print top features
top_features = [feature_cols_eng[i] for i in top_indices]
print(f"\nTop 20 Features:")
for i, (feat, imp) in enumerate(zip(top_features, importances[top_indices]), 1):
    print(f"   {i}. {feat}: {imp:.4f}")

# Summary
print("\n" + "="*70)
print("📊 DIAGNOSTIC SUMMARY")
print("="*70)
print(f"\n1. Baseline (53 samples, 3 features):     {acc_baseline*100:>6.2f}%")
print(f"2. Augmented (663 samples, 3 features):   {acc_augmented*100:>6.2f}%")
print(f"3. Engineered (663 samples, 64 features): {acc_engineered*100:>6.2f}%")
print(f"4. Selected (663 samples, 20 features):   {acc_selected*100:>6.2f}%")

print(f"\n💡 INSIGHTS:")

if acc_augmented > acc_baseline:
    print(f"   ✅ Data augmentation HELPED (+{(acc_augmented-acc_baseline)*100:.2f}%)")
else:
    print(f"   ⚠️  Data augmentation HURT ({(acc_augmented-acc_baseline)*100:.2f}%)")

if acc_engineered > acc_augmented:
    print(f"   ✅ Feature engineering HELPED (+{(acc_engineered-acc_augmented)*100:.2f}%)")
else:
    print(f"   ⚠️  Feature engineering HURT ({(acc_engineered-acc_augmented)*100:.2f}%)")
    print(f"   → Too many features may cause overfitting")
    print(f"   → Feature selection needed")

if acc_selected > acc_engineered:
    print(f"   ✅ Feature selection IMPROVED accuracy (+{(acc_selected-acc_engineered)*100:.2f}%)")

# Check for data leakage
print(f"\n🔍 Checking for potential issues...")

# Check if augmentation created unrealistic data
print(f"\n   Checking augmented data quality:")
print(f"   - Temperature range: [{df_aug['Temperature Data'].min():.1f}, {df_aug['Temperature Data'].max():.1f}]")
print(f"   - ECG range: [{df_aug['ECG Data'].min():.1f}, {df_aug['ECG Data'].max():.1f}]")
print(f"   - Pressure range: [{df_aug['Pressure Data'].min():.1f}, {df_aug['Pressure Data'].max():.1f}]")

# Check class distribution in test set
print(f"\n   Test set class distribution:")
print(f"   - {np.bincount(y_test)}")

print(f"\n✅ Diagnosis complete!")
print(f"\n🎯 RECOMMENDATION:")
if acc_selected > acc_baseline:
    print(f"   Use feature selection approach")
    print(f"   Expected accuracy: {acc_selected*100:.2f}%")
else:
    print(f"   Consider reverting to augmented data with original features")
    print(f"   Or use different feature engineering strategy")

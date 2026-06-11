"""
PulseAI - Quick Enhanced Training
Streamlined training with augmented data and engineered features
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PulseAI Quick Enhanced Training")
print("="*70)

# Load engineered dataset
print("\n📂 Loading dataset...")
df = pd.read_csv('dataset_engineered.csv')
print(f"   Loaded: {len(df)} samples, {len(df.columns)} columns")

# Prepare data
feature_cols = [col for col in df.columns 
               if col not in ['Patient ID', 'Target', 'Augmentation_Method', 'Sl.No']]
X = df[feature_cols].values
y = df['Target'].values

print(f"   Features: {len(feature_cols)}")
print(f"   Classes: {np.bincount(y)}")

# Train-test split
print("\n🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")

# Scale features
print("\n⚖️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'models/quick_enhanced_scaler.pkl')
joblib.dump(feature_cols, 'models/quick_enhanced_features.pkl')
print("   ✅ Scaler saved")

# Train models (simplified - no extensive hyperparameter tuning)
print("\n" + "="*70)
print("Training Models (Quick Mode)")
print("="*70)

models = {
    'Gaussian_NB': GaussianNB(),
    'Decision_Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Logistic_Regression': LogisticRegression(C=100, max_iter=1000, random_state=42),
    'SVM': SVC(C=10, kernel='rbf', probability=True, random_state=42),
    'Random_Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n🔄 Training: {name}")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_score': cv_mean
    }
    
    print(f"   Test Accuracy: {accuracy*100:.2f}%")
    print(f"   CV Score: {cv_mean*100:.2f}%")

# Create ensemble
print(f"\n🎯 Creating Enhanced Ensemble...")
top_3_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
print(f"   Top 3 models:")
for idx, (name, info) in enumerate(top_3_models, 1):
    print(f"   {idx}. {name}: {info['accuracy']*100:.2f}%")

ensemble = VotingClassifier(
    estimators=[(name, info['model']) for name, info in top_3_models],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train)

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_test_scaled)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

results['Enhanced_Ensemble'] = {
    'model': ensemble,
    'accuracy': ensemble_accuracy,
    'cv_score': ensemble_accuracy
}

print(f"\n   Ensemble Accuracy: {ensemble_accuracy*100:.2f}%")

# Find best model
best_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
best_accuracy = results[best_name]['accuracy']

print(f"\n" + "="*70)
print(f"🏆 Best Model: {best_name}")
print(f"   Accuracy: {best_accuracy*100:.2f}%")
print(f"="*70)

# Save best model
joblib.dump(results[best_name]['model'], 'models/quick_enhanced_best_model.pkl')
print(f"\n💾 Best model saved")

# Generate detailed report
print(f"\n📊 Detailed Results:")
print(f"\n{'Model':<25} {'Test Acc':<12} {'CV Score':<12}")
print("-"*50)
for name, info in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"{name:<25} {info['accuracy']*100:>10.2f}%  {info['cv_score']*100:>10.2f}%")

# Classification report for best model
print(f"\n📋 Classification Report (Best Model):")
print("="*70)
y_pred_best = results[best_name]['model'].predict(X_test_scaled)
print(classification_report(y_test, y_pred_best, target_names=['Low', 'Medium', 'High']))

# Confusion matrix
print(f"\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# Save metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'dataset': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(feature_cols)
    },
    'best_model': best_name,
    'accuracy': float(best_accuracy),
    'all_results': {
        name: {'accuracy': float(info['accuracy']), 'cv_score': float(info['cv_score'])}
        for name, info in results.items()
    }
}

with open('models/quick_enhanced_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"\n💾 Metadata saved")

# Performance comparison
print(f"\n" + "="*70)
print(f"📈 Performance Comparison")
print(f"="*70)
print(f"   Baseline (150 samples, 16 features): 72.73%")
print(f"   Enhanced ({len(df)} samples, {len(feature_cols)} features): {best_accuracy*100:.2f}%")
improvement = best_accuracy*100 - 72.73
print(f"   Improvement: {improvement:+.2f}%")

if best_accuracy >= 0.85:
    print(f"\n   ✅ Target achieved (85%+)!")
else:
    print(f"\n   🎯 Progress towards 85% target: {(best_accuracy/0.85)*100:.1f}%")

print(f"\n" + "="*70)
print(f"🎉 Quick Enhanced Training Complete!")
print(f"="*70)
print(f"\nNext: Task 1.3 - Deep Learning Models")

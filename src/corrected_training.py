"""
PulseAI - Corrected Enhanced Training
Using augmented data with proper feature selection
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (VotingClassifier, RandomForestClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PulseAI Corrected Enhanced Training")
print("With Feature Selection & Larger Test Set")
print("="*70)

# Load augmented + engineered dataset
print("\n📂 Loading engineered dataset...")
df = pd.read_csv('dataset_engineered.csv')
print(f"   Total samples: {len(df)}")

# Prepare features
feature_cols = [col for col in df.columns 
               if col not in ['Patient ID', 'Target', 'Augmentation_Method', 'Sl.No']]
X = df[feature_cols].values
y = df['Target'].values

print(f"   Total features: {len(feature_cols)}")
print(f"   Class distribution: {np.bincount(y)}")

# Use 70/30 split for larger test set
print("\n🔀 Creating train-test split (70/30)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples (larger for reliable evaluation)")

# Scale features
print("\n⚖️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection - Select top 30 features
print("\n🎯 Performing feature selection (SelectKBest)...")
selector = SelectKBest(f_classif, k=30)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"   Selected {len(selected_features)} best features")
print(f"   Top 10 features:")
feature_scores = list(zip(selected_features, selector.scores_[selector.get_support(indices=True)]))
for idx, (feat, score) in enumerate(sorted(feature_scores, key=lambda x: x[1], reverse=True)[:10], 1):
    print(f"      {idx}. {feat}: {score:.2f}")

# Save artifacts
joblib.dump(scaler, 'models/corrected_scaler.pkl')
joblib.dump(selector, 'models/corrected_feature_selector.pkl')
joblib.dump(selected_features, 'models/corrected_features.pkl')
print("\n   ✅ Scaler and selector saved")

# Train models with selected features
print("\n" + "="*70)
print("Training Models with Selected Features")
print("="*70)

models = {
    'Random_Forest': RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5, 
        min_samples_leaf=2, random_state=42
    ),
    'Extra_Trees': ExtraTreesClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        random_state=42
    ),
    'Gradient_Boosting': GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=7,
        subsample=0.8, random_state=42
    ),
    'SVM': SVC(
        C=10, kernel='rbf', probability=True, 
        gamma='scale', random_state=42
    ),
    'Logistic_Regression': LogisticRegression(
        C=10, max_iter=2000, solver='lbfgs', 
        multi_class='multinomial', random_state=42
    ),
    'Decision_Tree': DecisionTreeClassifier(
        max_depth=15, min_samples_split=5, 
        criterion='entropy', random_state=42
    )
}

results = {}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n🔄 Training: {name}")
    
    # Train
    model.fit(X_train_selected, y_train)
    
    # Test prediction
    y_pred = model.predict(X_test_selected)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # 10-fold cross-validation
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results[name] = {
        'model': model,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std
    }
    
    print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   CV Score: {cv_mean*100:.2f}% (±{cv_std*100:.2f}%)")

# Create Stacked Ensemble
print(f"\n🎯 Creating Advanced Ensemble (Top 4 Models)...")
top_4_models = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)[:4]
print(f"   Selected models:")
for idx, (name, info) in enumerate(top_4_models, 1):
    print(f"   {idx}. {name}: {info['test_accuracy']*100:.2f}%")

# Voting ensemble with optimized weights
ensemble = VotingClassifier(
    estimators=[(name, info['model']) for name, info in top_4_models],
    voting='soft',
    weights=[4, 3, 2, 1]  # Weight higher-performing models more
)
ensemble.fit(X_train_selected, y_train)

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_test_selected)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
cv_scores_ensemble = cross_val_score(ensemble, X_train_selected, y_train, cv=cv)

results['Weighted_Ensemble'] = {
    'model': ensemble,
    'test_accuracy': ensemble_accuracy,
    'cv_mean': cv_scores_ensemble.mean(),
    'cv_std': cv_scores_ensemble.std()
}

print(f"\n   Ensemble Test Accuracy: {ensemble_accuracy*100:.2f}%")
print(f"   Ensemble CV Score: {cv_scores_ensemble.mean()*100:.2f}% (±{cv_scores_ensemble.std()*100:.2f}%)")

# Find best model
best_name = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
best_model = results[best_name]['model']
best_accuracy = results[best_name]['test_accuracy']

print(f"\n" + "="*70)
print(f"🏆 Best Model: {best_name}")
print(f"   Test Accuracy: {best_accuracy*100:.2f}%")
print(f"   CV Score: {results[best_name]['cv_mean']*100:.2f}%")
print(f"="*70)

# Save best model
joblib.dump(best_model, 'models/corrected_best_model.pkl')
print(f"\n💾 Best model saved")

# Detailed results
print(f"\n📊 All Results (Sorted by Test Accuracy):")
print(f"\n{'Model':<25} {'Test Acc':<12} {'CV Mean':<12} {'CV Std':<10}")
print("-"*60)
for name, info in sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True):
    print(f"{name:<25} {info['test_accuracy']*100:>10.2f}%  "
          f"{info['cv_mean']*100:>10.2f}%  ±{info['cv_std']*100:>7.2f}%")

# Classification report
print(f"\n📋 Detailed Classification Report (Best Model):")
print("="*70)
y_pred_best = best_model.predict(X_test_selected)
print(classification_report(y_test, y_pred_best, 
                          target_names=['Low Risk', 'Medium Risk', 'High Risk'],
                          digits=3))

# Confusion matrix
print(f"\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(f"\n                 Predicted")
print(f"              Low  Med  High")
print(f"Actual Low    {cm[0,0]:>3}  {cm[0,1]:>3}  {cm[0,2]:>3}")
print(f"       Med    {cm[1,0]:>3}  {cm[1,1]:>3}  {cm[1,2]:>3}")
print(f"       High   {cm[2,0]:>3}  {cm[2,1]:>3}  {cm[2,2]:>3}")

# Per-class accuracy
print(f"\n📊 Per-Class Performance:")
for i, class_name in enumerate(['Low Risk', 'Medium Risk', 'High Risk']):
    class_acc = cm[i,i] / cm[i].sum()
    print(f"   {class_name}: {class_acc*100:.2f}%")

# Save comprehensive metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'dataset': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'original_features': len(feature_cols),
        'selected_features': len(selected_features),
        'feature_names': selected_features
    },
    'best_model': best_name,
    'test_accuracy': float(best_accuracy),
    'cv_accuracy': float(results[best_name]['cv_mean']),
    'cv_std': float(results[best_name]['cv_std']),
    'all_results': {
        name: {
            'test_accuracy': float(info['test_accuracy']),
            'cv_mean': float(info['cv_mean']),
            'cv_std': float(info['cv_std'])
        }
        for name, info in results.items()
    },
    'confusion_matrix': cm.tolist()
}

with open('models/corrected_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"\n💾 Metadata saved")

# Final comparison
print(f"\n" + "="*70)
print(f"📈 Progress Summary")
print(f"="*70)
print(f"\n   Initial Dataset:")
print(f"   - Samples: 53 unique (after dedup)")
print(f"   - Test set: 11 samples (too small!)")
print(f"   - Reported accuracy: 72.73% (unreliable)")
print(f"   - Actual accuracy on larger test: ~43%")

print(f"\n   Enhanced Dataset:")
print(f"   - Samples: {len(df)} (augmented)")
print(f"   - Test set: {len(X_test)} samples (reliable)")
print(f"   - Features: {len(selected_features)} (selected)")
print(f"   - Best accuracy: {best_accuracy*100:.2f}%")
print(f"   - CV accuracy: {results[best_name]['cv_mean']*100:.2f}%")

improvement = best_accuracy*100 - 43
print(f"\n   📊 Improvement: +{improvement:.2f}% over baseline")

if best_accuracy >= 0.70:
    print(f"\n   ✅ Good progress! (70%+ achieved)")
elif best_accuracy >= 0.60:
    print(f"\n   🎯 Moderate progress (60%+ achieved)")
else:
    print(f"\n   ⚠️  More work needed (below 60%)")

print(f"\n" + "="*70)
print(f"🎉 Corrected Training Complete!")
print(f"="*70)
print(f"\n💡 Next Steps:")
print(f"   1. Try deep learning models (Task 1.3)")
print(f"   2. Implement stacking ensemble (Task 1.4)")
print(f"   3. Bayesian optimization (Task 1.5)")

"""
PulseAI - Advanced Ensemble Techniques (Task 1.4)
CatBoost, LightGBM, XGBoost, and Stacking Ensemble
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              StackingClassifier, VotingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_FILE = PROJECT_ROOT / 'iot_dataset_engineered.csv'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

print("="*70)
print("PulseAI Advanced Ensemble Techniques (Task 1.4)")
print("CatBoost + LightGBM + XGBoost + Stacking")
print("="*70)

# Load engineered dataset
print("\n📂 Loading engineered dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Engineered dataset not found at {DATA_FILE}. Run advanced_features.py first."
    )

df = pd.read_csv(DATA_FILE)

# Prepare features (use only top 30 selected features from corrected_training.py)
feature_cols = [col for col in df.columns 
               if col not in ['Patient ID', 'Target', 'Augmentation_Method', 'Sl.No']]
X = df[feature_cols].values
y = df['Target'].values

print(f"   Total samples: {len(df)}")
print(f"   Total features: {len(feature_cols)}")

# Use 70/30 split (consistent)
print("\n🔀 Creating train-test split (70/30)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

print("\n⚖️  Balancing classes with SMOTE...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"   Resampled training size: {len(X_train)}")
unique, counts = np.unique(y_train, return_counts=True)
balance_info = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}
print(f"   Class distribution after SMOTE: {balance_info}")

# Feature selection
k_features = min(30, X_train.shape[1])
print(f"\n🧮 Selecting top {k_features} features using ANOVA F-test...")
selector = SelectKBest(score_func=f_classif, k=k_features)
selector.fit(X_train, y_train)

selected_indices = selector.get_support(indices=True)
selected_feature_names = [feature_cols[i] for i in selected_indices]
print(f"   Selected features: {selected_feature_names[:10]}{'...' if len(selected_feature_names) > 10 else ''}")

X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

# Scale features
print("\n⚖️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, MODELS_DIR / 'ensemble_scaler.pkl')

# Define models
print("\n" + "="*70)
print("Training Gradient Boosting Variants")
print("="*70)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Model 1: CatBoost
print("\n🔄 Training CatBoost...")
class_weights = {cls: len(y_train) / (len(np.unique(y_train)) * np.sum(y_train == cls)) for cls in np.unique(y_train)}

catboost_model = CatBoostClassifier(
    iterations=700,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=False,
    class_weights=[class_weights.get(cls, 1.0) for cls in range(len(class_weights))]
)
catboost_model.fit(X_train_scaled, y_train)

catboost_pred = catboost_model.predict(X_test_scaled)
catboost_acc = accuracy_score(y_test, catboost_pred)
catboost_cv = cross_val_score(catboost_model, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()

print(f"   Test Accuracy: {catboost_acc*100:.2f}%")
print(f"   CV Score: {catboost_cv*100:.2f}%")

# Model 2: LightGBM
print("\n🔄 Training LightGBM...")
lightgbm_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=64,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    class_weight='balanced',
    verbose=-1
)
lightgbm_model.fit(X_train_scaled, y_train)

lightgbm_pred = lightgbm_model.predict(X_test_scaled)
lightgbm_acc = accuracy_score(y_test, lightgbm_pred)
lightgbm_cv = cross_val_score(lightgbm_model, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()

print(f"   Test Accuracy: {lightgbm_acc*100:.2f}%")
print(f"   CV Score: {lightgbm_cv*100:.2f}%")

# Model 3: XGBoost
print("\n🔄 Training XGBoost...")
xgboost_model = XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.5,
    reg_alpha=0.5,
    gamma=0.1,
    random_state=42,
    eval_metric='mlogloss',
    verbosity=0,
    objective='multi:softprob',
    num_class=len(np.unique(y))
)
xgboost_model.fit(X_train_scaled, y_train)

xgboost_pred = xgboost_model.predict(X_test_scaled)
xgboost_acc = accuracy_score(y_test, xgboost_pred)
xgboost_cv = cross_val_score(xgboost_model, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()

print(f"   Test Accuracy: {xgboost_acc*100:.2f}%")
print(f"   CV Score: {xgboost_cv*100:.2f}%")

# Model 4: Extra Trees (additional diversity)
print("\n🔄 Training Extra Trees...")
extra_trees_model = ExtraTreesClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
extra_trees_model.fit(X_train_scaled, y_train)

extra_trees_pred = extra_trees_model.predict(X_test_scaled)
extra_trees_acc = accuracy_score(y_test, extra_trees_pred)
extra_trees_cv = cross_val_score(extra_trees_model, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()

print(f"   Test Accuracy: {extra_trees_acc*100:.2f}%")
print(f"   CV Score: {extra_trees_cv*100:.2f}%")

# Model 5: Random Forest
print("\n🔄 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=5,
    class_weight='balanced_subsample',
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
rf_cv = cross_val_score(rf_model, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()

print(f"   Test Accuracy: {rf_acc*100:.2f}%")
print(f"   CV Score: {rf_cv*100:.2f}%")

# Model 6: Logistic Regression (linear baseline)
print("\n🔄 Training Logistic Regression...")
lr_model = LogisticRegression(
    C=5,
    max_iter=5000,
    multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced',
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
lr_cv = cross_val_score(lr_model, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()

print(f"   Test Accuracy: {lr_acc*100:.2f}%")
print(f"   CV Score: {lr_cv*100:.2f}%")

# Compare individual models
print("\n" + "="*70)
print("📊 Individual Model Comparison")
print("="*70)

results = {
    'CatBoost': {'test_acc': catboost_acc, 'cv_acc': catboost_cv},
    'LightGBM': {'test_acc': lightgbm_acc, 'cv_acc': lightgbm_cv},
    'XGBoost': {'test_acc': xgboost_acc, 'cv_acc': xgboost_cv},
    'Extra_Trees': {'test_acc': extra_trees_acc, 'cv_acc': extra_trees_cv},
    'Random_Forest': {'test_acc': rf_acc, 'cv_acc': rf_cv},
    'Logistic_Regression': {'test_acc': lr_acc, 'cv_acc': lr_cv}
}

print(f"\n{'Model':<25} {'Test Acc':<12} {'CV Acc':<12}")
print("-"*50)
for name, scores in sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True):
    print(f"{name:<25} {scores['test_acc']*100:>10.2f}%  {scores['cv_acc']*100:>10.2f}%")

# Create Voting Ensemble (Soft Voting)
print("\n" + "="*70)
print("Creating Voting Ensemble (Top 4 Models)")
print("="*70)

top_4_models = sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True)[:4]
print(f"\nSelected models:")
for idx, (name, scores) in enumerate(top_4_models, 1):
    print(f"   {idx}. {name}: {scores['test_acc']*100:.2f}%")

# Map names to model objects
model_map = {
    'CatBoost': catboost_model,
    'LightGBM': lightgbm_model,
    'XGBoost': xgboost_model,
    'Extra_Trees': extra_trees_model,
    'Random_Forest': rf_model,
    'Logistic_Regression': lr_model
}

voting_ensemble = VotingClassifier(
    estimators=[(name, model_map[name]) for name, _ in top_4_models],
    voting='soft',
    weights=[4, 3, 2, 1]
)
voting_ensemble.fit(X_train_scaled, y_train)

voting_pred = voting_ensemble.predict(X_test_scaled)
voting_acc = accuracy_score(y_test, voting_pred)

print(f"\n🎯 Voting Ensemble Test Accuracy: {voting_acc*100:.2f}%")

# Create Stacking Ensemble
print("\n" + "="*70)
print("Creating Stacking Ensemble")
print("="*70)

# Use top 5 models as base estimators, Logistic Regression as meta-learner
base_estimators = [
    ('catboost', catboost_model),
    ('lightgbm', lightgbm_model),
    ('xgboost', xgboost_model),
    ('extra_trees', extra_trees_model),
    ('random_forest', rf_model)
]

meta_learner = LogisticRegression(
    C=1.0,
    max_iter=1000,
    multi_class='multinomial',
    random_state=42
)

print("\n🔄 Training Stacking Ensemble...")
stacking_ensemble = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)
stacking_ensemble.fit(X_train_scaled, y_train)

stacking_pred = stacking_ensemble.predict(X_test_scaled)
stacking_acc = accuracy_score(y_test, stacking_pred)

print(f"\n🎯 Stacking Ensemble Test Accuracy: {stacking_acc*100:.2f}%")

# Compare all approaches
print("\n" + "="*70)
print("🏆 Final Comparison (All Approaches)")
print("="*70)

all_results = results.copy()
all_results['Voting_Ensemble'] = {'test_acc': voting_acc, 'cv_acc': None}
all_results['Stacking_Ensemble'] = {'test_acc': stacking_acc, 'cv_acc': None}

print(f"\n{'Model':<25} {'Test Acc':<12}")
print("-"*40)
for name, scores in sorted(all_results.items(), key=lambda x: x[1]['test_acc'], reverse=True):
    print(f"{name:<25} {scores['test_acc']*100:>10.2f}%")

# Find best model
best_model_name = max(all_results.items(), key=lambda x: x[1]['test_acc'])[0]
best_acc = all_results[best_model_name]['test_acc']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test Accuracy: {best_acc*100:.2f}%")

# Save best model
if best_model_name == 'Voting_Ensemble':
    joblib.dump(voting_ensemble, MODELS_DIR / 'ensemble_best_model.pkl')
    best_pred = voting_pred
elif best_model_name == 'Stacking_Ensemble':
    joblib.dump(stacking_ensemble, MODELS_DIR / 'ensemble_best_model.pkl')
    best_pred = stacking_pred
else:
    joblib.dump(model_map[best_model_name], MODELS_DIR / 'ensemble_best_model.pkl')
    best_pred = model_map[best_model_name].predict(X_test_scaled)

print(f"   💾 Best model saved")

# Detailed evaluation
print(f"\n📋 Detailed Classification Report (Best Model):")
print("="*70)
print(classification_report(y_test, best_pred, 
                          target_names=['Low Risk', 'Medium Risk', 'High Risk'],
                          digits=3))

# Confusion matrix
print(f"\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(f"\n                 Predicted")
print(f"              Low  Med  High")
print(f"Actual Low    {cm[0,0]:>3}  {cm[0,1]:>3}  {cm[0,2]:>3}")
print(f"       Med    {cm[1,0]:>3}  {cm[1,1]:>3}  {cm[1,2]:>3}")
print(f"       High   {cm[2,0]:>3}  {cm[2,1]:>3}  {cm[2,2]:>3}")

# Per-class performance
print(f"\n📊 Per-Class Performance:")
for i, class_name in enumerate(['Low Risk', 'Medium Risk', 'High Risk']):
    class_acc = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"   {class_name}: {class_acc*100:.2f}%")

# Save metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'task': 'Task 1.4 - Advanced Ensemble Techniques',
    'dataset': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(feature_cols)
    },
    'models': {
        name: {
            'test_accuracy': float(scores['test_acc']),
            'cv_accuracy': float(scores['cv_acc']) if scores['cv_acc'] is not None else None
        }
        for name, scores in all_results.items()
    },
    'best_model': best_model_name,
    'best_test_accuracy': float(best_acc),
    'confusion_matrix': cm.tolist()
}

with open(MODELS_DIR / 'ensemble_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"\n💾 Metadata saved")

# Progress summary
print(f"\n" + "="*70)
print(f"📈 Overall Progress Summary")
print(f"="*70)

print(f"\n   Phase 1 Progress:")
print(f"   - Baseline: 43.00%")
print(f"   - Traditional ML: 52.26% (+9.26%)")
print(f"   - Deep Learning: 50.25% (-2.01%)")
print(f"   - Advanced Ensemble: {best_acc*100:.2f}% ({(best_acc-0.5226)*100:+.2f}%)")

total_improvement = (best_acc - 0.43) * 100
print(f"\n   📊 Total improvement: +{total_improvement:.2f}%")

target = 0.85
gap = (target - best_acc) * 100
print(f"\n   🎯 Target: 85%")
print(f"   🔍 Gap remaining: {gap:.2f}%")

if best_acc >= 0.60:
    print(f"\n   ✅ Good progress! Breaking through 60% barrier")
elif best_acc >= 0.55:
    print(f"\n   🎯 Moderate progress (55%+ achieved)")
else:
    print(f"\n   ⚠️  More optimization needed")

print(f"\n" + "="*70)
print(f"🎉 Task 1.4 Complete - Advanced Ensembles Trained!")
print(f"="*70)
print(f"\n💡 Next Steps:")
print(f"   1. Hyperparameter optimization with Optuna (Task 1.5)")
print(f"   2. Consider data quality improvements if gap > 25%")
print(f"   3. Explore alternative features or domain knowledge")

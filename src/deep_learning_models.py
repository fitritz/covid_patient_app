"""
PulseAI - Deep Learning Models (Task 1.3)
Multi-Layer Perceptron and 1D CNN for breaking through 52% plateau
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PulseAI Deep Learning Models (Task 1.3)")
print("Breaking through the 52% plateau with Neural Networks")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

# Use 70/30 split (consistent with corrected_training.py)
print("\n🔀 Creating train-validation-test splits...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Further split training into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"   Training: {len(X_train)} samples")
print(f"   Validation: {len(X_val)} samples")
print(f"   Testing: {len(X_test)} samples")

# Scale features
print("\n⚖️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert labels to one-hot encoding for neural networks
y_train_cat = to_categorical(y_train, num_classes=3)
y_val_cat = to_categorical(y_val, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

print(f"   Input shape: {X_train_scaled.shape}")
print(f"   Output shape: {y_train_cat.shape}")

# Save scaler
joblib.dump(scaler, 'models/deep_learning_scaler.pkl')

# Model 1: Multi-Layer Perceptron (MLP) with Dropout
print("\n" + "="*70)
print("Model 1: Multi-Layer Perceptron (MLP)")
print("="*70)

mlp_model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(3, activation='softmax')
], name='MLP')

mlp_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 MLP Architecture:")
mlp_model.summary()

# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1
)

print("\n🔄 Training MLP...")
mlp_history = mlp_model.fit(
    X_train_scaled, y_train_cat,
    validation_data=(X_val_scaled, y_val_cat),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

# Evaluate MLP
mlp_val_loss, mlp_val_acc = mlp_model.evaluate(X_val_scaled, y_val_cat, verbose=0)
mlp_test_loss, mlp_test_acc = mlp_model.evaluate(X_test_scaled, y_test_cat, verbose=0)

print(f"\n   Validation Accuracy: {mlp_val_acc*100:.2f}%")
print(f"   Test Accuracy: {mlp_test_acc*100:.2f}%")
print(f"   Epochs trained: {len(mlp_history.history['loss'])}")

# Save MLP model
mlp_model.save('models/mlp_model.keras')
print(f"   💾 MLP model saved")

# Model 2: 1D Convolutional Neural Network
print("\n" + "="*70)
print("Model 2: 1D Convolutional Neural Network (1D-CNN)")
print("="*70)

# Reshape for 1D CNN (samples, timesteps, features)
X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_val_cnn = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

cnn_model = keras.Sequential([
    layers.Input(shape=(X_train_cnn.shape[1], 1)),
    
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    
    layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.4),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(3, activation='softmax')
], name='CNN_1D')

cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 1D-CNN Architecture:")
cnn_model.summary()

print("\n🔄 Training 1D-CNN...")
cnn_history = cnn_model.fit(
    X_train_cnn, y_train_cat,
    validation_data=(X_val_cnn, y_val_cat),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

# Evaluate CNN
cnn_val_loss, cnn_val_acc = cnn_model.evaluate(X_val_cnn, y_val_cat, verbose=0)
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)

print(f"\n   Validation Accuracy: {cnn_val_acc*100:.2f}%")
print(f"   Test Accuracy: {cnn_test_acc*100:.2f}%")
print(f"   Epochs trained: {len(cnn_history.history['loss'])}")

# Save CNN model
cnn_model.save('models/cnn_1d_model.keras')
print(f"   💾 1D-CNN model saved")

# Model 3: Deep MLP with More Layers
print("\n" + "="*70)
print("Model 3: Deep MLP (More Layers & Regularization)")
print("="*70)

deep_mlp_model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(3, activation='softmax')
], name='Deep_MLP')

deep_mlp_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Deep MLP Architecture:")
deep_mlp_model.summary()

print("\n🔄 Training Deep MLP...")
deep_mlp_history = deep_mlp_model.fit(
    X_train_scaled, y_train_cat,
    validation_data=(X_val_scaled, y_val_cat),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

# Evaluate Deep MLP
deep_mlp_val_loss, deep_mlp_val_acc = deep_mlp_model.evaluate(X_val_scaled, y_val_cat, verbose=0)
deep_mlp_test_loss, deep_mlp_test_acc = deep_mlp_model.evaluate(X_test_scaled, y_test_cat, verbose=0)

print(f"\n   Validation Accuracy: {deep_mlp_val_acc*100:.2f}%")
print(f"   Test Accuracy: {deep_mlp_test_acc*100:.2f}%")
print(f"   Epochs trained: {len(deep_mlp_history.history['loss'])}")

# Save Deep MLP model
deep_mlp_model.save('models/deep_mlp_model.keras')
print(f"   💾 Deep MLP model saved")

# Compare all models
print("\n" + "="*70)
print("📊 Model Comparison")
print("="*70)

results = {
    'MLP': {'val_acc': mlp_val_acc, 'test_acc': mlp_test_acc},
    '1D_CNN': {'val_acc': cnn_val_acc, 'test_acc': cnn_test_acc},
    'Deep_MLP': {'val_acc': deep_mlp_val_acc, 'test_acc': deep_mlp_test_acc}
}

print(f"\n{'Model':<15} {'Validation Acc':<18} {'Test Acc':<12}")
print("-"*50)
for name, scores in sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True):
    print(f"{name:<15} {scores['val_acc']*100:>15.2f}%  {scores['test_acc']*100:>10.2f}%")

# Find best model
best_model_name = max(results.items(), key=lambda x: x[1]['test_acc'])[0]
best_test_acc = results[best_model_name]['test_acc']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test Accuracy: {best_test_acc*100:.2f}%")

# Load best model for detailed evaluation
if best_model_name == 'MLP':
    best_model = mlp_model
    X_test_final = X_test_scaled
elif best_model_name == '1D_CNN':
    best_model = cnn_model
    X_test_final = X_test_cnn
else:
    best_model = deep_mlp_model
    X_test_final = X_test_scaled

# Detailed predictions
y_pred_proba = best_model.predict(X_test_final, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print(f"\n📋 Detailed Classification Report (Best Model):")
print("="*70)
print(classification_report(y_test, y_pred, 
                          target_names=['Low Risk', 'Medium Risk', 'High Risk'],
                          digits=3))

# Confusion matrix
print(f"\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"\n                 Predicted")
print(f"              Low  Med  High")
print(f"Actual Low    {cm[0,0]:>3}  {cm[0,1]:>3}  {cm[0,2]:>3}")
print(f"       Med    {cm[1,0]:>3}  {cm[1,1]:>3}  {cm[1,2]:>3}")
print(f"       High   {cm[2,0]:>3}  {cm[2,1]:>3}  {cm[2,2]:>3}")

# Per-class accuracy
print(f"\n📊 Per-Class Performance:")
for i, class_name in enumerate(['Low Risk', 'Medium Risk', 'High Risk']):
    class_acc = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"   {class_name}: {class_acc*100:.2f}%")

# Save comprehensive metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'task': 'Task 1.3 - Deep Learning Models',
    'dataset': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'features': len(feature_cols),
        'feature_names': feature_cols
    },
    'models': {
        'MLP': {
            'val_accuracy': float(mlp_val_acc),
            'test_accuracy': float(mlp_test_acc),
            'epochs': len(mlp_history.history['loss'])
        },
        '1D_CNN': {
            'val_accuracy': float(cnn_val_acc),
            'test_accuracy': float(cnn_test_acc),
            'epochs': len(cnn_history.history['loss'])
        },
        'Deep_MLP': {
            'val_accuracy': float(deep_mlp_val_acc),
            'test_accuracy': float(deep_mlp_test_acc),
            'epochs': len(deep_mlp_history.history['loss'])
        }
    },
    'best_model': best_model_name,
    'best_test_accuracy': float(best_test_acc),
    'confusion_matrix': cm.tolist()
}

with open('models/deep_learning_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"\n💾 Metadata saved")

# Progress comparison
print(f"\n" + "="*70)
print(f"📈 Overall Progress Summary")
print(f"="*70)

baseline_acc = 0.43
traditional_ml_acc = 0.5226
deep_learning_acc = best_test_acc

print(f"\n   Baseline (original 53 samples): ~43%")
print(f"   Traditional ML (Logistic Regression): {traditional_ml_acc*100:.2f}%")
print(f"   Deep Learning ({best_model_name}): {deep_learning_acc*100:.2f}%")

improvement_ml = traditional_ml_acc - baseline_acc
improvement_dl = deep_learning_acc - traditional_ml_acc
total_improvement = deep_learning_acc - baseline_acc

print(f"\n   Improvement from ML: +{improvement_ml*100:.2f}%")
print(f"   Improvement from DL: +{improvement_dl*100:.2f}%")
print(f"   Total improvement: +{total_improvement*100:.2f}%")

target = 0.85
gap = target - deep_learning_acc
print(f"\n   🎯 Target: 85%")
print(f"   🔍 Gap remaining: {gap*100:.2f}%")

if deep_learning_acc >= 0.70:
    print(f"\n   ✅ Excellent progress! Deep learning broke through 70% barrier!")
elif deep_learning_acc >= 0.60:
    print(f"\n   🎯 Good progress! Deep learning reached 60%+")
elif deep_learning_acc > traditional_ml_acc:
    print(f"\n   📈 Moderate improvement with deep learning")
else:
    print(f"\n   ⚠️  Deep learning didn't outperform traditional ML")

print(f"\n" + "="*70)
print(f"🎉 Task 1.3 Complete - Deep Learning Models Trained!")
print(f"="*70)
print(f"\n💡 Next Steps:")
print(f"   1. Advanced Ensemble (Task 1.4) - Stack DL + ML models")
print(f"   2. Hyperparameter optimization (Task 1.5)")
print(f"   3. Consider data quality improvements if gap > 15%")

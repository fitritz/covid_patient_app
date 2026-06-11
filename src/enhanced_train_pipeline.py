"""
PulseAI - Enhanced Training Pipeline
Retrains models with augmented data and engineered features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from data_preprocessing import DataPreprocessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

# Setup logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'enhanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Enhanced training pipeline with augmented data and engineered features"""
    
    print("="*70)
    print("PulseAI Enhanced Training Pipeline")
    print("Augmented Data + Engineered Features")
    print("="*70)
    
    try:
        # ========== STEP 1: Load Engineered Dataset ==========
        logger.info("Step 1: Loading engineered dataset...")
        df = pd.read_csv('dataset_engineered.csv')
        
        print(f"\n✅ Dataset loaded:")
        print(f"   - Total samples: {len(df)}")
        print(f"   - Total features: {len(df.columns) - 3}")  # Exclude Patient ID, Target, Augmentation_Method
        print(f"   - Augmentation methods: {df['Augmentation_Method'].unique()}")
        
        # Prepare data (remove non-feature columns)
        feature_cols = [col for col in df.columns 
                       if col not in ['Patient ID', 'Target', 'Augmentation_Method', 'Sl.No']]
        
        X = df[feature_cols].values
        y = df['Target'].values
        
        print(f"   - Feature matrix: {X.shape}")
        print(f"   - Target distribution: {np.bincount(y)}")
        
        # ========== STEP 2: Train-Test Split ==========
        logger.info("Step 2: Creating train-test split...")
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✅ Data split:")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Testing samples: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler and feature names
        import joblib
        joblib.dump(scaler, 'models/enhanced_scaler.pkl')
        joblib.dump(feature_cols, 'models/enhanced_feature_names.pkl')
        print(f"   - ✅ Scaler and feature names saved")
        
        # ========== STEP 3: Train Models ==========
        logger.info("Step 3: Training models with enhanced features...")
        print(f"\n{'='*70}")
        print(f"Training Models with {len(feature_cols)} Features")
        print(f"{'='*70}")
        
        trainer = ModelTrainer(random_state=42)
        
        # Convert to DataFrame format expected by ModelTrainer
        X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        
        # Train models with hyperparameter tuning
        results = trainer.train_with_tuning(
            X_train_df, y_train_series,
            cv_folds=5
        )
        
        # ========== STEP 4: Evaluate Models ==========
        logger.info("Step 4: Evaluating models...")
        print(f"\n{'='*70}")
        print(f"Model Evaluation on Test Set")
        print(f"{'='*70}")
        
        test_results = trainer.evaluate_models(X_test_df, y_test_series)
        
        # ========== STEP 5: Create Ensemble ==========
        logger.info("Step 5: Creating enhanced ensemble...")
        ensemble_model = trainer.create_ensemble_model(results, top_n=3)
        ensemble_model.fit(X_train_df, y_train_series)
        
        # Evaluate ensemble
        from sklearn.metrics import accuracy_score, classification_report
        y_pred_ensemble = ensemble_model.predict(X_test_df)
        ensemble_accuracy = accuracy_score(y_test_series, y_pred_ensemble)
        
        print(f"\n🎯 Enhanced Ensemble Performance:")
        print(f"   Accuracy: {ensemble_accuracy*100:.2f}%")
        print(f"\n{classification_report(y_test_series, y_pred_ensemble, target_names=['Low', 'Medium', 'High'])}")
        
        # Add ensemble to results
        results['Enhanced_Ensemble'] = {
            'model': ensemble_model,
            'accuracy': ensemble_accuracy,
            'best_params': 'Voting Classifier (Top 3 Models)'
        }
        
        # ========== STEP 6: Save Models ==========
        logger.info("Step 6: Saving enhanced models...")
        trainer.save_all_models(results, prefix='enhanced_')
        
        # Save best model
        best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model = results[best_model_name]['model']
        best_accuracy = results[best_model_name]['accuracy']
        
        joblib.dump(best_model, 'models/enhanced_best_model.pkl')
        
        # Save metadata
        metadata = {
            'best_model': best_model_name,
            'accuracy': float(best_accuracy),
            'features_used': len(feature_cols),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        with open('models/enhanced_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n💾 Models saved:")
        print(f"   - Best model: {best_model_name} ({best_accuracy*100:.2f}%)")
        print(f"   - Total models: {len(results)}")
        
        # ========== STEP 7: Generate Reports ==========
        logger.info("Step 7: Generating comprehensive reports...")
        
        evaluator = ModelEvaluator(output_dir='reports')
        
        # Confusion matrices
        evaluator.plot_confusion_matrices(
            results, y_test_series,
            class_names=['Low', 'Medium', 'High']
        )
        
        # Model comparison
        evaluator.plot_model_comparison(test_results)
        
        # Metrics radar
        evaluator.plot_metrics_radar(test_results)
        
        # Classification reports
        evaluator.generate_classification_reports(results)
        
        # Executive summary
        evaluator.generate_summary_report(test_results)
        
        print(f"\n📊 Reports generated:")
        print(f"   - Confusion matrices")
        print(f"   - Model comparison charts")
        print(f"   - Metrics radar chart")
        print(f"   - Classification reports")
        print(f"   - Executive summary")
        
        # ========== FINAL SUMMARY ==========
        print(f"\n{'='*70}")
        print(f"🎉 Enhanced Training Complete!")
        print(f"{'='*70}")
        
        print(f"\n📊 Training Summary:")
        print(f"   Dataset: 663 samples (augmented)")
        print(f"   Features: {len(feature_cols)} (engineered)")
        print(f"   Models trained: {len(results)}")
        print(f"   Best model: {best_model_name}")
        print(f"   Best accuracy: {best_accuracy*100:.2f}%")
        
        print(f"\n📈 Performance Comparison:")
        print(f"   Baseline (150 samples, 16 features): 72.73%")
        print(f"   Enhanced (663 samples, {len(feature_cols)} features): {best_accuracy*100:.2f}%")
        improvement = best_accuracy*100 - 72.73
        print(f"   Improvement: {improvement:+.2f}%")
        
        if best_accuracy >= 0.85:
            print(f"\n   🎯 ✅ Target achieved (85%+)!")
        else:
            print(f"\n   🎯 Target: 85%+ (current: {best_accuracy*100:.2f}%)")
        
        logger.info("Training pipeline completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    results = main()

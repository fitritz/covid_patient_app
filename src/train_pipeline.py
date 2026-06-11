"""
Main Training Pipeline for PulseAI
Orchestrates data preprocessing, model training, and evaluation
"""

import sys
import os
import logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

# Ensure critical directories exist before configuring logging
os.makedirs(LOG_DIR, exist_ok=True)

# Add src to path
sys.path.insert(0, BASE_DIR)

from data_preprocessing import DataPreprocessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main training pipeline
    """
    try:
        # Create output directories
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        logger.info("=" * 100)
        logger.info("PULSEAI - ML MODEL ENHANCEMENT PIPELINE")
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 100)
        
        # ===========================
        # STEP 1: DATA PREPROCESSING
        # ===========================
        logger.info("\n" + "=" * 100)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("=" * 100)
        
        preprocessor = DataPreprocessor(scaler_type='standard')
        
        X_train, X_test, y_train, y_test, feature_columns = preprocessor.get_preprocessed_data(
            filepath=os.path.join(PROJECT_ROOT, 'iot_dataset_expanded.csv'),
            engineer_features=True,
            test_size=0.2,
            random_state=42,
            scale_features=True
        )
        
        logger.info(f"\n✓ Data preprocessing complete!")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Testing samples:  {len(X_test)}")
        logger.info(f"  Total features:   {len(feature_columns)}")
        logger.info(f"  Feature names:    {feature_columns[:5]}... (showing first 5)")
        
        # ===========================
        # STEP 2: MODEL TRAINING
        # ===========================
        logger.info("\n" + "=" * 100)
        logger.info("STEP 2: MODEL TRAINING WITH HYPERPARAMETER TUNING")
        logger.info("=" * 100)
        
        trainer = ModelTrainer(random_state=42)
        
        # Train models with tuning
        trained_models = trainer.train_with_tuning(X_train, y_train, cv_folds=5)
        
        logger.info(f"\n✓ Model training complete!")
        logger.info(f"  Total models trained: {len(trained_models)}")
        
        # Create ensemble model
        ensemble = trainer.create_ensemble_model(X_train, y_train, top_n=3)
        
        # ===========================
        # STEP 3: MODEL EVALUATION
        # ===========================
        logger.info("\n" + "=" * 100)
        logger.info("STEP 3: MODEL EVALUATION")
        logger.info("=" * 100)
        
        results_df = trainer.evaluate_models(X_test, y_test)
        
        logger.info("\n✓ Model evaluation complete!")
        logger.info(f"\nFinal Results (Top 5 Models):")
        logger.info("\n" + results_df.head().to_string(index=False))
        
        # ===========================
        # STEP 4: VISUALIZATION & REPORTING
        # ===========================
        logger.info("\n" + "=" * 100)
        logger.info("STEP 4: GENERATING VISUALIZATIONS AND REPORTS")
        logger.info("=" * 100)
        
        evaluator = ModelEvaluator(output_dir=REPORTS_DIR)
        
        # Generate all visualizations
        evaluator.plot_confusion_matrices(trainer.results, y_test)
        evaluator.plot_model_comparison(results_df)
        evaluator.plot_metrics_radar(results_df)
        evaluator.generate_classification_reports(trainer.results)
        evaluator.generate_summary_report(results_df)
        
        logger.info("\n✓ Visualizations and reports generated!")
        
        # ===========================
        # STEP 5: SAVE MODELS
        # ===========================
        logger.info("\n" + "=" * 100)
        logger.info("STEP 5: SAVING MODELS")
        logger.info("=" * 100)
        
        # Save best model
        trainer.save_model(
            model_path=os.path.join(MODELS_DIR, 'best_model.pkl'),
            metadata_path=os.path.join(MODELS_DIR, 'model_metadata.json')
        )
        
        # Save all models
        trainer.save_all_models(output_dir=MODELS_DIR)
        
        logger.info("\n✓ Models saved successfully!")
        
        # ===========================
        # FINAL SUMMARY
        # ===========================
        logger.info("\n" + "=" * 100)
        logger.info("PIPELINE EXECUTION COMPLETE!")
        logger.info("=" * 100)
        
        logger.info(f"\n🏆 Best Model: {trainer.best_model_name}")
        logger.info(f"   Accuracy: {results_df.iloc[0]['Accuracy']:.4f} ({results_df.iloc[0]['Accuracy']*100:.2f}%)")
        logger.info(f"   F1 Score: {results_df.iloc[0]['F1_Score']:.4f}")
        
        logger.info("\n📁 Output Locations:")
        logger.info(f"   Models:   models/")
        logger.info(f"   Reports:  reports/")
        logger.info(f"   Logs:     logs/training.log")
        
        logger.info(f"\n✨ Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 100 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

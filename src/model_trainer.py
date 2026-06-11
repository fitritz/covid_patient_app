"""
Advanced Model Training Module for PulseAI
Implements multiple ML algorithms with hyperparameter tuning
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import xgboost as xgb
from typing import Dict, Any, Tuple, List
import logging
import joblib
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Professional model training pipeline with hyperparameter tuning
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def get_base_models(self) -> Dict[str, Any]:
        """
        Define base models to train
        
        Returns:
            Dictionary of model name and model object
        """
        models = {
            'Gaussian_Naive_Bayes': GaussianNB(),
            'Decision_Tree': DecisionTreeClassifier(random_state=self.random_state),
            'Logistic_Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'SVM': SVC(random_state=self.random_state, probability=True),
            'Random_Forest': RandomForestClassifier(random_state=self.random_state),
            'Gradient_Boosting': GradientBoostingClassifier(random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='mlogloss')
        }
        return models
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """
        Define hyperparameter grids for tuning
        
        Returns:
            Dictionary of model name and parameter grid
        """
        param_grids = {
            'Gaussian_Naive_Bayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            },
            'Decision_Tree': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },
            'Logistic_Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'Random_Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'Gradient_Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }
        return param_grids
    
    def train_with_tuning(self, 
                          X_train: pd.DataFrame, 
                          y_train: pd.Series,
                          cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train models with hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of trained models
        """
        logger.info("=" * 80)
        logger.info("Starting model training with hyperparameter tuning...")
        logger.info("=" * 80)
        
        base_models = self.get_base_models()
        param_grids = self.get_hyperparameter_grids()
        
        for model_name, model in base_models.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training: {model_name}")
            logger.info(f"{'=' * 60}")
            
            try:
                if model_name in param_grids:
                    # Hyperparameter tuning with GridSearchCV
                    logger.info(f"Performing hyperparameter tuning...")
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grids[model_name],
                        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    
                    # Get best model
                    best_model = grid_search.best_estimator_
                    logger.info(f"Best parameters: {grid_search.best_params_}")
                    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
                    
                    self.models[model_name] = {
                        'model': best_model,
                        'best_params': grid_search.best_params_,
                        'cv_score': grid_search.best_score_
                    }
                else:
                    # Train without tuning
                    model.fit(X_train, y_train)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                    
                    self.models[model_name] = {
                        'model': model,
                        'best_params': {},
                        'cv_score': cv_scores.mean()
                    }
                    logger.info(f"CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        logger.info("\n" + "=" * 80)
        logger.info("Model training complete!")
        logger.info("=" * 80)
        
        return self.models
    
    def evaluate_models(self, 
                        X_test: pd.DataFrame, 
                        y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame with evaluation metrics
        """
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating models on test set...")
        logger.info("=" * 80)
        
        results_list = []
        
        for model_name, model_info in self.models.items():
            logger.info(f"\nEvaluating: {model_name}")
            
            model = model_info['model']
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results_list.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'CV_Score': model_info['cv_score']
            })
            
            logger.info(f"Accuracy:  {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall:    {recall:.4f}")
            logger.info(f"F1 Score:  {f1:.4f}")
            
            # Store detailed results
            self.results[model_name] = {
                'predictions': y_pred,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_score': model_info['cv_score']
                },
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        # Identify best model
        best_idx = results_df['Accuracy'].idxmax()
        self.best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]['model']
        
        logger.info("\n" + "=" * 80)
        logger.info(f"BEST MODEL: {self.best_model_name}")
        logger.info(f"Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
        logger.info("=" * 80)
        
        return results_df
    
    def create_ensemble_model(self, 
                              X_train: pd.DataFrame, 
                              y_train: pd.Series,
                              top_n: int = 3) -> Any:
        """
        Create ensemble model from top performing models
        
        Args:
            X_train: Training features
            y_train: Training target
            top_n: Number of top models to include in ensemble
            
        Returns:
            Trained ensemble model
        """
        logger.info(f"\nCreating ensemble model from top {top_n} models...")
        
        # Get top models by CV score
        sorted_models = sorted(
            self.models.items(), 
            key=lambda x: x[1]['cv_score'], 
            reverse=True
        )[:top_n]
        
        estimators = [(name, model_info['model']) for name, model_info in sorted_models]
        
        logger.info(f"Ensemble members: {[name for name, _ in estimators]}")
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        ensemble.fit(X_train, y_train)
        
        self.models['Ensemble_Voting'] = {
            'model': ensemble,
            'best_params': {'members': [name for name, _ in estimators]},
            'cv_score': np.mean([info['cv_score'] for _, info in sorted_models])
        }
        
        logger.info("Ensemble model created successfully!")
        
        return ensemble
    
    def save_model(self, model_path: str, metadata_path: str = None):
        """
        Save best model and metadata
        
        Args:
            model_path: Path to save model file
            metadata_path: Path to save metadata JSON
        """
        if self.best_model is None:
            logger.error("No trained model to save. Train models first.")
            return
        
        logger.info(f"\nSaving best model: {self.best_model_name}")
        
        # Save model
        joblib.dump(self.best_model, model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Save metadata
        if metadata_path:
            metadata = {
                'model_name': self.best_model_name,
                'timestamp': datetime.now().isoformat(),
                'best_params': self.models[self.best_model_name]['best_params'],
                'metrics': self.results[self.best_model_name]['metrics'],
                'classification_report': self.results[self.best_model_name]['classification_report']
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Metadata saved to: {metadata_path}")
    
    def save_all_models(self, output_dir: str):
        """
        Save all trained models
        
        Args:
            output_dir: Directory to save models
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"\nSaving all models to: {output_dir}")
        
        for model_name, model_info in self.models.items():
            model_path = os.path.join(output_dir, f"{model_name}.pkl")
            joblib.dump(model_info['model'], model_path)
            logger.info(f"Saved: {model_name}")
        
        logger.info("All models saved successfully!")


if __name__ == "__main__":
    # Test the trainer
    logger.info("Model trainer module loaded successfully!")

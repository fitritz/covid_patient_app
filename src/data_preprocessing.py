"""
Data Preprocessing Module for PulseAI
Handles data loading, cleaning, feature engineering, and splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Professional data preprocessing pipeline for PulseAI
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize preprocessor
        
        Args:
            scaler_type: Type of scaling ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.feature_columns = None
        self.target_column = 'Target'
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {filepath}")
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        # Remove 'Sl.No' if exists (not needed for modeling)
        if 'Sl.No' in df.columns:
            df = df.drop('Sl.No', axis=1)
            logger.info("Dropped 'Sl.No' column")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
            df = df.dropna()
            logger.info(f"Dropped rows with missing values. New shape: {df.shape}")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
            df = df.drop_duplicates()
            logger.info(f"Removed duplicates. New shape: {df.shape}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features to improve model performance
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Engineering features...")
        
        df = df.copy()
        
        # Temperature-related features
        if 'Temperature Data' in df.columns:
            df['Temp_Squared'] = df['Temperature Data'] ** 2
            df['Temp_IsNormal'] = ((df['Temperature Data'] >= 36) & 
                                    (df['Temperature Data'] <= 37.5)).astype(int)
        
        # ECG-related features
        if 'ECG Data' in df.columns:
            df['ECG_Squared'] = df['ECG Data'] ** 2
            df['ECG_IsZero'] = (df['ECG Data'] == 0).astype(int)
            df['ECG_High'] = (df['ECG Data'] > 20).astype(int)
        
        # Pressure-related features
        if 'Pressure Data' in df.columns:
            df['Pressure_Squared'] = df['Pressure Data'] ** 2
            df['Pressure_IsNormal'] = ((df['Pressure Data'] >= 60) & 
                                        (df['Pressure Data'] <= 80)).astype(int)
        
        # Interaction features
        if all(col in df.columns for col in ['Temperature Data', 'ECG Data']):
            df['Temp_ECG_Interaction'] = df['Temperature Data'] * df['ECG Data']
        
        if all(col in df.columns for col in ['ECG Data', 'Pressure Data']):
            df['ECG_Pressure_Interaction'] = df['ECG Data'] * df['Pressure Data']
        
        if all(col in df.columns for col in ['Temperature Data', 'Pressure Data']):
            df['Temp_Pressure_Interaction'] = df['Temperature Data'] * df['Pressure Data']
        
        # Combined risk indicator
        if all(col in df.columns for col in ['Temperature Data', 'ECG Data', 'Pressure Data']):
            df['Vital_Signs_Sum'] = (df['Temperature Data'] + 
                                      df['ECG Data'] + 
                                      df['Pressure Data'])
            df['Vital_Signs_Mean'] = (df['Temperature Data'] + 
                                       df['ECG Data'] + 
                                       df['Pressure Data']) / 3
        
        logger.info(f"Feature engineering complete. New shape: {df.shape}")
        return df
    
    def prepare_data(self, 
                     df: pd.DataFrame, 
                     test_size: float = 0.2, 
                     random_state: int = 42,
                     scale_features: bool = True) -> Tuple:
        """
        Prepare data for modeling: split and scale
        
        Args:
            df: Input DataFrame
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            scale_features: Whether to scale features
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_columns)
        """
        logger.info("Preparing data for modeling...")
        
        # Separate features and target
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        
        self.feature_columns = X.columns.tolist()
        logger.info(f"Features: {self.feature_columns}")
        logger.info(f"Target distribution:\n{y.value_counts().sort_index()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        logger.info(f"Train set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Scale features
        if scale_features:
            logger.info(f"Scaling features using {self.scaler_type} scaler")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Convert back to DataFrame to preserve column names
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        return X_train, X_test, y_train, y_test, self.feature_columns
    
    def get_preprocessed_data(self, 
                              filepath: str, 
                              engineer_features: bool = True,
                              test_size: float = 0.2,
                              random_state: int = 42,
                              scale_features: bool = True) -> Tuple:
        """
        Complete preprocessing pipeline
        
        Args:
            filepath: Path to CSV file
            engineer_features: Whether to create engineered features
            test_size: Proportion of test set
            random_state: Random seed
            scale_features: Whether to scale features
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_columns)
        """
        # Load and clean data
        df = self.load_data(filepath)
        df = self.clean_data(df)
        
        # Engineer features if requested
        if engineer_features:
            df = self.engineer_features(df)
        
        # Prepare data
        return self.prepare_data(df, test_size, random_state, scale_features)


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor(scaler_type='standard')
    X_train, X_test, y_train, y_test, features = preprocessor.get_preprocessed_data(
    filepath="../iot_dataset_expanded.csv",
        engineer_features=True
    )
    print(f"\nPreprocessing complete!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {len(features)}")

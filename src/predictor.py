"""
Model Inference Module for PulseAI
Handles loading trained models and making predictions
"""

import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PulseAIPredictor:
    """
    Production-ready model inference class
    """
    
    def __init__(self, model_path: str, metadata_path: str = None):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model file
            metadata_path: Path to model metadata JSON
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.class_labels = {0: "Low", 1: "Medium", 2: "High"}
        
        self.load_model()
        
    def load_model(self):
        """
        Load trained model and metadata
        """
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info("✓ Model loaded successfully!")
            
            if self.metadata_path and os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✓ Metadata loaded: {self.metadata['model_name']}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_input(self, data: Union[Dict, List, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess input data for prediction with feature engineering
        
        Args:
            data: Input data (dict, list, or DataFrame)
            
        Returns:
            Preprocessed DataFrame with engineered features
        """
        if isinstance(data, dict):
            # Single prediction from dict
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            # Single or multiple predictions from list
            if isinstance(data[0], dict):
                df = pd.DataFrame(data)
            else:
                # Assume it's a single sample as list
                df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input must be dict, list, or DataFrame")
        
        # Ensure required columns exist
        required_cols = ['Patient ID', 'Temperature Data', 'ECG Data', 'Pressure Data']
        
        # If column names don't match, try to map them
        if not all(col in df.columns for col in required_cols):
            # Try to use column positions
            if len(df.columns) >= 4:
                df.columns = required_cols[:len(df.columns)]
        
        # Apply feature engineering (same as training)
        df = self.engineer_features(df)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features (must match training pipeline)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
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
        
        return df
    
    def predict(self, data: Union[Dict, List, pd.DataFrame]) -> Dict[str, Any]:
        """
        Make prediction on input data
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess input
            df = self.preprocess_input(data)
            
            # Make prediction
            prediction = self.model.predict(df)
            
            # Get probability scores if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(df)
            
            # Format results
            results = {
                'success': True,
                'predictions': [
                    {
                        'risk_level_code': int(pred),
                        'risk_level': self.class_labels.get(int(pred), 'Unknown'),
                        'probabilities': {
                            'Low': float(prob[0]) if probabilities is not None else None,
                            'Medium': float(prob[1]) if probabilities is not None else None,
                            'High': float(prob[2]) if probabilities is not None else None
                        } if probabilities is not None else None,
                        'confidence': float(max(prob)) if probabilities is not None else None
                    }
                    for pred, prob in zip(prediction, probabilities if probabilities is not None else [None]*len(prediction))
                ]
            }
            
            # Add model info if metadata available
            if self.metadata:
                results['model_info'] = {
                    'model_name': self.metadata.get('model_name'),
                    'model_accuracy': self.metadata.get('metrics', {}).get('accuracy'),
                    'timestamp': self.metadata.get('timestamp')
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_single(self, 
                       patient_id: int,
                       temperature: float,
                       ecg: float,
                       pressure: float) -> Dict[str, Any]:
        """
        Convenience method for single prediction
        
        Args:
            patient_id: Patient ID
            temperature: Temperature reading
            ecg: ECG reading
            pressure: Blood pressure reading
            
        Returns:
            Prediction result dictionary
        """
        data = {
            'Patient ID': patient_id,
            'Temperature Data': temperature,
            'ECG Data': ecg,
            'Pressure Data': pressure
        }
        
        result = self.predict(data)
        
        # Return single prediction
        if result['success'] and result['predictions']:
            return result['predictions'][0]
        else:
            return result
    
    def get_risk_assessment(self, prediction_result: Dict[str, Any]) -> str:
        """
        Generate human-readable risk assessment
        
        Args:
            prediction_result: Result from predict_single()
            
        Returns:
            Risk assessment message
        """
        if not prediction_result.get('risk_level'):
            return "Unable to assess risk"
        
        risk_level = prediction_result['risk_level']
        confidence = prediction_result.get('confidence', 0)
        
        assessment = f"🏥 Patient Risk Assessment: {risk_level}\n"
        
        if confidence:
            assessment += f"   Confidence: {confidence*100:.1f}%\n"
        
        if risk_level == "Low":
            assessment += "   ✅ Patient condition appears stable.\n"
            assessment += "   Recommendation: Continue routine monitoring."
        elif risk_level == "Medium":
            assessment += "   ⚠️  Patient condition requires attention.\n"
            assessment += "   Recommendation: Increased monitoring and consultation."
        else:  # High
            assessment += "   🚨 Patient condition is critical!\n"
            assessment += "   Recommendation: Immediate medical intervention required."
        
        return assessment


# Example usage and testing
if __name__ == "__main__":
    # This will be used after training
    print("PulseAI Predictor Module")
    print("=" * 60)
    print("\nUsage Example:")
    print("""
    predictor = PulseAIPredictor(
        model_path='../models/best_model.pkl',
        metadata_path='../models/model_metadata.json'
    )
    
    result = predictor.predict_single(
        patient_id=1,
        temperature=36.5,
        ecg=85,
        pressure=120
    )
    
    print(predictor.get_risk_assessment(result))
    """)

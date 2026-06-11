"""
Demo script to test PulseAI model with sample predictions
"""

import sys
import os
# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from advanced_features import AdvancedFeatureEngineer
from predictor import PulseAIPredictor
import joblib

def load_best_model():
    """Initialize a production-ready predictor using the saved model"""
    model_path = 'models/best_gradient_boosting_final.pkl'
    metadata_path = 'models/model_metadata.json'

    print(f"Loading model from {model_path}...")
    predictor = PulseAIPredictor(model_path=model_path, metadata_path=metadata_path)
    print("✓ Predictor initialized successfully!")

    return predictor

def make_prediction(predictor, feature_engineer, temperature, ecg, pressure):
    """Make a prediction using the `PulseAIPredictor` wrapper"""
    # Build a DataFrame matching the features the model was trained on
    # Metadata indicates features: Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate
    # Map available inputs: temperature -> BodyTemp, ecg -> HeartRate, pressure -> SystolicBP/DiastolicBP
    df_input = pd.DataFrame([{
        'Age': 30,  # default placeholder age
        'SystolicBP': pressure,
        'DiastolicBP': pressure,
        'BS': 7.0,  # default blood sugar placeholder
        'BodyTemp': temperature,
        'HeartRate': ecg
    }])

    model = predictor.model

    try:
        pred = model.predict(df_input)[0]
    except Exception as e:
        # Return error-like response
        return f"Prediction error: {e}", None, None

    probabilities = None
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(df_input)[0]
        probabilities = [prob[0], prob[1] if len(prob) > 1 else None, prob[2] if len(prob) > 2 else None]

    # Map numeric codes or labels to emoji text
    # Some models use numeric labels (0,1,2) while metadata used strings; handle both
    emoji_map = {0: "Low Risk 🟢", 1: "Medium Risk 🟡", 2: "High Risk 🔴",
                 'low': "Low Risk 🟢", 'Low': "Low Risk 🟢",
                 'medium': "Medium Risk 🟡", 'Medium': "Medium Risk 🟡",
                 'high': "High Risk 🔴", 'High': "High Risk 🔴"}

    label = pred
    # If the model used string labels, try to normalize
    if isinstance(pred, str):
        norm = pred.capitalize()
        display = emoji_map.get(norm, norm)
    else:
        display = emoji_map.get(pred, str(pred))

    return display, probabilities, pred

def display_sample_data():
    """Display some sample data from the dataset"""
    print("\n📊 Sample Data from Dataset:")
    print("=" * 60)
    
    # Try to load the dataset
    if os.path.exists('dataset.csv'):
        df = pd.read_csv('dataset.csv')
        print(df.head(10).to_string(index=False))
        print(f"\nDataset shape: {df.shape}")
        print(f"Features: {', '.join(df.columns.tolist())}")
    else:
        print("Dataset not found.")

def main():
    print("=" * 60)
    print("🏥 PulseAI - IoT Health Monitoring System Demo")
    print("=" * 60)
    
    # Initialize predictor
    predictor = load_best_model()

    # Initialize feature engineer (used for inference pipeline)
    feature_engineer = AdvancedFeatureEngineer()
    
    # Display sample data
    display_sample_data()
    
    print("\n" + "=" * 60)
    print("🔮 Making Sample Predictions")
    print("=" * 60)
    
    # Test cases with different vital signs
    test_cases = [
        {"name": "Patient A (Normal)", "temp": 36.5, "ecg": 75, "pressure": 120},
        {"name": "Patient B (Mild Alert)", "temp": 37.8, "ecg": 95, "pressure": 140},
        {"name": "Patient C (High Alert)", "temp": 39.2, "ecg": 120, "pressure": 170},
        {"name": "Patient D (Low Vitals)", "temp": 35.5, "ecg": 55, "pressure": 90},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Temperature: {case['temp']}°C")
        print(f"   ECG: {case['ecg']} bpm")
        print(f"   Pressure: {case['pressure']} mmHg")
        
        risk_level, probabilities, pred_class = make_prediction(
            predictor, feature_engineer,
            case['temp'], case['ecg'], case['pressure']
        )
        
        print(f"   → Prediction: {risk_level}")
        print(f"   → Confidence: Low={probabilities[0]:.1%}, Med={probabilities[1]:.1%}, High={probabilities[2]:.1%}")
    
    print("\n" + "=" * 60)
    print("✨ Interactive Prediction Mode")
    print("=" * 60)
    print("Enter patient vitals for real-time prediction (or 'q' to quit)")
    
    while True:
        print("\n")
        temp_input = input("Enter Temperature (°C) [or 'q' to quit]: ")
        if temp_input.lower() == 'q':
            break
            
        try:
            temperature = float(temp_input)
            ecg = float(input("Enter ECG (bpm): "))
            pressure = float(input("Enter Blood Pressure (mmHg): "))
            
            risk_level, probabilities, pred_class = make_prediction(
                predictor, feature_engineer, temperature, ecg, pressure
            )
            
            print(f"\n🏥 Diagnosis: {risk_level}")
            print(f"📊 Confidence Breakdown:")
            print(f"   - Low Risk:    {probabilities[0]:.1%}")
            print(f"   - Medium Risk: {probabilities[1]:.1%}")
            print(f"   - High Risk:   {probabilities[2]:.1%}")
            
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Thank you for using PulseAI!")
    print("=" * 60)

if __name__ == "__main__":
    main()

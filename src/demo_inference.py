"""
PulseAI - Model Inference Demo
Demonstrates how to use the trained model for predictions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import PulseAIPredictor
import pandas as pd

def main():
    print("=" * 80)
    print("PULSEAI - MODEL INFERENCE DEMO")
    print("=" * 80)
    print()
    
    # Load the trained model
    print("Loading trained model...")
    predictor = PulseAIPredictor(
        model_path='../models/best_model.pkl',
        metadata_path='../models/model_metadata.json'
    )
    print()
    
    # Demo 1: Single patient prediction
    print("=" * 80)
    print("DEMO 1: Single Patient Prediction")
    print("=" * 80)
    print()
    
    print("Patient Data:")
    print("  Patient ID: 1")
    print("  Temperature: 36.5°C")
    print("  ECG: 85 bpm")
    print("  Pressure: 120 mmHg")
    print()
    
    result = predictor.predict_single(
        patient_id=1,
        temperature=36.5,
        ecg=85,
        pressure=120
    )
    
    print(predictor.get_risk_assessment(result))
    print()
    
    if result.get('probabilities'):
        print("Risk Probabilities:")
        for risk, prob in result['probabilities'].items():
            if prob is not None:
                print(f"  {risk}: {prob*100:.1f}%")
    print()
    
    # Demo 2: High-risk patient
    print("=" * 80)
    print("DEMO 2: High-Risk Patient")
    print("=" * 80)
    print()
    
    print("Patient Data:")
    print("  Patient ID: 2")
    print("  Temperature: 32.0°C")
    print("  ECG: 23 bpm")
    print("  Pressure: 77 mmHg")
    print()
    
    result = predictor.predict_single(
        patient_id=2,
        temperature=32.0,
        ecg=23,
        pressure=77
    )
    
    print(predictor.get_risk_assessment(result))
    print()
    
    if result.get('probabilities'):
        print("Risk Probabilities:")
        for risk, prob in result['probabilities'].items():
            if prob is not None:
                print(f"  {risk}: {prob*100:.1f}%")
    print()
    
    # Demo 3: Batch predictions
    print("=" * 80)
    print("DEMO 3: Batch Predictions")
    print("=" * 80)
    print()
    
    # Multiple patients
    patients = [
        {'Patient ID': 1, 'Temperature Data': 36.5, 'ECG Data': 85, 'Pressure Data': 120},
        {'Patient ID': 2, 'Temperature Data': 32.0, 'ECG Data': 0, 'Pressure Data': 77},
        {'Patient ID': 3, 'Temperature Data': 32.0, 'ECG Data': 16, 'Pressure Data': 77},
        {'Patient ID': 4, 'Temperature Data': 37.0, 'ECG Data': 90, 'Pressure Data': 80},
        {'Patient ID': 5, 'Temperature Data': 32.0, 'ECG Data': 23, 'Pressure Data': 77},
    ]
    
    results = predictor.predict(patients)
    
    print("Batch Prediction Results:")
    print("-" * 80)
    
    if results['success']:
        for i, (patient, prediction) in enumerate(zip(patients, results['predictions']), 1):
            print(f"\nPatient {patient['Patient ID']}:")
            print(f"  Vitals: Temp={patient['Temperature Data']}, ECG={patient['ECG Data']}, BP={patient['Pressure Data']}")
            print(f"  Risk Level: {prediction['risk_level']}")
            if prediction.get('confidence'):
                print(f"  Confidence: {prediction['confidence']*100:.1f}%")
    
    print()
    print("=" * 80)
    print()
    
    # Display model information
    if predictor.metadata:
        print("MODEL INFORMATION:")
        print(f"  Model Name: {predictor.metadata.get('model_name')}")
        print(f"  Accuracy: {predictor.metadata.get('metrics', {}).get('accuracy', 0)*100:.2f}%")
        print(f"  F1 Score: {predictor.metadata.get('metrics', {}).get('f1_score', 0):.4f}")
        print(f"  Training Date: {predictor.metadata.get('timestamp')}")
    
    print()
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

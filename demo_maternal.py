"""
PulseAI - Maternal Health Risk Prediction Demo
Uses the tuned Gradient Boosting model with 87% high-risk recall
"""

import joblib
import numpy as np
import pandas as pd
import os
import json

def load_final_model():
    """Load the final XGBoost model (default parameters, not tuned)"""
    model_path = 'models/best_xgboost_final.pkl'
    scaler_path = 'models/best_scaler_final.pkl'
    
    # Feature names are fixed for this model
    feature_names = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
    
    print("Loading final model (XGBoost with default parameters)...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print("✓ Model loaded successfully!")
    print(f"✓ Features: {', '.join(feature_names)}")
    print(f"✓ Model Performance: 83.3% accuracy, 87% high-risk recall")
    print("✓ Note: Default XGBoost outperforms tuned version")
    
    return model, scaler, feature_names

def make_prediction(model, scaler, age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate):
    """Make a prediction for maternal health risk"""
    # Create input array with the 6 features
    input_data = np.array([[age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate]])
    
    # Scale the data
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0]
    
    # Map prediction to risk level
    risk_levels = {0: "High Risk 🔴", 1: "Low Risk 🟢", 2: "Medium Risk 🟡"}
    
    return risk_levels[prediction], probability, prediction

def display_sample_data():
    """Display some sample data from the maternal health dataset"""
    print("\n📊 Sample Data from Maternal Health Risk Dataset:")
    print("=" * 80)
    
    if os.path.exists('maternal_health_risk.csv'):
        df = pd.read_csv('maternal_health_risk.csv')
        print(df.head(10).to_string(index=False))
        print(f"\nDataset shape: {df.shape}")
        print(f"\nRisk Distribution:")
        if 'RiskLevel' in df.columns:
            print(df['RiskLevel'].value_counts().to_string())
    else:
        print("Dataset file not found.")

def main():
    print("=" * 80)
    print("🏥 PulseAI - Maternal Health Risk Prediction System")
    print("=" * 80)
    print("\nThis system predicts maternal health risk as Low, Medium, or High")
    print("based on 6 vital health indicators.")
    print("\n⭐ Model Highlights:")
    print("   - 83.3% Overall Accuracy")
    print("   - 87% High-Risk Recall (catches 87% of high-risk cases)")
    print("   - Uses default XGBoost (tuning made it worse!)")
    print("   - Optimized for patient safety (minimizes false negatives)")
    
    # Load model
    model, scaler, feature_names = load_final_model()
    
    # Display sample data
    display_sample_data()
    
    print("\n" + "=" * 80)
    print("🔮 Making Sample Predictions")
    print("=" * 80)
    
    # Test cases with different health profiles
    test_cases = [
        {
            "name": "Patient A (Healthy Profile)",
            "age": 25, "systolic": 115, "diastolic": 75, 
            "bs": 7.5, "temp": 98.2, "hr": 75
        },
        {
            "name": "Patient B (Mild Concerns)",
            "age": 35, "systolic": 130, "diastolic": 85, 
            "bs": 8.5, "temp": 98.8, "hr": 85
        },
        {
            "name": "Patient C (Multiple Risk Factors)",
            "age": 42, "systolic": 145, "diastolic": 95, 
            "bs": 10.5, "temp": 99.5, "hr": 95
        },
        {
            "name": "Patient D (High Risk Profile)",
            "age": 45, "systolic": 160, "diastolic": 100, 
            "bs": 12.0, "temp": 100.0, "hr": 105
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Age: {case['age']} years")
        print(f"   Blood Pressure: {case['systolic']}/{case['diastolic']} mmHg")
        print(f"   Blood Sugar: {case['bs']} mmol/L")
        print(f"   Body Temperature: {case['temp']}°F")
        print(f"   Heart Rate: {case['hr']} bpm")
        
        risk_level, probabilities, pred_class = make_prediction(
            model, scaler, 
            case['age'], case['systolic'], case['diastolic'],
            case['bs'], case['temp'], case['hr']
        )
        
        print(f"   → Prediction: {risk_level}")
        print(f"   → Confidence: High={probabilities[0]:.1%}, Low={probabilities[1]:.1%}, Med={probabilities[2]:.1%}")
    
    print("\n" + "=" * 80)
    print("✨ Interactive Prediction Mode")
    print("=" * 80)
    print("Enter patient health indicators for real-time risk assessment (or 'q' to quit)\n")
    print("Typical ranges:")
    print("  - Age: 10-70 years")
    print("  - Systolic BP: 70-160 mmHg")
    print("  - Diastolic BP: 49-100 mmHg")
    print("  - Blood Sugar: 6.0-19.0 mmol/L")
    print("  - Body Temperature: 98.0-103.0°F")
    print("  - Heart Rate: 7-90 bpm")
    
    while True:
        print("\n" + "-" * 80)
        age_input = input("Enter Age (years) [or 'q' to quit]: ")
        if age_input.lower() == 'q':
            break
            
        try:
            age = float(age_input)
            systolic_bp = float(input("Enter Systolic Blood Pressure (mmHg): "))
            diastolic_bp = float(input("Enter Diastolic Blood Pressure (mmHg): "))
            blood_sugar = float(input("Enter Blood Sugar (mmol/L): "))
            body_temp = float(input("Enter Body Temperature (°F): "))
            heart_rate = float(input("Enter Heart Rate (bpm): "))
            
            risk_level, probabilities, pred_class = make_prediction(
                model, scaler, age, systolic_bp, diastolic_bp, 
                blood_sugar, body_temp, heart_rate
            )
            
            print(f"\n🏥 MATERNAL HEALTH RISK ASSESSMENT")
            print(f"{'=' * 50}")
            print(f"Risk Level: {risk_level}")
            print(f"\n📊 Confidence Breakdown:")
            print(f"   - High Risk:   {probabilities[0]:.1%}")
            print(f"   - Low Risk:    {probabilities[1]:.1%}")
            print(f"   - Medium Risk: {probabilities[2]:.1%}")
            
            if pred_class == 0:  # High Risk
                print(f"\n⚠️  IMMEDIATE MEDICAL ATTENTION RECOMMENDED")
            elif pred_class == 2:  # Medium Risk
                print(f"\n⚡ Regular monitoring and follow-up advised")
            else:  # Low Risk
                print(f"\n✅ Continue routine prenatal care")
            
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Thank you for using PulseAI Maternal Health Risk Prediction!")
    print("=" * 80)

if __name__ == "__main__":
    main()

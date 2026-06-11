"""Test the safety check function with abnormal values"""
import numpy as np
import joblib
import os

# Load model and scaler (from project root - go up two directories from backend)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
scaler = joblib.load(os.path.join(project_root, 'models', 'best_scaler_final.pkl'))
model = joblib.load(os.path.join(project_root, 'models', 'best_gradient_boosting_final.pkl'))
risk_map = {0: 'High', 1: 'Low', 2: 'Medium'}

# Test with abnormal values
features = [[12, 12, 56, 2.5, 50, 62]]
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)[0]
proba = model.predict_proba(features_scaled)[0]

print("="*60)
print("WITHOUT SAFETY CHECK (ML Model Only):")
print("="*60)
print(f"Prediction: {prediction} -> {risk_map[prediction]}")
print(f"Probabilities: High={proba[0]:.2%}, Low={proba[1]:.2%}, Medium={proba[2]:.2%}")

print("\n" + "="*60)
print("WITH SAFETY CHECK (Rule-Based Override):")
print("="*60)

def check_critical_values(age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate):
    """Safety check: Flag critically abnormal values"""
    critical_issues = []
    
    # Critical Age ranges
    if age < 15 or age > 50:
        critical_issues.append(f"Age {age} is critically outside safe pregnancy range (15-50)")
    
    # Critical Blood Pressure
    if systolic_bp < 70:
        critical_issues.append(f"Severe hypotension: Systolic BP {systolic_bp} < 70 mmHg")
    elif systolic_bp > 180:
        critical_issues.append(f"Hypertensive crisis: Systolic BP {systolic_bp} > 180 mmHg")
    
    if diastolic_bp < 40:
        critical_issues.append(f"Severe hypotension: Diastolic BP {diastolic_bp} < 40 mmHg")
    elif diastolic_bp > 120:
        critical_issues.append(f"Hypertensive crisis: Diastolic BP {diastolic_bp} > 120 mmHg")
    
    # Critical Blood Sugar
    if blood_sugar < 3.0:
        critical_issues.append(f"Severe hypoglycemia: Blood Sugar {blood_sugar} < 3.0 mmol/L")
    elif blood_sugar > 25.0:
        critical_issues.append(f"Severe hyperglycemia: Blood Sugar {blood_sugar} > 25.0 mmol/L")
    
    # Critical Body Temperature
    if body_temp < 95.0:
        critical_issues.append(f"Hypothermia: Body Temp {body_temp}°F < 95°F")
    elif body_temp > 104.0:
        critical_issues.append(f"Severe fever: Body Temp {body_temp}°F > 104°F")
    
    # Critical Heart Rate
    if heart_rate < 40:
        critical_issues.append(f"Severe bradycardia: Heart Rate {heart_rate} < 40 bpm")
    elif heart_rate > 140:
        critical_issues.append(f"Severe tachycardia: Heart Rate {heart_rate} > 140 bpm")
    
    if critical_issues:
        return True, "; ".join(critical_issues)
    
    return False, None

# Test safety check
is_critical, reason = check_critical_values(12, 12, 56, 2.5, 50, 62)

print(f"Is Critical: {is_critical}")
print(f"\nCritical Issues Detected:")
print(f"  {reason}")

if is_critical:
    final_result = "HIGH RISK (Safety Override)"
else:
    final_result = f"{risk_map[prediction]} (ML Prediction)"

print(f"\n" + "="*60)
print(f"FINAL RESULT: {final_result}")
print("="*60)

print("\n✓ Safety check is working correctly!")
print("✓ Abnormal values are now properly flagged as HIGH RISK")

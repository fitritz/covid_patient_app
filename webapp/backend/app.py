from flask import Flask, request, jsonify, send_from_directory
from flask_pymongo import PyMongo
from flask_cors import CORS
from bson.objectid import ObjectId
import joblib
import numpy as np
import pandas as pd
import datetime
import json
import os
import logging
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Logging Configuration ---
# Ensure log file is created in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, 'predict.log')
frontend_build_dir = os.path.abspath(os.path.join(script_dir, '..', 'frontend', 'build'))

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s',
                    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])

app = Flask(__name__, static_folder=frontend_build_dir, static_url_path='')

# --- CORS Configuration for Production ---
# Allow requests from Vercel frontend
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://pulseai-frontend.vercel.app",
            frontend_url
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# --- Database Configuration ---
# IMPORTANT: Set MONGO_URI environment variable in production
mongo_uri = os.getenv("MONGO_URI")

# Development convenience: if no MONGO_URI provided, fall back to an in-memory mongomock
if not mongo_uri:
    try:
        import mongomock
        import pymongo

        # Monkey-patch pymongo MongoClient to use mongomock for an in-memory DB
        pymongo.MongoClient = mongomock.MongoClient
        app.config["MONGO_URI"] = "mongodb://localhost:27017/pulseai_dev"
        mongo = PyMongo(app)
        logging.warning("MONGO_URI not set — using in-memory mongomock for development.")
    except Exception:
        raise ValueError("MONGO_URI environment variable is required, or install 'mongomock' for local development fallback.")
else:
    app.config["MONGO_URI"] = mongo_uri
    mongo = PyMongo(app)

# --- ML Model Loading ---
# Construct absolute paths to model files for robustness
script_dir = os.path.dirname(os.path.abspath(__file__))

# For production deployment: models are in backend/models/
# For local development: models can be in project root/models/
local_model_path = os.path.join(script_dir, 'models', 'best_gradient_boosting_final.pkl')
local_scaler_path = os.path.join(script_dir, 'models', 'best_scaler_final.pkl')

# Fallback to project root for local development
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
root_model_path = os.path.join(project_root, 'models', 'best_gradient_boosting_final.pkl')
root_scaler_path = os.path.join(project_root, 'models', 'best_scaler_final.pkl')

# Use local models if they exist (production), else use root models (development)
model_path = local_model_path if os.path.exists(local_model_path) else root_model_path
scaler_path = local_scaler_path if os.path.exists(local_scaler_path) else root_scaler_path

model = None
scaler = None
feature_names = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
risk_map = {0: "High", 1: "Low", 2: "Medium"}
DEFAULT_MODEL_METRICS = {
    "test_accuracy": "86.7%",
    "high_risk_recall": "94.5%",
    "high_risk_precision": "96%",
    "model_type": "Gradient Boosting Classifier"
}


def clone_model_metrics():
    """Return a fresh copy of the default model metrics dict."""
    return dict(DEFAULT_MODEL_METRICS)


def update_patient_summary(patient_id, risk_level, timestamp, model_metrics=None):
    """Persist the latest risk summary details for the patient document."""
    if not patient_id:
        return

    update_fields = {}

    if risk_level:
        update_fields['risk_level'] = risk_level

    if timestamp:
        update_fields['last_check_in'] = timestamp

    if model_metrics:
        update_fields['last_model_metrics'] = model_metrics

    if not update_fields:
        return

    patients_collection = mongo.db.patients
    try:
        patients_collection.update_one(
            {'_id': ObjectId(patient_id)},
            {'$set': update_fields}
        )
    except Exception:
        logging.exception("Failed to update patient summary for %s", patient_id)


def get_latest_reading(patient_id_str):
    """Fetch the most recent reading for a patient."""
    if not patient_id_str:
        return None
    readings_collection = mongo.db.readings
    return readings_collection.find_one(
        {'patient_id': patient_id_str},
        sort=[('timestamp', -1)]
    )


def load_model_assets(force_reload=False):
    """Ensure the model and scaler are loaded before inference."""
    global model, scaler

    if not force_reload and model is not None and scaler is not None:
        return True

    logging.info(f"Attempting to load model from: {model_path}")
    logging.info(f"Attempting to load scaler from: {scaler_path}")
    logging.info(f"Using feature names: {feature_names}")
    logging.info(f"Using risk mapping: {risk_map}")

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logging.info("ML model assets loaded successfully.")
        logging.info(f"Model type: Gradient Boosting Classifier")
        logging.info(f"Accuracy: 86.7%, High-Risk Recall: 94.5%")
        logging.info(f"False Negatives: 3/55 (5.5%)")
        return True
    except Exception as e:
        logging.error("Error loading ML assets.", exc_info=True)
        model = None
        scaler = None
        return False


# Attempt an initial load so that the first request is fast.
load_model_assets()

def predict_risk(data, patient_age):
    if not load_model_assets():
        logging.error("Model, scaler, or feature names not loaded.")
        return "N/A"

    try:
        # 1. Map frontend keys to model feature names and perform unit conversions
        feature_dict = {
            'Age': patient_age,
            'SystolicBP': data.get('systolic_bp'),
            'DiastolicBP': data.get('diastolic_bp'),
            'BS': data.get('bs'),
            'HeartRate': data.get('heart_rate'),
            # Convert Celsius to Fahrenheit for the model
            'BodyTemp': (float(data.get('body_temp')) * 9/5) + 32 if data.get('body_temp') is not None else None
        }
        logging.debug(f"Initial feature dictionary: {feature_dict}")

        # SAFETY CHECK: Rule-based critical value detection for patient readings
        is_critical, critical_reason = check_critical_values(
            feature_dict['Age'],
            feature_dict['SystolicBP'],
            feature_dict['DiastolicBP'],
            feature_dict['BS'],
            feature_dict['BodyTemp'],
            feature_dict['HeartRate']
        )
        
        if is_critical:
            # Override ML prediction - automatically flag as High Risk
            logging.warning(f"CRITICAL VALUES DETECTED (Patient {patient_age}): {critical_reason}")
            logging.info(f"Safety override - Automatic High Risk due to critical values")
            return "High"

        # If values are reasonable, proceed with ML prediction
        # 2. Create DataFrame from the dictionary
        input_df = pd.DataFrame([feature_dict])
        logging.debug(f"DataFrame before column alignment:\n{input_df.to_string()}")

        # 3. Ensure all required columns from training are present, in the correct order
        missing_cols = set(feature_names) - set(input_df.columns)
        for c in missing_cols:
            input_df[c] = 0
        
        input_df = input_df[feature_names]
        logging.debug(f"DataFrame after column alignment and ordering:\n{input_df.to_string()}")

        # 4. Scale the data
        scaled_data = scaler.transform(input_df)
        logging.debug(f"Scaled data for prediction: {scaled_data}")
        
        # 5. Make prediction
        prediction = model.predict(scaled_data)
        logging.debug(f"Raw model prediction: {prediction}")
        
        # 6. Map prediction to risk level
        risk_level = risk_map.get(prediction[0], "N/A") if risk_map else "N/A"
        logging.info(f"ML prediction - Risk level: {risk_level}")
        return risk_level
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return "Error"

# --- API Endpoints ---
@app.route('/api/predict/<patient_id>', methods=['POST'])
def run_prediction(patient_id):
    # 1. Get patient to find their age
    patients_collection = mongo.db.patients
    patient = patients_collection.find_one_or_404({'_id': ObjectId(patient_id)})
    
    # 2. Get the new reading data from the request
    new_reading_data = request.get_json()
    
    # 3. Predict risk
    risk_level = predict_risk(new_reading_data, patient['age'])
    
    # 4. Store the new reading in the database
    readings_collection = mongo.db.readings
    model_metrics = clone_model_metrics()
    reading_timestamp = datetime.datetime.utcnow()
    db_reading = new_reading_data.copy()
    db_reading['patient_id'] = patient_id
    db_reading['timestamp'] = reading_timestamp
    db_reading['risk_level'] = risk_level
    db_reading['model_metrics'] = model_metrics

    result = readings_collection.insert_one(db_reading)
    update_patient_summary(patient_id, risk_level, reading_timestamp, model_metrics)
    
    # Prepare for JSON response - convert ObjectId and datetime
    db_reading['_id'] = str(result.inserted_id)
    db_reading['timestamp'] = reading_timestamp.isoformat()
    
    # 5. Prepare the response
    # Based on the final model evaluation from the notebook
    # Accuracy: ~91%, High-Risk Recall: ~87% -> FN Rate: ~13%
    response_data = {
        'reading': db_reading,
        'model_metrics': model_metrics
    }
    
    return jsonify(response_data), 201


def check_critical_values(age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate):
    """
    Safety check: Flag critically abnormal values that indicate immediate high risk.
    Based on medical emergency thresholds. Returns (is_critical, reason).
    """
    critical_issues = []
    
    # Critical Age ranges (pregnancy complications increase outside 15-45)
    if age < 15 or age > 50:
        critical_issues.append(f"Age {age} is critically outside safe pregnancy range (15-50)")
    
    # Critical Blood Pressure (Severe Hypotension or Hypertensive Crisis)
    if systolic_bp < 70:
        critical_issues.append(f"Severe hypotension: Systolic BP {systolic_bp} < 70 mmHg")
    elif systolic_bp > 180:
        critical_issues.append(f"Hypertensive crisis: Systolic BP {systolic_bp} > 180 mmHg")
    
    if diastolic_bp < 40:
        critical_issues.append(f"Severe hypotension: Diastolic BP {diastolic_bp} < 40 mmHg")
    elif diastolic_bp > 120:
        critical_issues.append(f"Hypertensive crisis: Diastolic BP {diastolic_bp} > 120 mmHg")
    
    # Critical Blood Sugar (Severe Hypoglycemia or Hyperglycemia)
    if blood_sugar < 3.0:
        critical_issues.append(f"Severe hypoglycemia: Blood Sugar {blood_sugar} < 3.0 mmol/L")
    elif blood_sugar > 25.0:
        critical_issues.append(f"Severe hyperglycemia: Blood Sugar {blood_sugar} > 25.0 mmol/L")
    
    # Critical Body Temperature (Hypothermia or Severe Fever)
    if body_temp < 95.0:  # °F
        critical_issues.append(f"Hypothermia: Body Temp {body_temp}°F < 95°F")
    elif body_temp > 104.0:  # °F
        critical_issues.append(f"Severe fever: Body Temp {body_temp}°F > 104°F")
    
    # Critical Heart Rate (Severe Bradycardia or Tachycardia)
    if heart_rate < 40:
        critical_issues.append(f"Severe bradycardia: Heart Rate {heart_rate} < 40 bpm")
    elif heart_rate > 140:
        critical_issues.append(f"Severe tachycardia: Heart Rate {heart_rate} > 140 bpm")
    
    if critical_issues:
        return True, "; ".join(critical_issues)
    
    return False, None


@app.route('/api/predict', methods=['POST'])
def predict_direct():
    """Direct prediction endpoint without patient_id (for standalone predictions)"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features
        age = float(data.get('Age'))
        systolic_bp = float(data.get('SystolicBP'))
        diastolic_bp = float(data.get('DiastolicBP'))
        blood_sugar = float(data.get('BS'))
        # BodyTemp arrives from the frontend in Celsius; convert to Fahrenheit for safety checks/model
        body_temp_c = float(data.get('BodyTemp'))
        body_temp = (body_temp_c * 9/5) + 32
        heart_rate = float(data.get('HeartRate'))
        
        # SAFETY CHECK: Rule-based critical value detection
        is_critical, critical_reason = check_critical_values(
            age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate
        )
        
        if is_critical:
            # Override ML prediction - automatically flag as High Risk
            logging.warning(f"CRITICAL VALUES DETECTED: {critical_reason}")
            response = {
                'risk_level': 'High',
                'confidence': 0.99,  # High confidence due to critical values
                'prediction': 0,  # 0 = High in risk_map
                'safety_override': True,
                'reason': critical_reason
            }
            logging.info(f"Safety override - Automatic High Risk due to critical values")
            return jsonify(response), 200
        
        # If values are reasonable, proceed with ML prediction
        # Create feature array in correct order
        features = [[age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate]]
        
        # Scale features
        if scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Map prediction to risk level
        risk_level = risk_map.get(prediction, 'Unknown')
        confidence = float(max(prediction_proba))
        
        # Return response
        response = {
            'risk_level': risk_level,
            'confidence': confidence,
            'prediction': int(prediction),
            'safety_override': False
        }
        
        logging.info(f"ML prediction - Risk: {risk_level}, Confidence: {confidence:.2%}")
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Error in direct prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/patients/<patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    """
    Delete a patient and all their readings from the database.
    """
    patients_collection = mongo.db.patients
    readings_collection = mongo.db.readings
    
    # Check if patient exists
    patient = patients_collection.find_one({'_id': ObjectId(patient_id)})
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    
    # Delete all readings for this patient
    readings_collection.delete_many({'patient_id': patient_id})
    
    # Delete the patient
    patients_collection.delete_one({'_id': ObjectId(patient_id)})
    
    return jsonify({"message": "Patient deleted successfully"}), 200


@app.route('/api/patients', methods=['GET', 'POST'])
def manage_patients():
    patients_collection = mongo.db.patients
    if request.method == 'GET':
        patients = []
        for patient in patients_collection.find():
            patient_id_str = str(patient['_id'])

            latest_reading = get_latest_reading(patient_id_str)
            if latest_reading:
                patient['risk_level'] = latest_reading.get('risk_level', patient.get('risk_level', 'N/A'))
                patient['last_check_in'] = latest_reading.get('timestamp', patient.get('last_check_in'))
                if latest_reading.get('model_metrics'):
                    patient['last_model_metrics'] = latest_reading['model_metrics']

            patient['_id'] = patient_id_str

            last_check_in = patient.get('last_check_in')
            if isinstance(last_check_in, datetime.datetime):
                patient['last_check_in'] = last_check_in.isoformat()

            patients.append(patient)
        return jsonify(patients)
    
    elif request.method == 'POST':
        new_patient = request.get_json()
        # Add default empty readings array
        new_patient['readings'] = []
        result = patients_collection.insert_one(new_patient)
        new_patient['_id'] = str(result.inserted_id)
        return jsonify(new_patient), 201

@app.route('/api/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    patients_collection = mongo.db.patients
    patient = patients_collection.find_one_or_404({'_id': ObjectId(patient_id)})
    patient['_id'] = str(patient['_id'])
    
    # Fetch readings for the patient
    readings_collection = mongo.db.readings
    readings = []
    # Sort by timestamp descending
    for reading in readings_collection.find({'patient_id': patient_id}).sort('timestamp', -1):
        reading['_id'] = str(reading['_id'])
        # Ensure timestamp is a string for JSON serialization
        if isinstance(reading['timestamp'], datetime.datetime):
            reading['timestamp'] = reading['timestamp'].isoformat()
        readings.append(reading)
    
    patient['readings'] = readings
    return jsonify(patient)


@app.route('/api/patients/<patient_id>', methods=['PUT'])
def update_patient(patient_id):
    """
    Update patient details (name, age, photo).
    """
    patients_collection = mongo.db.patients
    
    # Check if patient exists
    patient = patients_collection.find_one({'_id': ObjectId(patient_id)})
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    
    data = request.get_json()
    
    # Build update fields
    update_fields = {}
    if 'name' in data and data['name']:
        update_fields['name'] = data['name'].strip()
    if 'age' in data and data['age']:
        update_fields['age'] = int(data['age'])
    if 'photo' in data:
        update_fields['photo'] = data['photo']
    
    if not update_fields:
        return jsonify({"error": "No valid fields to update"}), 400
    
    # Update the patient
    patients_collection.update_one(
        {'_id': ObjectId(patient_id)},
        {'$set': update_fields}
    )
    
    # Return updated patient
    updated_patient = patients_collection.find_one({'_id': ObjectId(patient_id)})
    updated_patient['_id'] = str(updated_patient['_id'])
    
    return jsonify(updated_patient), 200


@app.route('/api/patients/<patient_id>/photo', methods=['PUT'])
def update_patient_photo(patient_id):
    """
    Update the profile photo for a patient.
    Accepts either a base64 encoded image or a URL.
    """
    patients_collection = mongo.db.patients
    
    # Check if patient exists
    patient = patients_collection.find_one({'_id': ObjectId(patient_id)})
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    
    data = request.get_json()
    photo_data = data.get('photo')
    
    if not photo_data:
        return jsonify({"error": "No photo data provided"}), 400
    
    # Update the patient's photo
    patients_collection.update_one(
        {'_id': ObjectId(patient_id)},
        {'$set': {'photo': photo_data}}
    )
    
    return jsonify({"message": "Photo updated successfully", "photo": photo_data}), 200


@app.route('/api/patients/<patient_id>/readings', methods=['POST'])
def add_reading(patient_id):
    """
    Adds a new reading for a patient without running a prediction.
    The risk level is set to 'N/A' by default.
    """
    # 1. Check if patient exists
    patients_collection = mongo.db.patients
    if not patients_collection.find_one({'_id': ObjectId(patient_id)}):
        return jsonify({"error": "Patient not found"}), 404

    # 2. Get the new reading data from the request
    new_reading_data = request.get_json()
    
    # 3. Store the new reading in the database with default risk
    readings_collection = mongo.db.readings
    db_reading = new_reading_data.copy()
    db_reading['patient_id'] = patient_id
    db_reading['timestamp'] = datetime.datetime.utcnow()
    db_reading['risk_level'] = "N/A"  # Default value
    
    result = readings_collection.insert_one(db_reading)
    
    # 4. Prepare for JSON response
    db_reading['_id'] = str(result.inserted_id)
    db_reading['timestamp'] = db_reading['timestamp'].isoformat()
    
    return jsonify(db_reading), 201


@app.route('/api/readings/<reading_id>/predict', methods=['PUT'])
def predict_for_reading(reading_id):
    """
    Runs a prediction for a specific, existing reading.
    """
    readings_collection = mongo.db.readings
    patients_collection = mongo.db.patients

    # 1. Find the reading
    reading = readings_collection.find_one_or_404({'_id': ObjectId(reading_id)})
    
    # 2. Find the associated patient to get their age
    patient = patients_collection.find_one_or_404({'_id': ObjectId(reading['patient_id'])})

    # 3. Predict risk using the data from the existing reading
    risk_level = predict_risk(reading, patient['age'])

    # 4. Update the reading in the database with the new risk level & metrics
    model_metrics = clone_model_metrics()
    readings_collection.update_one(
        {'_id': ObjectId(reading_id)},
        {'$set': {
            'risk_level': risk_level,
            'model_metrics': model_metrics
        }}
    )

    update_patient_summary(
        reading['patient_id'],
        risk_level,
        reading.get('timestamp'),
        model_metrics
    )

    # 5. Fetch the updated reading to return it
    updated_reading = readings_collection.find_one({'_id': ObjectId(reading_id)})
    updated_reading['_id'] = str(updated_reading['_id'])
    if isinstance(updated_reading.get('timestamp'), datetime.datetime):
        updated_reading['timestamp'] = updated_reading['timestamp'].isoformat()

    return jsonify(updated_reading)


# API info endpoint
@app.route('/api', methods=['GET'])
def api_info():
    return jsonify({
        "service": "PulseAI Backend API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "patients": "/api/patients",
            "predict": "/api/predict"
        }
    })


# Health check endpoint for deployment
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "PulseAI Backend"})


@app.route('/', defaults={'path': ''}, methods=['GET'])
@app.route('/<path:path>', methods=['GET'])
def serve_frontend(path):
    if path.startswith('api/'):
        return jsonify({"error": "Not found"}), 404

    if path and os.path.exists(os.path.join(frontend_build_dir, path)):
        return send_from_directory(frontend_build_dir, path)

    index_path = os.path.join(frontend_build_dir, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(frontend_build_dir, 'index.html')

    return jsonify({
        "service": "PulseAI Backend API",
        "status": "running",
        "frontend": "build not found",
        "health": "/health",
        "api": "/api"
    })


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    app.run(debug=debug, host='0.0.0.0', port=port)

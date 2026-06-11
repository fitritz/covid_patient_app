import React, { useState } from 'react';
import InputField from '../common/InputField';
import Button from '../common/Button';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorAlert from '../common/ErrorAlert';
import ResultCard from '../common/ResultCard';
import ModelInfoPanel from '../common/ModelInfoPanel';
import { getApiUrl } from '../../config/api';
import './PredictionPage.css';

const PredictionPage = () => {
  const [formData, setFormData] = useState({
    age: '',
    systolic_bp: '',
    diastolic_bp: '',
    blood_sugar: '',
    body_temp: '',
    heart_rate: ''
  });

  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Field labels (no range restrictions)
  const fieldLabels = {
    age: 'Age',
    systolic_bp: 'Systolic BP',
    diastolic_bp: 'Diastolic BP',
    blood_sugar: 'Blood Sugar',
  body_temp: 'Body Temperature (°C)',
    heart_rate: 'Heart Rate'
  };

  // Handle input change
  const handleChange = (name, value) => {
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));

    // Clear error for this field
    if (errors[name]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[name];
        return newErrors;
      });
    }
  };

  // Validate single field (only check if filled and is a number)
  const validateField = (name, value) => {
    const label = fieldLabels[name];

    if (!value) {
      return `${label} is required`;
    }

    if (isNaN(parseFloat(value))) {
      return `${label} must be a valid number`;
    }

    return null;
  };

  // Validate all fields
  const validateForm = () => {
    const newErrors = {};
    let isValid = true;

    Object.keys(formData).forEach(key => {
      const error = validateField(key, formData[key]);
      if (error) {
        newErrors[key] = error;
        isValid = false;
      }
    });

    setErrors(newErrors);
    return isValid;
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(getApiUrl('/api/predict'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          Age: parseFloat(formData.age),
          SystolicBP: parseFloat(formData.systolic_bp),
          DiastolicBP: parseFloat(formData.diastolic_bp),
          BS: parseFloat(formData.blood_sugar),
          BodyTemp: parseFloat(formData.body_temp),
          HeartRate: parseFloat(formData.heart_rate)
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setResult({
        riskLevel: data.risk_level,
        confidence: data.confidence,
        inputData: formData,
        safetyOverride: data.safety_override || false,
        safetyReason: data.reason || null
      });
    } catch (err) {
      setError(err.message || 'Failed to get prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Reset form
  const handleReset = () => {
    setFormData({
      age: '',
      systolic_bp: '',
      diastolic_bp: '',
      blood_sugar: '',
      body_temp: '',
      heart_rate: ''
    });
    setErrors({});
    setResult(null);
    setError(null);
  };

  // Handle retry after error
  const handleRetry = () => {
    setError(null);
    handleSubmit({ preventDefault: () => {} });
  };

  // Handle new assessment after result
  const handleNewAssessment = () => {
    handleReset();
  };

  // Show loading state
  if (loading) {
    return (
      <div className="prediction-page">
        <div className="prediction-container">
          <LoadingSpinner variant="fullscreen" size="large" />
        </div>
      </div>
    );
  }

  // Show result
  if (result) {
    return (
      <div className="prediction-page">
        <div className="prediction-container">
          <ModelInfoPanel />
          <ResultCard
            riskLevel={result.riskLevel}
            confidence={result.confidence}
            inputData={result.inputData}
            onNewAssessment={handleNewAssessment}
            safetyOverride={result.safetyOverride}
            safetyReason={result.safetyReason}
          />
        </div>
      </div>
    );
  }

  // Show form
  return (
    <div className="prediction-page">
      <div className="prediction-container">
        <header className="prediction-header">
          <h1 className="prediction-title">Maternal Health Risk Predictor</h1>
          <p className="prediction-subtitle">
            Enter vital signs to assess maternal health risk using our AI model
          </p>
        </header>

        <ModelInfoPanel />

        {error && (
          <ErrorAlert
            message={error}
            onRetry={handleRetry}
            onDismiss={() => setError(null)}
          />
        )}

        <form onSubmit={handleSubmit} className="prediction-form">
          <div className="prediction-form-grid">
            <InputField
              name="age"
              label="Age"
              unit="years"
              value={formData.age}
              onChange={(e) => handleChange('age', e.target.value)}
              error={errors.age}
              type="number"
              step="1"
              required
            />

            <InputField
              name="systolic_bp"
              label="Systolic BP"
              unit="mmHg"
              value={formData.systolic_bp}
              onChange={(e) => handleChange('systolic_bp', e.target.value)}
              error={errors.systolic_bp}
              type="number"
              step="1"
              required
            />

            <InputField
              name="diastolic_bp"
              label="Diastolic BP"
              unit="mmHg"
              value={formData.diastolic_bp}
              onChange={(e) => handleChange('diastolic_bp', e.target.value)}
              error={errors.diastolic_bp}
              type="number"
              step="1"
              required
            />

            <InputField
              name="blood_sugar"
              label="Blood Sugar"
              unit="mmol/L"
              value={formData.blood_sugar}
              onChange={(e) => handleChange('blood_sugar', e.target.value)}
              error={errors.blood_sugar}
              type="number"
              step="0.1"
              required
            />

            <InputField
              name="body_temp"
              label="Body Temperature"
              unit="°C"
              value={formData.body_temp}
              onChange={(e) => handleChange('body_temp', e.target.value)}
              error={errors.body_temp}
              type="number"
              step="0.1"
              required
            />

            <InputField
              name="heart_rate"
              label="Heart Rate"
              unit="bpm"
              value={formData.heart_rate}
              onChange={(e) => handleChange('heart_rate', e.target.value)}
              error={errors.heart_rate}
              type="number"
              step="1"
              required
            />
          </div>

          <div className="prediction-form-actions">
            <Button type="submit" variant="primary" fullWidth>
              Predict Risk
            </Button>
            <Button type="button" variant="secondary" onClick={handleReset} fullWidth>
              Reset Form
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default PredictionPage;

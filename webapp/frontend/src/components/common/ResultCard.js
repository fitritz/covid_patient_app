import React from 'react';
import Button from './Button';
import './ResultCard.css';

const ResultCard = ({ riskLevel, confidence, inputData, onNewAssessment, safetyOverride, safetyReason }) => {
  // Determine risk styling
  const getRiskConfig = (level) => {
    const configs = {
      'low': {
        label: 'Low Risk',
        className: 'risk-low',
        icon: '✓',
        message: 'The vital signs indicate low risk. Continue regular monitoring.'
      },
      'mid': {
        label: 'Medium Risk',
        className: 'risk-medium',
        icon: '⚠',
        message: 'Moderate risk detected. Consider additional monitoring and consultation.'
      },
      'high': {
        label: 'High Risk',
        className: 'risk-high',
        icon: '!',
        message: safetyOverride 
          ? 'CRITICAL: One or more vital signs are dangerously abnormal. Seek immediate emergency medical attention.'
          : 'High risk detected. Immediate medical attention is recommended.'
      }
    };
    return configs[level?.toLowerCase()] || configs['mid'];
  };

  const riskConfig = getRiskConfig(riskLevel);
  const confidencePercent = (confidence * 100).toFixed(1);

  // Format input data for display
  const formatInputData = () => {
    return [
      { label: 'Age', value: inputData.age, unit: 'years' },
      { label: 'Systolic BP', value: inputData.systolic_bp, unit: 'mmHg' },
      { label: 'Diastolic BP', value: inputData.diastolic_bp, unit: 'mmHg' },
      { label: 'Blood Sugar', value: inputData.blood_sugar, unit: 'mmol/L' },
      { label: 'Body Temperature', value: inputData.body_temp, unit: '°C' },
      { label: 'Heart Rate', value: inputData.heart_rate, unit: 'bpm' }
    ];
  };

  const handlePrint = () => {
    window.print();
  };

  const currentDate = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });

  return (
    <div className="result-card">
      <div className="result-header">
        <h2 className="result-title">PulseAI Risk Assessment Report</h2>
        <div className="result-date">{currentDate}</div>
      </div>

      <div className="result-content-grid">
        {/* Left Column: Patient Vitals */}
        <div className="result-section result-inputs">
          <h3 className="result-section-title">Patient Vitals</h3>
          <div className="result-inputs-grid">
            {formatInputData().map((item, index) => (
              <div key={index} className="result-input-item">
                <span className="result-input-label">
                  {item.label} <span className="result-input-unit">({item.unit})</span>
                </span>
                <span className="result-input-value">
                  {item.value}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Right Column: Assessment */}
        <div className="result-section result-assessment">
          <h3 className="result-section-title">Assessment Analysis</h3>
          
          {/* Risk Badge */}
          <div className={`result-risk-badge ${riskConfig.className}`}>
            <span className="result-risk-icon" aria-hidden="true">{riskConfig.icon}</span>
            <div className="result-risk-details">
              <span className="result-risk-label">{riskConfig.label}</span>
              <span className="result-risk-sublabel">Risk Level</span>
            </div>
          </div>

          {/* Confidence Score */}
          <div className="result-confidence">
            <div className="result-confidence-header">
              <span className="result-confidence-label">Model Confidence</span>
              <span className="result-confidence-value">{confidencePercent}%</span>
            </div>
            <div className="result-confidence-bar-container">
              <div 
                className="result-confidence-bar"
                style={{ width: `${confidencePercent}%` }}
                role="progressbar"
                aria-valuenow={confidencePercent}
                aria-valuemin="0"
                aria-valuemax="100"
                aria-label={`Confidence score: ${confidencePercent}%`}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Full Width: Messages & Warnings */}
      <div className="result-details">
        {/* Safety Override Warning */}
        {safetyOverride && (
          <div className="result-safety-warning">
            <div className="safety-warning-header">
              <span className="safety-warning-icon" aria-hidden="true">🚨</span>
              <strong>Critical Values Detected</strong>
            </div>
            <div className="safety-warning-list">
              {safetyReason.split(';').map((reason, index) => (
                reason.trim() && (
                  <div key={index} className="safety-warning-item">
                    {reason.trim()}
                  </div>
                )
              ))}
            </div>
            <p className="safety-warning-action">
              These values fall outside safe medical ranges. This assessment was automatically 
              flagged as HIGH RISK. Please seek emergency medical care immediately.
            </p>
          </div>
        )}

        {/* Risk Message */}
        <div className="result-message-container">
          <h4 className="result-message-title">Recommendation</h4>
          <p className="result-message">{riskConfig.message}</p>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="result-actions">
        <Button variant="primary" onClick={onNewAssessment} fullWidth>
          New Assessment
        </Button>
        <Button variant="secondary" onClick={handlePrint} fullWidth>
          Print Official Report
        </Button>
      </div>

      {/* Disclaimer */}
      <div className="result-disclaimer">
        <p>
          <strong>Medical Disclaimer:</strong> This is an AI-assisted prediction tool and should not replace professional medical advice. 
          Always consult with healthcare professionals for proper diagnosis and treatment.
        </p>
        <div className="result-footer-brand">Generated by PulseAI System</div>
      </div>
    </div>
  );
};

export default ResultCard;

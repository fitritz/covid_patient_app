import React from 'react';
import './PendingPrediction.css';

const getIdString = (id) => {
  if (!id) return null;
  if (typeof id === 'string') return id;
  if (typeof id === 'object') {
    if (id.$oid) return id.$oid;
    if (id.$id) return id.$id;
    if (typeof id.toString === 'function') return id.toString();
  }
  return String(id);
};

const PendingPrediction = ({ reading, onRunPrediction, isPredicting }) => {
  if (!reading) {
    return null;
  }

  const handlePredictClick = () => {
    onRunPrediction(getIdString(reading._id));
  };

  return (
    <div className="pending-prediction-card">
      <div className="section-header">
        <h3>Pending Health Prediction</h3>
        <button 
          onClick={handlePredictClick} 
          className="btn btn-primary"
          disabled={isPredicting}
        >
          {isPredicting ? 'Predicting...' : 'Run Prediction'}
        </button>
      </div>
      <div className="pending-vitals">
        <div className="vital-item">
          <span className="vital-label">Systolic BP</span>
          <span className="vital-value">{reading.systolic_bp}</span>
        </div>
        <div className="vital-item">
          <span className="vital-label">Diastolic BP</span>
          <span className="vital-value">{reading.diastolic_bp}</span>
        </div>
        <div className="vital-item">
          <span className="vital-label">Blood Sugar (BS)</span>
          <span className="vital-value">{reading.bs}</span>
        </div>
        <div className="vital-item">
          <span className="vital-label">Heart Rate</span>
          <span className="vital-value">{reading.heart_rate}</span>
        </div>
        <div className="vital-item">
          <span className="vital-label">Body Temp (°C)</span>
          <span className="vital-value">{reading.body_temp}</span>
        </div>
      </div>
    </div>
  );
};

export default PendingPrediction;

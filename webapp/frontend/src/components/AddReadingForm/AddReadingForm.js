import React, { useState } from 'react';
import './AddReadingForm.css';

const AddReadingForm = ({ isOpen, onSubmit, onClose }) => {
  const [systolicBP, setSystolicBP] = useState('');
  const [diastolicBP, setDiastolicBP] = useState('');
  const [bs, setBs] = useState('');
  const [bodyTemp, setBodyTemp] = useState('');
  const [heartRate, setHeartRate] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      systolic_bp: parseInt(systolicBP),
      diastolic_bp: parseInt(diastolicBP),
      bs: parseFloat(bs),
      body_temp: parseFloat(bodyTemp),
      heart_rate: parseInt(heartRate),
    });
    // Clear form
    setSystolicBP('');
    setDiastolicBP('');
    setBs('');
    setBodyTemp('');
    setHeartRate('');
    onClose(); // Close modal on submit
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Manual Entry</h3>
          <button onClick={onClose} className="close-button">&times;</button>
        </div>
        <form onSubmit={handleSubmit} className="add-reading-form">
          <div className="form-grid">
            <div className="form-group">
              <label>Systolic BP</label>
              <input type="number" placeholder="e.g., 120" value={systolicBP} onChange={e => setSystolicBP(e.target.value)} required />
            </div>
            <div className="form-group">
              <label>Diastolic BP</label>
              <input type="number" placeholder="e.g., 80" value={diastolicBP} onChange={e => setDiastolicBP(e.target.value)} required />
            </div>
            <div className="form-group">
              <label>Blood Sugar (mmol/L)</label>
              <input type="number" step="0.1" placeholder="e.g., 5.5" value={bs} onChange={e => setBs(e.target.value)} required />
            </div>
            <div className="form-group">
              <label>Body Temp (°C)</label>
              <input type="number" step="0.1" placeholder="e.g., 36.6" value={bodyTemp} onChange={e => setBodyTemp(e.target.value)} required />
            </div>
            <div className="form-group">
              <label>Heart Rate (bpm)</label>
              <input type="number" placeholder="e.g., 75" value={heartRate} onChange={e => setHeartRate(e.target.value)} required />
            </div>
          </div>
          <div className="modal-footer">
            <button type="button" onClick={onClose} className="btn btn-secondary">Cancel</button>
            <button type="submit" className="btn btn-primary">Add Reading</button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AddReadingForm;

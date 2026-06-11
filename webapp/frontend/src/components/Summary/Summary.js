import React from 'react';
import './Summary.css';

const normalizeRiskLevel = (value) => {
  if (!value) return '';
  return String(value).toLowerCase();
};

const Summary = ({ patients }) => {
  const totalPatients = patients.length;
  const highRisk = patients.filter(p => normalizeRiskLevel(p.risk_level) === 'high').length;
  const mediumRisk = patients.filter(p => normalizeRiskLevel(p.risk_level) === 'medium').length;
  const lowRisk = patients.filter(p => normalizeRiskLevel(p.risk_level) === 'low').length;

  return (
    <div className="summary-cards">
      <div className="card">
        <h3>Total Patients</h3>
        <p>{totalPatients}</p>
      </div>
      <div className="card">
        <h3>High Risk</h3>
        <p>{highRisk}</p>
      </div>
      <div className="card">
        <h3>Medium Risk</h3>
        <p>{mediumRisk}</p>
      </div>
      <div className="card">
        <h3>Low Risk</h3>
        <p>{lowRisk}</p>
      </div>
    </div>
  );
};

export default Summary;

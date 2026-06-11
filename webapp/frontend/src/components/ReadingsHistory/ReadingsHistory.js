import React from 'react';
import './ReadingsHistory.css';

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

const ReadingsHistory = ({ readings }) => {

  const getRiskLevelClass = (level) => {
    if (!level) return 'risk-na';
    const lowerLevel = level.toLowerCase();
    if (lowerLevel === 'high') return 'risk-high';
    if (lowerLevel === 'medium') return 'risk-medium';
    if (lowerLevel === 'low') return 'risk-low';
    return 'risk-na';
  };

  return (
    <div className="readings-history-card">
      <h3>Reading History</h3>
      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Systolic BP</th>
              <th>Diastolic BP</th>
              <th>Blood Sugar</th>
              <th>Body Temp</th>
              <th>Heart Rate</th>
              <th>Risk Level</th>
            </tr>
          </thead>
          <tbody>
            {readings && readings.length > 0 ? (
              readings.map((reading) => (
                <tr key={getIdString(reading._id)}>
                  <td>{new Date(reading.timestamp?.$date || reading.timestamp).toLocaleString()}</td>
                  <td>{reading.systolic_bp}</td>
                  <td>{reading.diastolic_bp}</td>
                  <td>{reading.bs}</td>
                  <td>{reading.body_temp}°C</td>
                  <td>{reading.heart_rate}</td>
                  <td>
                    <span className={`risk-badge ${getRiskLevelClass(reading.risk_level)}`}>
                      {reading.risk_level || 'N/A'}
                    </span>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="7">No readings found.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ReadingsHistory;

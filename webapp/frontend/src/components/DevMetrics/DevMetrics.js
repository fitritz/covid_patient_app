import React from 'react';
import './DevMetrics.css';

const DevMetrics = ({ metrics }) => {
  if (!metrics) {
    return null;
  }

  return (
    <div className="dev-metrics-card">
      <h4>Developer Info: Model Performance</h4>
      <div className="metrics-grid">
        <div className="metric-item">
          <span className="metric-label">Test Accuracy</span>
          <span className="metric-value">{metrics.test_accuracy}</span>
        </div>
        <div className="metric-item">
          <span className="metric-label">High-Risk FN Rate</span>
          <span className="metric-value">{metrics.high_risk_false_negative_rate}</span>
        </div>
      </div>
      <p className="metrics-disclaimer">
        These metrics are based on the final test set evaluation of the deployed model. For developer reference only.
      </p>
    </div>
  );
};

export default DevMetrics;

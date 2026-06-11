import React, { useState } from 'react';
import './ModelInfoPanel.css';

const ModelInfoPanel = () => {
  const [isExpanded, setIsExpanded] = useState(false);

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="model-info-panel">
      <button
        className="model-info-header"
        onClick={toggleExpanded}
        aria-expanded={isExpanded}
        aria-controls="model-info-content"
      >
        <div className="model-info-header-content">
          <span className="model-info-icon" aria-hidden="true">ℹ️</span>
          <h3 className="model-info-title">About the Model</h3>
        </div>
        <span 
          className={`model-info-toggle ${isExpanded ? 'expanded' : ''}`}
          aria-hidden="true"
        >
          ▼
        </span>
      </button>

      {isExpanded && (
        <div 
          id="model-info-content" 
          className="model-info-content"
          role="region"
          aria-label="Model Information Details"
        >
          <div className="model-info-grid">
            <div className="model-info-item">
              <span className="model-info-label">Algorithm</span>
              <span className="model-info-value">Gradient Boosting Classifier</span>
            </div>
            
            <div className="model-info-item">
              <span className="model-info-label">Overall Accuracy</span>
              <span className="model-info-value model-info-highlight">86.7%</span>
            </div>
            
            <div className="model-info-item">
              <span className="model-info-label">High-Risk Recall</span>
              <span className="model-info-value model-info-highlight">94.5%</span>
            </div>
            
            <div className="model-info-item">
              <span className="model-info-label">False Negatives</span>
              <span className="model-info-value">3 out of 55</span>
            </div>
            
            <div className="model-info-item">
              <span className="model-info-label">Training Data</span>
              <span className="model-info-value">1,014 cases</span>
            </div>
            
            <div className="model-info-item">
              <span className="model-info-label">Last Updated</span>
              <span className="model-info-value">Recent Session</span>
            </div>
          </div>

          <div className="model-info-note">
            <p>
              <strong>Note:</strong> This model is optimized for maternal health risk assessment,
              with a focus on minimizing false negatives to ensure high-risk cases are properly identified.
              It analyzes six vital health indicators to provide risk predictions.
            </p>
          </div>

          <a 
            href="/model-documentation.html" 
            className="model-info-link"
            target="_blank"
            rel="noopener noreferrer"
          >
            View Full Documentation →
          </a>
        </div>
      )}
    </div>
  );
};

export default ModelInfoPanel;

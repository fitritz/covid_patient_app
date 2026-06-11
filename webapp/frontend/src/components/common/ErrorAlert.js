import React from 'react';
import Button from './Button';
import './ErrorAlert.css';

const ErrorAlert = ({ message, onRetry, onDismiss }) => {
  return (
    <div className="error-alert" role="alert" aria-live="assertive">
      <div className="error-alert-content">
        <span className="error-alert-icon" aria-hidden="true">⚠️</span>
        <div className="error-alert-text">
          <h3 className="error-alert-title">Error</h3>
          <p className="error-alert-message">{message || 'Something went wrong. Please try again.'}</p>
        </div>
      </div>
      
      <div className="error-alert-actions">
        {onRetry && (
          <Button variant="primary" onClick={onRetry}>
            Retry
          </Button>
        )}
        {onDismiss && (
          <Button variant="secondary" onClick={onDismiss}>
            Dismiss
          </Button>
        )}
      </div>
    </div>
  );
};

export default ErrorAlert;

import React from 'react';
import './LoadingSpinner.css';

const LoadingSpinner = ({ variant = 'fullscreen', size = 'medium' }) => {
  if (variant === 'inline') {
    return <span className={`spinner spinner-inline spinner-${size}`} aria-label="Loading" />;
  }

  return (
    <div className="spinner-fullscreen" role="status" aria-live="polite">
      <div className={`spinner spinner-${size}`} />
      <p className="spinner-text">Loading PulseAI...</p>
    </div>
  );
};

export default LoadingSpinner;

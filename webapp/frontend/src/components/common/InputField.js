import React from 'react';
import './InputField.css';

const InputField = ({
  name,
  label,
  unit,
  value,
  onChange,
  error,
  min,
  max,
  placeholder,
  type = 'number',
  step = 'any',
  disabled = false,
  required = true
}) => {
  const isValid = value && !error;
  const hasError = !!error;

  return (
    <div className="input-field">
      <label htmlFor={name} className="input-label">
        {label} {unit && <span className="input-unit">({unit})</span>}
        {required && <span className="input-required" aria-label="required">*</span>}
      </label>
      
      <div className="input-wrapper">
        <input
          id={name}
          name={name}
          type={type}
          value={value}
          onChange={onChange}
          placeholder={placeholder || `Enter ${label.toLowerCase()}`}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          required={required}
          className={`input ${hasError ? 'input-error' : ''} ${isValid ? 'input-valid' : ''} ${disabled ? 'input-disabled' : ''}`}
          aria-invalid={hasError}
          aria-describedby={hasError ? `${name}-error` : undefined}
        />
        
        {isValid && !hasError && (
          <span className="input-icon input-icon-valid" aria-label="valid">✓</span>
        )}
        
        {hasError && (
          <span className="input-icon input-icon-error" aria-label="error">⚠</span>
        )}
      </div>
      
      {hasError && (
        <p id={`${name}-error`} className="input-helper input-helper-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
};

export default InputField;

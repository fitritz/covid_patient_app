import React from 'react';
import LoadingSpinner from './LoadingSpinner';
import './Button.css';

const Button = ({
  variant = 'primary',
  type = 'button',
  onClick,
  disabled = false,
  loading = false,
  children,
  fullWidth = false,
  ...props
}) => {
  const className = `btn btn-${variant} ${fullWidth ? 'btn-full-width' : ''} ${loading ? 'btn-loading' : ''}`;

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={className}
      aria-busy={loading}
      {...props}
    >
      {loading ? (
        <>
          <LoadingSpinner variant="inline" />
          <span>Processing...</span>
        </>
      ) : (
        children
      )}
    </button>
  );
};

export default Button;

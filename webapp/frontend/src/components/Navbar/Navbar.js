import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand">
        <span className="navbar-logo">❤️</span>
        <span className="navbar-title">PulseAI</span>
        <span className="navbar-tagline">Maternal Health Monitor</span>
      </Link>
      <div className="navbar-links">
        <Link to="/" className="navbar-link">Dashboard</Link>
        <Link to="/predict" className="navbar-link navbar-link-primary">Predict Risk</Link>
      </div>
    </nav>
  );
};

export default Navbar;

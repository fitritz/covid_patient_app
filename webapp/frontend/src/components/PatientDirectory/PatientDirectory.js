import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { StaggerContainer, FadeUpItem, GlowCard, TextReveal } from '../common/MotionWrappers';
import { getApiUrl } from '../../config/api';
import './PatientDirectory.css';

const normalizeTimestamp = (value) => {
  if (!value) return null;
  if (typeof value === 'string') {
    return new Date(value);
  }
  if (typeof value === 'object' && value.$date) {
    return new Date(value.$date);
  }
  return null;
};

const PatientDirectory = ({ patients, onDeletePatient }) => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('name'); // 'name', 'risk', 'age'
  const [deletingId, setDeletingId] = useState(null);

  const handleDelete = async (e, patientId, patientName) => {
    e.stopPropagation(); // Prevent card click navigation
    
    if (!window.confirm(`Are you sure you want to delete ${patientName}? This will also delete all their readings.`)) {
      return;
    }
    
    setDeletingId(patientId);
    
    try {
      const response = await fetch(getApiUrl(`/api/patients/${patientId}`), {
        method: 'DELETE',
      });
      
      if (response.ok) {
        if (onDeletePatient) {
          onDeletePatient(patientId);
        }
      } else {
        alert('Failed to delete patient');
      }
    } catch (error) {
      console.error('Error deleting patient:', error);
      alert('Error deleting patient');
    } finally {
      setDeletingId(null);
    }
  };

  const filteredAndSortedPatients = useMemo(() => {
    if (!patients) return [];

    let result = [...patients];

    // Filter
    if (searchTerm && searchTerm.trim() !== '') {
      const lowerTerm = searchTerm.toLowerCase().trim();
      result = result.filter(p => 
        p.name.toLowerCase().includes(lowerTerm) || 
        (p.patient_id && p.patient_id.toString().includes(lowerTerm))
      );
    }

    // Sort
    result.sort((a, b) => {
      if (sortBy === 'name') return a.name.localeCompare(b.name);
      if (sortBy === 'age') return a.age - b.age;
      if (sortBy === 'risk') {
        const riskOrder = { 'High': 3, 'Medium': 2, 'Low': 1, 'N/A': 0 };
        return (riskOrder[b.risk_level] || 0) - (riskOrder[a.risk_level] || 0);
      }
      return 0;
    });

    return result;
  }, [patients, searchTerm, sortBy]);

  const clearSearch = () => setSearchTerm('');

  return (
    <div className="patient-directory">
      <div className="directory-header">
        <h2 className="creative-title">
          <span className="title-word">Patient</span>
          <span className="title-word highlight">Directory</span>
        </h2>
        
        <div className="directory-controls">
          <div className="search-bar">
            <span className="search-icon">🔍</span>
            <input 
              type="text" 
              placeholder="Search patients..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            {searchTerm && (
              <button className="clear-search-btn" onClick={clearSearch}>
                ✕
              </button>
            )}
          </div>
          
          <div className="sort-dropdown">
            <span className="sort-label">Sort by:</span>
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
              <option value="name">Name</option>
              <option value="risk">Risk Level</option>
              <option value="age">Age</option>
            </select>
          </div>
        </div>
      </div>

      {(!filteredAndSortedPatients || filteredAndSortedPatients.length === 0) ? (
        <div className="empty-state">
          <p>No patients found matching your criteria.</p>
        </div>
      ) : (
        <StaggerContainer className="patient-grid" animationKey={`grid-${filteredAndSortedPatients.length}-${searchTerm}`}>
          {filteredAndSortedPatients.map((patient) => (
            <FadeUpItem key={patient._id}>
              <GlowCard 
                className={`patient-card ${deletingId === patient._id ? 'deleting' : ''}`}
                onClick={() => navigate(`/patients/${patient._id}`)}
              >
                {/* Large centered photo */}
                <div className="patient-photo-container">
                  <div className="patient-avatar-large">
                    {patient.photo ? (
                      <img src={patient.photo} alt={patient.name} className="avatar-photo" />
                    ) : (
                      <span className="avatar-initial">{patient.name.charAt(0)}</span>
                    )}
                  </div>
                  <div className="patient-risk-indicator" data-risk={patient.risk_level}>
                    <span className="risk-dot"></span>
                    {patient.risk_level || 'Unknown'}
                  </div>
                </div>
                
                <div className="patient-card-body">
                  <h3 className="patient-name">{patient.name}</h3>
                  
                  <div className="patient-meta">
                    <div className="meta-item">
                      <span className="meta-icon">👤</span>
                      <span>{patient.age} years</span>
                    </div>
                    <div className="meta-item">
                      <span className="meta-icon">📅</span>
                      <span>
                        {(() => {
                          const parsedDate = normalizeTimestamp(patient.last_check_in);
                          return parsedDate ? parsedDate.toLocaleDateString() : 'No check-in';
                        })()}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="patient-card-actions">
                  <button 
                    className="action-btn view-btn"
                    onClick={(e) => { e.stopPropagation(); navigate(`/patients/${patient._id}`); }}
                  >
                    View Profile
                  </button>
                  <button 
                    className="action-btn delete-btn"
                    onClick={(e) => handleDelete(e, patient._id, patient.name)}
                    disabled={deletingId === patient._id}
                  >
                    {deletingId === patient._id ? '...' : '🗑️'}
                  </button>
                </div>
              </GlowCard>
            </FadeUpItem>
          ))}
        </StaggerContainer>
      )}
    </div>
  );
};

export default PatientDirectory;

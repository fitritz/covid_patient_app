import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Summary from '../Summary/Summary';
import PatientDirectory from '../PatientDirectory/PatientDirectory';
import AddPatientForm from '../AddPatientForm/AddPatientForm';
import { PageTransition, StaggerContainer, FadeUpItem } from '../common/MotionWrappers';
import { getApiUrl } from '../../config/api';
import './Dashboard.css';

const Dashboard = () => {
  const [patients, setPatients] = useState([]);

  const fetchPatients = async () => {
    const response = await fetch(getApiUrl('/api/patients'));
    const data = await response.json();
    setPatients(data);
  };

  useEffect(() => {
    fetchPatients();
  }, []);

  const handleAddPatient = async (patient) => {
    const response = await fetch(getApiUrl('/api/patients'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(patient),
    });
    if (response.ok) {
      fetchPatients(); // Refresh the patient list
    } else {
      alert('Failed to add patient.');
    }
  };

  const handleDeletePatient = (patientId) => {
    setPatients(prevPatients => prevPatients.filter(p => p._id !== patientId));
  };

  return (
    <PageTransition>
      <div className="dashboard">
        {/* Section 1: Hero / Stats / Actions - Takes up full viewport height */}
        <section className="dashboard-hero-section">
          <StaggerContainer>
            <FadeUpItem>
              <div className="hero-header">
                <h1 className="hero-title">Maternal Health Monitor</h1>
                <p className="hero-subtitle">Real-time IoT analytics & risk prediction</p>
              </div>
            </FadeUpItem>

            <FadeUpItem>
              <Summary patients={patients} />
            </FadeUpItem>
            
            <FadeUpItem>
              <div className="dashboard-actions">
                <AddPatientForm onAddPatient={handleAddPatient} />
              </div>
            </FadeUpItem>

            <motion.div 
              className="scroll-indicator"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1, y: [0, 10, 0] }}
              transition={{ delay: 2, duration: 2, repeat: Infinity }}
            >
              <span>Scroll for Patients</span>
              <div className="arrow-down">↓</div>
            </motion.div>
          </StaggerContainer>
        </section>
        
        {/* Section 2: Patient Directory - Appears after scroll */}
        <section className="dashboard-directory-section">
          <PatientDirectory patients={patients} onDeletePatient={handleDeletePatient} />
        </section>
      </div>
    </PageTransition>
  );
};

export default Dashboard;

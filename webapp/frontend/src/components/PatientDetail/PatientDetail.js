import React, { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import PatientHeader from '../PatientHeader/PatientHeader';
import AddReadingForm from '../AddReadingForm/AddReadingForm';
import VitalSignsTrends from '../VitalSignsTrends/VitalSignsTrends';
import PendingPrediction from '../PendingPrediction/PendingPrediction';
import ReadingsHistory from '../ReadingsHistory/ReadingsHistory';
import { PageTransition, StaggerContainer, FadeUpItem } from '../common/MotionWrappers';
import { getApiUrl } from '../../config/api';
import './PatientDetail.css';

const PatientDetail = () => {
  const { id } = useParams();
  const [patient, setPatient] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  const normalizeId = (value) => {
    if (!value) return null;
    if (typeof value === 'string') return value;
    if (typeof value === 'object') {
      if (value.$oid) return value.$oid;
      if (value.$id) return value.$id;
      if (typeof value.toString === 'function') return value.toString();
    }
    return String(value);
  };

  const fetchPatient = useCallback(async () => {
    try {
      const response = await fetch(getApiUrl(`/api/patients/${id}`));
      if (!response.ok) {
        throw new Error('Patient not found');
      }
      const data = await response.json();
      setPatient(data);
    } catch (error) {
      console.error("Failed to fetch patient:", error);
      setPatient(null);
    }
  }, [id]);

  useEffect(() => {
    fetchPatient();
  }, [fetchPatient]);

  const handleAddReading = async (reading) => {
    const response = await fetch(getApiUrl(`/api/patients/${id}/readings`), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(reading),
    });
    if (response.ok) {
      fetchPatient(); // Refresh patient data
      setIsModalOpen(false); // Close modal on success
    } else {
      alert('Failed to add reading.');
    }
  };

  const handleRunPrediction = async (readingId) => {
    const normalizedReadingId = normalizeId(readingId);
    if (!normalizedReadingId) {
      alert('Unable to run prediction: reading ID is missing.');
      return;
    }

    setIsPredicting(true);
    try {
      const response = await fetch(getApiUrl(`/api/readings/${normalizedReadingId}/predict`), {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (response.ok) {
        const updatedReading = await response.json();
        setPatient(prevPatient => {
          if (!prevPatient) {
            return prevPatient;
          }
          const updatedId = normalizeId(updatedReading._id);
          const updatedReadings = prevPatient.readings.map(r => {
            const currentId = normalizeId(r._id);
            return currentId === updatedId ? { ...r, ...updatedReading } : r;
          });
          return { ...prevPatient, readings: updatedReadings };
        });

        // Fetch fresh data to ensure charts and history stay in sync with the database
        await fetchPatient();

      } else {
        alert('Failed to run prediction.');
      }
    } catch (error) {
      console.error("Prediction error:", error);
      alert('An error occurred during prediction.');
    } finally {
      setIsPredicting(false);
    }
  };
  
  if (!patient) {
    return <div className="loading-container">Loading patient data...</div>;
  }

  const pendingReading = patient?.readings?.find(r => r.risk_level === 'N/A');
  const pendingReadingId = normalizeId(pendingReading?._id);
  const readingsForHistory = patient?.readings?.filter(r => normalizeId(r._id) !== pendingReadingId) || [];

  const chartData = patient.readings ? patient.readings.map(r => {
    // The timestamp from MongoDB can be either an object or a string.
    const date = new Date(r.timestamp.$date || r.timestamp);
    return {
      // Format to 'HH:MM:SS'
      name: date.toLocaleTimeString('en-US', { hour12: false }),
      ...r
    };
  }).slice(-10).reverse() : []; // reverse to show oldest first on the chart


  return (
    <PageTransition>
      <div className="patient-detail-container">
        <div className="toolbar">
          <Link to="/" className="back-button">&larr; Back to Dashboard</Link>
          <button onClick={() => setIsModalOpen(true)} className="btn btn-primary">+ Manual Entry</button>
        </div>

        <StaggerContainer>
          <FadeUpItem>
            <PatientHeader 
              patient={patient} 
              onPhotoUpdate={(photo) => setPatient(prev => ({ ...prev, photo }))}
              onPatientUpdate={(updatedPatient) => setPatient(prev => ({ ...prev, ...updatedPatient }))}
            />
          </FadeUpItem>

          <FadeUpItem>
            <PendingPrediction 
              reading={pendingReading} 
              onRunPrediction={handleRunPrediction}
              isPredicting={isPredicting} 
            />
          </FadeUpItem>

          <FadeUpItem>
            <VitalSignsTrends data={chartData} />
          </FadeUpItem>

          <FadeUpItem>
            <ReadingsHistory 
              readings={readingsForHistory} 
            />
          </FadeUpItem>
        </StaggerContainer>

        <AddReadingForm 
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          onSubmit={handleAddReading}
        />
      </div>
    </PageTransition>
  );
};

export default PatientDetail;

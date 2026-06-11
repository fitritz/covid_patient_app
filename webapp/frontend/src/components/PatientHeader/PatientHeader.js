import React, { useState, useRef } from 'react';
import { getApiUrl } from '../../config/api';
import './PatientHeader.css';

const PatientHeader = ({ patient, onPhotoUpdate, onPatientUpdate }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState('');
  const [editAge, setEditAge] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const fileInputRef = useRef(null);

  if (!patient) return null;

  const getInitials = (name) => {
    if (!name) return '';
    const names = name.split(' ');
    if (names.length > 1) {
      return `${names[0][0]}${names[1][0]}`;
    }
    return names[0][0];
  };

  const handlePhotoClick = () => {
    fileInputRef.current?.click();
  };

  const handleEditClick = () => {
    setEditName(patient.name || '');
    setEditAge(patient.age || '');
    setIsEditing(true);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditName('');
    setEditAge('');
  };

  const handleSaveEdit = async () => {
    if (!editName.trim()) {
      alert('Name is required');
      return;
    }
    if (!editAge || editAge < 1 || editAge > 120) {
      alert('Age must be between 1 and 120');
      return;
    }

    setIsSaving(true);
    try {
      const response = await fetch(getApiUrl(`/api/patients/${patient._id}`), {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          name: editName.trim(), 
          age: parseInt(editAge, 10) 
        }),
      });

      if (response.ok) {
        const updatedPatient = await response.json();
        if (onPatientUpdate) {
          onPatientUpdate(updatedPatient);
        }
        setIsEditing(false);
      } else {
        alert('Failed to update patient details');
      }
    } catch (error) {
      console.error('Error updating patient:', error);
      alert('Error updating patient details');
    } finally {
      setIsSaving(false);
    }
  };

  // Image compression utility
  const compressImage = (file, maxSizeKB = 500) => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          let { width, height } = img;
          
          const maxDim = 800;
          if (width > height && width > maxDim) {
            height = (height * maxDim) / width;
            width = maxDim;
          } else if (height > maxDim) {
            width = (width * maxDim) / height;
            height = maxDim;
          }
          
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, width, height);
          
          let quality = 0.8;
          let result = canvas.toDataURL('image/jpeg', quality);
          
          while (result.length > maxSizeKB * 1024 && quality > 0.1) {
            quality -= 0.1;
            result = canvas.toDataURL('image/jpeg', quality);
          }
          
          resolve(result);
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    });
  };

  const handlePhotoChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    setIsUploading(true);

    try {
      // Compress image automatically
      const photoData = await compressImage(file, 500);
      
      const response = await fetch(getApiUrl(`/api/patients/${patient._id}/photo`), {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ photo: photoData }),
      });

      if (response.ok) {
        if (onPhotoUpdate) {
          onPhotoUpdate(photoData);
        }
      } else {
        alert('Failed to update photo');
      }
    } catch (error) {
      console.error('Error updating photo:', error);
      alert('Error updating photo');
    } finally {
      setIsUploading(false);
    }
  };

  const renderPatientDetails = () => {
    const details = [];
    
    if (patient.patient_id) details.push(`ID: ${patient.patient_id}`);
    if (patient.age) details.push(`${patient.age} years`);
    if (patient.gender) details.push(patient.gender);
    if (patient.contact) details.push(patient.contact);

    return details.map((detail, index) => (
      <React.Fragment key={index}>
        <span>{detail}</span>
        {index < details.length - 1 && <span>&bull;</span>}
      </React.Fragment>
    ));
  };

  return (
    <div className="patient-header-card">
      <div 
        className={`patient-avatar ${patient.photo ? 'has-photo' : ''} ${isUploading ? 'uploading' : ''}`}
        onClick={handlePhotoClick}
        title="Click to update photo"
      >
        {patient.photo ? (
          <img src={patient.photo} alt={patient.name} className="avatar-image" />
        ) : (
          <span>{getInitials(patient.name)}</span>
        )}
        <div className="avatar-overlay">
          <span className="overlay-icon">📷</span>
        </div>
        {isUploading && <div className="upload-spinner" />}
      </div>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handlePhotoChange}
        accept="image/*"
        style={{ display: 'none' }}
      />
      <div className="patient-info">
        {isEditing ? (
          <div className="edit-form">
            <div className="edit-field">
              <label>Name</label>
              <input
                type="text"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="Patient name"
                maxLength="50"
              />
            </div>
            <div className="edit-field">
              <label>Age</label>
              <input
                type="number"
                value={editAge}
                onChange={(e) => setEditAge(e.target.value)}
                placeholder="Age"
                min="1"
                max="120"
              />
            </div>
            <div className="edit-actions">
              <button 
                className="btn-save" 
                onClick={handleSaveEdit}
                disabled={isSaving}
              >
                {isSaving ? 'Saving...' : 'Save'}
              </button>
              <button 
                className="btn-cancel" 
                onClick={handleCancelEdit}
                disabled={isSaving}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="name-row">
              <h1>{patient.name}</h1>
              <button className="edit-btn" onClick={handleEditClick} title="Edit patient details">
                ✏️
              </button>
            </div>
            <p>
              {renderPatientDetails()}
            </p>
          </>
        )}
      </div>
    </div>
  );
};

export default PatientHeader;

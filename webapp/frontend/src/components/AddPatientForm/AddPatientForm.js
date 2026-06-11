import React, { useState, useRef } from 'react';
import './AddPatientForm.css';

// Image compression utility
const compressImage = (file, maxSizeKB = 500) => {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        let { width, height } = img;
        
        // Calculate new dimensions (max 800px)
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
        
        // Start with quality 0.8 and reduce if needed
        let quality = 0.8;
        let result = canvas.toDataURL('image/jpeg', quality);
        
        // Reduce quality until under size limit
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

const AddPatientForm = ({ onAddPatient }) => {
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [photo, setPhoto] = useState(null);
  const [photoPreview, setPhotoPreview] = useState(null);
  const [errors, setErrors] = useState({});
  const [isCompressing, setIsCompressing] = useState(false);
  const fileInputRef = useRef(null);

  const handlePhotoChange = async (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        setErrors(prev => ({ ...prev, photo: 'Please select an image file' }));
        return;
      }
      
      setIsCompressing(true);
      setErrors(prev => ({ ...prev, photo: null }));
      
      try {
        // Compress image automatically
        const compressedImage = await compressImage(file, 500);
        setPhoto(compressedImage);
        setPhotoPreview(compressedImage);
      } catch (err) {
        setErrors(prev => ({ ...prev, photo: 'Failed to process image' }));
      } finally {
        setIsCompressing(false);
      }
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!name.trim()) {
      newErrors.name = 'Name is required';
    } else if (name.trim().length < 2) {
      newErrors.name = 'Name must be at least 2 characters';
    }
    
    if (!age) {
      newErrors.age = 'Age is required';
    } else if (age < 1 || age > 120) {
      newErrors.age = 'Age must be between 1 and 120';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!validateForm()) {
      return;
    }
    const patientData = { 
      name: name.trim(), 
      age: parseInt(age, 10) 
    };
    if (photo) {
      patientData.photo = photo;
    }
    onAddPatient(patientData);
    setName('');
    setAge('');
    setPhoto(null);
    setPhotoPreview(null);
    setErrors({});
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="add-patient-section">
      <h3 className="section-title">Add New Patient</h3>
      <form onSubmit={handleSubmit} className="patient-form">
        <div className="form-row">
          <div className="photo-upload-field">
            <div 
              className={`photo-upload-area ${photoPreview ? 'has-photo' : ''} ${isCompressing ? 'compressing' : ''}`}
              onClick={triggerFileInput}
            >
              {isCompressing ? (
                <div className="compress-spinner"></div>
              ) : photoPreview ? (
                <img src={photoPreview} alt="Preview" className="photo-preview" />
              ) : (
                <div className="photo-placeholder">
                  <span className="upload-icon">📷</span>
                  <span className="upload-text">Photo</span>
                </div>
              )}
            </div>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handlePhotoChange}
              accept="image/*"
              style={{ display: 'none' }}
            />
            {errors.photo && <span className="error-text">{errors.photo}</span>}
          </div>
          <div className="form-field">
            <input
              type="text"
              placeholder="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className={errors.name ? 'error' : ''}
              maxLength="50"
            />
            {errors.name && <span className="error-text">{errors.name}</span>}
          </div>
          <div className="form-field">
            <input
              type="number"
              placeholder="Age"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              className={errors.age ? 'error' : ''}
              min="1"
              max="120"
            />
            {errors.age && <span className="error-text">{errors.age}</span>}
          </div>
          <button type="submit" className="add-btn">Add Patient</button>
        </div>
      </form>
    </div>
  );
};

export default AddPatientForm;

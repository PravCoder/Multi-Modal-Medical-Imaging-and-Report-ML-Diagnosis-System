import React, { useEffect, useState, useRef } from "react";
import axios from "axios";

import '../styles/home.css';

const HomePage = () => {
  const [imagePreview, setImagePreview] = useState(null);
  const [patientDetails, setPatientDetails] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const colorPalette = {
    black: '#000000',
    darkBlue: '#14213d',
    orange: '#fca311',
    lightGray: '#e5e5e5',
    white: '#ffffff'
  };

  const diseases = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture'
  ];

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    
    // Simulate API call - replace with actual API integration
    setTimeout(() => {
      const mockResults = {
        diseases: diseases.map(disease => ({
          name: disease,
          probability: Math.random() * 100
        })),
        report: {
          findings: "The lungs are clear with no focal consolidation, effusion, or pneumothorax. The cardiomediastinal silhouette is within normal limits. No acute bony abnormalities.",
          impression: "No acute cardiopulmonary process."
        }
      };
      
      setResults(mockResults);
      setLoading(false);
    }, 2000);
  };

  const getProgressBarColor = (percentage) => {
    if (percentage >= 70) return '#ff4d4d';
    if (percentage >= 40) return '#fca311';
    return '#4caf50';
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="app" style={{ backgroundColor: colorPalette.white, color: colorPalette.black }}>
      <header className="app-header" style={{ backgroundColor: colorPalette.darkBlue, color: colorPalette.orange }}>
        <h1>Chest X-ray AI Diagnosis System</h1>
      </header>

      <main className="main-content">
        <form onSubmit={handleSubmit} className="input-form">
          <div className="input-section">
            <h2 style={{ color: colorPalette.darkBlue }}>Upload Chest X-ray</h2>
            <div className="image-upload-area" onClick={triggerFileInput}>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageUpload}
                accept="image/*"
                style={{ display: 'none' }}
              />
              {imagePreview ? (
                <img src={imagePreview} alt="X-ray preview" className="image-preview" />
              ) : (
                <div className="upload-placeholder">
                  <p>Click to upload X-ray image</p>
                </div>
              )}
            </div>
          </div>

          <div className="input-section">
            <h2 style={{ color: colorPalette.darkBlue }}>Patient Details</h2>
            <textarea
              value={patientDetails}
              onChange={(e) => setPatientDetails(e.target.value)}
              placeholder="Enter patient history, symptoms, and other relevant information..."
              className="patient-details-textarea"
              rows="6"
            />
          </div>

          <button 
            type="submit" 
            className="submit-button"
            disabled={!imagePreview || loading}
            style={{ 
              backgroundColor: colorPalette.orange,
              color: colorPalette.darkBlue
            }}
          >
            {loading ? 'Processing...' : 'Generate Diagnosis'}
          </button>
        </form>

        {results && (
          <div className="results-section">
            <h2 style={{ color: colorPalette.darkBlue }}>Diagnosis Results</h2>
            
            <div className="disease-predictions">
              <h3 style={{ color: colorPalette.darkBlue }}>Disease Probabilities</h3>
              {results.diseases.map((disease, index) => (
                <div key={index} className="disease-item">
                  <div className="disease-label">
                    <span>{disease.name}</span>
                    <span>{disease.probability.toFixed(2)}%</span>
                  </div>
                  <div className="progress-bar-container">
                    <div 
                      className="progress-bar" 
                      style={{ 
                        width: `${disease.probability}%`,
                        backgroundColor: getProgressBarColor(disease.probability)
                      }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>

            <div className="report-section">
              <h3 style={{ color: colorPalette.darkBlue }}>Radiology Report</h3>
              <div className="report-content">
                <div className="report-subsection">
                  <h4 style={{ color: colorPalette.darkBlue }}>Findings + Impression</h4>
                  <p>{results.report.findings}</p>
                </div>
                {/* <div className="report-subsection">
                  <h4 style={{ color: colorPalette.darkBlue }}>Impression</h4>
                  <p>{results.report.impression}</p>
                </div> */}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default HomePage;




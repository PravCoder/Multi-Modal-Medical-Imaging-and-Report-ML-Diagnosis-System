import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import "../styles/home.css";
import { api } from "../api";


const HomePage = () => {
  const [imagePreview, setImagePreview] = useState(null);
  const [file, setFile] = useState(null);
  const [patientDetails, setPatientDetails] = useState("");
  const [results, setResults] = useState(null); // { diseases: [...], report_text: "..." }
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const fileInputRef = useRef(null);

  const [loadingSample, setLoadingSample] = useState(false);

  const colorPalette = {
    black: "#000000",
    darkBlue: "#14213d",
    orange: "#fca311",
    lightGray: "#e5e5e5",
    white: "#ffffff",
  };

  const getProgressBarColor = (percentage) => {
    if (percentage >= 70) return "#ff4d4d";
    if (percentage >= 40) return "#fca311";
    return "#4caf50";
  };

  const handleImageUpload = (event) => {
    const f = event.target.files?.[0];
    if (!f) return;
    setFile(f);
    const reader = new FileReader();
    reader.onloadend = () => setImagePreview(reader.result);
    reader.readAsDataURL(f);
  };

  const triggerFileInput = () => fileInputRef.current?.click();

  async function base64ToFile(base64, mime, filename) {
    const res = await fetch(`data:${mime};base64,${base64}`);
    const blob = await res.blob();
    return new File([blob], filename, { type: mime });
  }


  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setErrorMsg("");
    setResults(null);

    try {
        if (!file) throw new Error("Please upload an X-ray image first.");

        const formData = new FormData();
        formData.append("image", file);
        formData.append("patient_details", patientDetails);

        // IMPORTANT: URL must match Django exactly. If you used path("api/predict/", ...),
        // call "/api/predict/" with the trailing slash.
        const { data } = await api.post("/api/predict/", formData, {
        headers: { "Content-Type": "multipart/form-data" }, // axios sets it, but explicit is fine
        });

        // Backend shape: { diseases: [{ name, probability }], report_text: "..." }
        setResults(data);
    } catch (e) {
        // Prefer server-provided error message if present
        const msg =
        e.response?.data?.error ||
        e.response?.data?.detail ||
        e.message ||
        "Something went wrong.";
        setErrorMsg(msg);
    } finally {
        setLoading(false);
    }
  };


  const handleLoadSample = async () => {
    setLoadingSample(true);
    setErrorMsg("");
    try {
      const { data } = await api.post("/api/load-sample/"); // POST with no body
      // data: { image_name, image_mime, image_base64, patient_details }

      // Convert to File so your existing inference flow works
      const f = await base64ToFile(data.image_base64, data.image_mime, data.image_name);
      setFile(f);

      // Set preview
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result);
      reader.readAsDataURL(f);

      // Fill patient details textbox
      setPatientDetails(data.patient_details || "");
    } catch (e) {
      const msg =
        e.response?.data?.error ||
        e.response?.data?.detail ||
        e.message ||
        "Failed to load sample.";
      setErrorMsg(msg);
    } finally {
      setLoadingSample(false);
    }
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
            <div
              className="image-upload-area"
              onClick={triggerFileInput}
              style={{ cursor: "pointer", border: "1px dashed #ccc", padding: 16, borderRadius: 8 }}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageUpload}
                accept="image/*"
                style={{ display: "none" }}
              />
              {imagePreview ? (
                <img
                  src={imagePreview}
                  alt="X-ray preview"
                  className="image-preview"
                  style={{ maxWidth: "100%", borderRadius: 8 }}
                />
              ) : (
                <div className="upload-placeholder" style={{ textAlign: "center" }}>
                  <p>Click to upload X-ray image</p>
                </div>
              )}
            </div>
          </div>

          <div className="input-section" style={{ marginTop: 24 }}>
            <h2 style={{ color: colorPalette.darkBlue }}>Patient Details</h2>
            <textarea
              value={patientDetails}
              onChange={(e) => setPatientDetails(e.target.value)}
              placeholder="Enter patient details here: Age and sex, presenting symptoms (onset, duration, severity), relevant medical history (e.g. heart disease, COPD, cancer, smoking), recent surgeries, hospitalizations, or travel, current medications or treatments (oxygen, chemo, etc.), vital signs if available (temp, HR, BP, RR, SpO2), specific question for radiology (e.g. rule out pneumonia, effusion, fracture)
"
              className="patient-details-textarea"
              rows="6"
              style={{ width: "100%", padding: 12, borderRadius: 8, border: "1px solid #ccc" }}
            />
          </div>

          <button
            type="submit"
            className="submit-button"
            disabled={!file || loading}
            style={{
              marginTop: 16,
              padding: "12px 16px",
              borderRadius: 10,
              border: "none",
              cursor: !file || loading ? "not-allowed" : "pointer",
              backgroundColor: colorPalette.orange,
              color: colorPalette.darkBlue,
              fontWeight: 700,
            }}
          >
            {loading ? "Processing..." : "Generate Diagnosis"}
          </button>
        </form>

        <button
          type="button"
          onClick={handleLoadSample}
          disabled={loading || loadingSample}
          style={{
            marginTop: 8,
            marginRight: 12,
            padding: "10px 14px",
            borderRadius: 10,
            border: "1px solid #ccc",
            background: loadingSample ? "#eee" : "#fafafa",
            cursor: loadingSample ? "not-allowed" : "pointer",
          }}
        >
          {loadingSample ? "Loading sample..." : "Load random sample"}
        </button>


        {errorMsg && (
          <div style={{ marginTop: 16, color: "#b00020" }}>
            {errorMsg}
          </div>
        )}

        {results && (
          <div className="results-section" style={{ marginTop: 32 }}>
            <h2 style={{ color: colorPalette.darkBlue }}>Diagnosis Results</h2>

            <div className="disease-predictions" style={{ marginTop: 16 }}>
              <h3 style={{ color: colorPalette.darkBlue }}>Disease Probabilities</h3>
              {(results.diseases || []).map((disease, index) => {
                const pct = Number(disease.probability) || 0;
                const clamped = Math.min(100, Math.max(0, pct));
                return (
                  <div key={index} className="disease-item" style={{ marginBottom: 12 }}>
                    <div
                      className="disease-label"
                      style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}
                    >
                      <span>{disease.name}</span>
                      <span>{clamped.toFixed(2)}%</span>
                    </div>
                    <div
                      className="progress-bar-container"
                      style={{
                        width: "100%",
                        height: 10,
                        backgroundColor: colorPalette.lightGray,
                        borderRadius: 6,
                        overflow: "hidden",
                      }}
                    >
                      <div
                        className="progress-bar"
                        style={{
                          width: `${clamped}%`,
                          height: "100%",
                          backgroundColor: getProgressBarColor(clamped),
                          transition: "width 300ms ease",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="report-section" style={{ marginTop: 24 }}>
              <h3 style={{ color: colorPalette.darkBlue }}>Radiology Report</h3>
              <div
                className="report-content"
                style={{ background: "#fafafa", border: "1px solid #eee", borderRadius: 8, padding: 16 }}
              >
                <div className="report-subsection">
                  <h4 style={{ color: colorPalette.darkBlue, marginTop: 0 }}>Findings + Impression</h4>
                  <p style={{ whiteSpace: "pre-wrap", marginBottom: 0 }}>
                    {results.report_text || ""}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default HomePage;

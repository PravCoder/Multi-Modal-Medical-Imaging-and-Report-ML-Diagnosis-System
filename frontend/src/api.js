import axios from "axios";

// If your API is on a different origin/port, set it here, e.g.:
// const API_BASE = "http://localhost:8000";

const apiUrl = "/choreo-apis/multimodalmedicalapp/backend/v1";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ? import.meta.env.VITE_API_URL : apiUrl,
});
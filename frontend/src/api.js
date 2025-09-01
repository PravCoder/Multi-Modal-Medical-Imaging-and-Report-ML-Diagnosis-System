import axios from "axios";

// If your API is on a different origin/port, set it here, e.g.:
// const API_BASE = "http://localhost:8000";
const API_BASE = "http://127.0.0.1:8000/"; // same origin (proxy or served together)

export const api = axios.create({
  baseURL: API_BASE,
  // withCredentials: true, // enable if you use cookie auth/CSRF
});

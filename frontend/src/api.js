import axios from "axios";

const apiUrl = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL
  : "/choreo-apis/multimodalmedicalapp/backend/v1";

export const api = axios.create({
  baseURL: apiUrl,
  // no need for Authorization header in frontend
});
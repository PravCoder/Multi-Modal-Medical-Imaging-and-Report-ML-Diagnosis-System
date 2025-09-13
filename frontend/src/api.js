import axios from "axios";

// for locally working:
// const apiUrl = import.meta.env.VITE_API_URL
//   ? import.meta.env.VITE_API_URL
//   : "/choreo-apis/multimodalmedicalapp/backend/v1";


// for production working:
const apiUrl = window?.configs?.apiUrl ? window.configs.apiUrl : "/";

export const api = axios.create({
  baseURL: apiUrl,
  // no need for Authorization header in frontend
});
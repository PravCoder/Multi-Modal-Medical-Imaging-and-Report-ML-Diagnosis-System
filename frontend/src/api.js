// import axios from "axios";

// // for locally working:
// // const apiUrl = import.meta.env.VITE_API_URL
// //   ? import.meta.env.VITE_API_URL
// //   : "/choreo-apis/multimodalmedicalapp/backend/v1";


// // for production only:
// const apiUrl = window?.configs?.apiUrl ? window.configs.apiUrl : "/";

// export const api = axios.create({
//   baseURL: apiUrl,
//   // no need for Authorization header in frontend
// });

// frontend/src/api.js
import axios from "axios";

// Local dev
const LOCAL_API_URL = import.meta.env.VITE_API_URL;

// Choreo injected configs
const CHOREO_CONFIGS = window?.configs;

let cachedToken = null;
let tokenExpiry = null;

// Get OAuth2 token from Choreo
async function getChoreoToken() {
  if (cachedToken && tokenExpiry && Date.now() < tokenExpiry) {
    return cachedToken;
  }

  const { apiUrl, consumerKey, consumerSecret, tokenUrl } = CHOREO_CONFIGS;

  const basicAuth = btoa(`${consumerKey}:${consumerSecret}`);

  const tokenResponse = await axios.post(
    tokenUrl,
    new URLSearchParams({ grant_type: "client_credentials" }),
    {
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        Authorization: `Basic ${basicAuth}`,
      },
    }
  );

  cachedToken = tokenResponse.data.access_token;
  tokenExpiry = Date.now() + (tokenResponse.data.expires_in - 10) * 1000;

  return cachedToken;
}

// Return Axios instance
export async function getApi() {
  if (LOCAL_API_URL) {
    return axios.create({ baseURL: LOCAL_API_URL });
  }

  if (CHOREO_CONFIGS) {
    const token = await getChoreoToken();
    return axios.create({
      baseURL: CHOREO_CONFIGS.apiUrl,
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
  }

  throw new Error("No API URL configured!");
}

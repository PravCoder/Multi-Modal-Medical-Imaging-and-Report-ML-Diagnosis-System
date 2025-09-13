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

// Detect if we are running locally (VITE_API_URL) or on Choreo
const LOCAL_API_URL = import.meta.env.VITE_API_URL;

// Check if Choreo injected configs
const CHOREO_CONFIGS = window?.configs;

let cachedToken = null;
let tokenExpiry = null;

// 1️⃣ Get OAuth2 token from Choreo if needed
async function getChoreoToken() {
  if (cachedToken && tokenExpiry && Date.now() < tokenExpiry) {
    return cachedToken; // reuse token if not expired
  }

  const { apiUrl, consumerKey, consumerSecret, tokenUrl } = CHOREO_CONFIGS;

  const tokenResponse = await axios.post(
    tokenUrl,
    new URLSearchParams({
      grant_type: "client_credentials",
      client_id: consumerKey,
      client_secret: consumerSecret,
    }),
    {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    }
  );

  cachedToken = tokenResponse.data.access_token;
  tokenExpiry = Date.now() + (tokenResponse.data.expires_in - 10) * 1000; // 10s buffer

  return cachedToken;
}

// 2️⃣ Get an Axios instance
export async function getApi() {
  if (LOCAL_API_URL) {
    // local dev: no auth
    return axios.create({
      baseURL: LOCAL_API_URL,
    });
  }

  if (CHOREO_CONFIGS) {
    // deployed on Choreo: use OAuth2 token
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

const DEFAULT_PRODUCTION_API_URL = 'https://covid-patient-api.vercel.app';

const API_BASE_URL =
  process.env.REACT_APP_API_URL ||
  (process.env.NODE_ENV === 'development' ? '' : DEFAULT_PRODUCTION_API_URL);

export const getApiUrl = (path) => {
  return `${API_BASE_URL}${path}`;
};

export default API_BASE_URL;

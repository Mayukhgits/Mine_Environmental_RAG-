// Backend origin. Override with VITE_API_URL in frontend/.env (e.g. VITE_API_URL=http://192.168.1.10:8000)
export const API_ORIGIN = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');
export const API_BASE = `${API_ORIGIN}/api`;

/** Turns "/static/extracted_images/x.png" into a full, loadable URL. */
export const assetUrl = (path) => {
  if (!path) return '';
  if (/^(https?:|data:|blob:)/i.test(path)) return path;
  return `${API_ORIGIN}/${String(path).replace(/^\/+/, '')}`;
};

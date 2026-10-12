import axios from 'axios';
import { API_BASE } from './config';

const http = axios.create({ baseURL: API_BASE, timeout: 30_000 });

/** Converts any axios/FastAPI error into a message that is safe to show to a user. */
export function getErrorMessage(err, fallback = 'Something went wrong.') {
  if (axios.isCancel(err)) return 'Request cancelled.';
  if (err?.code === 'ECONNABORTED') return 'The request timed out. The server may still be working - try again shortly.';
  if (!err?.response) return `Cannot reach the backend at ${API_BASE}. Is it running?`;
  const detail = err.response.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join('; '); // FastAPI validation errors
  return fallback;
}

export const fetchDocuments = async () => {
  const res = await http.get('/documents');
  return res.data.documents ?? [];
};

/** onProgress(percent | null) - upload progress only; indexing time comes after 100%. */
export const uploadPDF = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await http.post('/upload', formData, {
    timeout: 15 * 60 * 1000, // big PDFs take a while to index
    onUploadProgress: (e) => onProgress?.(e.total ? Math.round((e.loaded * 100) / e.total) : null),
  });
  return res.data;
};

export const deleteDocument = async (docHash) => {
  const res = await http.delete(`/documents/${docHash}`);
  return res.data;
};

/** Field names match the backend: message / history / documents (doc hashes or file names). */
export const sendChatMessage = async (message, history = [], documents = [], signal) => {
  const res = await http.post('/chat', { message, history, documents }, { timeout: 180_000, signal });
  return res.data; // { response, sources, meta }
};

export const fetchHealth = async () => {
  try {
    const res = await http.get('/health', { timeout: 8000 });
    return { online: true, llm: !!res.data.llm_configured };
  } catch (err) {
    // 503 = backend is up but Qdrant is not
    return { online: false, llm: !!err?.response?.data?.llm_configured };
  }
};

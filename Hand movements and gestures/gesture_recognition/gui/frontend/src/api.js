const API_BASE = '/api';

// Helper function for handling API errors
async function handleResponse(response) {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || errorData.message || 'API request failed');
  }
  return response.json();
}

export const fetchStatus = async () => {
  const res = await fetch(`${API_BASE}/status`);
  return await handleResponse(res);
};

export const fetchGestures = async () => {
  const res = await fetch(`${API_BASE}/gestures`);
  return await handleResponse(res);
};

export const fetchPredictionHistory = async () => {
  const res = await fetch(`${API_BASE}/history`);
  return await handleResponse(res);
};

export const startRecording = async (label) => {
  const res = await fetch(`${API_BASE}/record/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label })
  });
  return await handleResponse(res);
};

export const stopRecording = async () => {
  const res = await fetch(`${API_BASE}/record/stop`, { method: 'POST' });
  return await handleResponse(res);
};

export const startTraining = async () => {
  const res = await fetch(`${API_BASE}/train`, { method: 'POST' });
  return await handleResponse(res);
};

export const addGesture = async (label) => {
  const res = await fetch(`${API_BASE}/gestures`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label })
  });
  return await handleResponse(res);
};

export const deleteGesture = async (label) => {
  const res = await fetch(`${API_BASE}/gestures`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label })
  });
  return await handleResponse(res);
};

export const fetchConfig = async () => {
  const res = await fetch(`${API_BASE}/config`);
  return await handleResponse(res);
};

export const updateConfig = async (config) => {
  const res = await fetch(`${API_BASE}/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  });
  return await handleResponse(res);
};

export const exportDataset = async () => {
  const res = await fetch(`${API_BASE}/dataset/export`, { method: 'POST' });
  return await handleResponse(res);
};

export const resetDataset = async () => {
  const res = await fetch(`${API_BASE}/dataset/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  return await handleResponse(res);
};

export const clearPredictionHistory = async () => {
  const res = await fetch(`${API_BASE}/history`, { method: 'DELETE' });
  return await handleResponse(res);
};

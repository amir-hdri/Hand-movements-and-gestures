const API_BASE = '/api';

export const fetchStatus = async () => {
    const res = await fetch(`${API_BASE}/status`);
    return await res.json();
};

export const fetchGestures = async () => {
    const res = await fetch(`${API_BASE}/gestures`);
    return await res.json();
};

export const startRecording = async (label) => {
    const res = await fetch(`${API_BASE}/record/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label })
    });
    return await res.json();
};

export const stopRecording = async () => {
    const res = await fetch(`${API_BASE}/record/stop`, { method: 'POST' });
    return await res.json();
};

export const startTraining = async () => {
    const res = await fetch(`${API_BASE}/train`, { method: 'POST' });
    return await res.json();
};

export const addGesture = async (label) => {
    const res = await fetch(`${API_BASE}/gestures`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label })
    });
    return await res.json();
};

export const fetchConfig = async () => {
    const res = await fetch(`${API_BASE}/config`);
    return await res.json();
};

export const updateConfig = async (thresholds) => {
    const res = await fetch(`${API_BASE}/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thresholds })
    });
    return await res.json();
};

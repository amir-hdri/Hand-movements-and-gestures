import React, { useState, useEffect } from 'react';
import { Typography, Box, Button, Grid, Paper, FormControlLabel, Switch, Select, MenuItem } from '@mui/material';
import axios from 'axios';

const LiveInference = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [robotEnabled, setRobotEnabled] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [status, setStatus] = useState("Idle");

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = () => {
    axios.get('/api/models')
      .then(res => {
        setModels(res.data.models);
        if (res.data.models.length > 0) setSelectedModel(res.data.models[0]);
      })
      .catch(err => console.error(err));
  };

  const startInference = () => {
    if (!selectedModel) return alert("Select a model first");
    axios.post('/api/inference/start', { model: selectedModel, robot: robotEnabled })
      .then(() => {
        setIsRunning(true);
        setStatus("Running...");
      })
      .catch(err => alert("Error: " + err.message));
  };

  const stopInference = () => {
    axios.post('/api/inference/stop')
      .then(() => {
        setIsRunning(false);
        setStatus("Stopped");
      })
      .catch(err => alert("Error: " + err.message));
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Live Inference</Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: '500px', display: 'flex', justifyContent: 'center', alignItems: 'center', bgcolor: 'black' }}>
            <img
              src="/api/video_feed"
              alt="Video Feed"
              style={{ maxWidth: '100%', maxHeight: '100%' }}
            />
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6">Configuration</Typography>
            <Select
              fullWidth
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              displayEmpty
            >
              <MenuItem value="" disabled>Select Model</MenuItem>
              {models.map((m) => (
                <MenuItem key={m} value={m}>{m}</MenuItem>
              ))}
            </Select>
            <FormControlLabel
              control={<Switch checked={robotEnabled} onChange={(e) => setRobotEnabled(e.target.checked)} />}
              label="Enable Robot Control"
              sx={{ mt: 2 }}
            />
            <Box mt={2}>
              {!isRunning ? (
                <Button variant="contained" color="primary" fullWidth onClick={startInference}>Start Inference</Button>
              ) : (
                <Button variant="contained" color="secondary" fullWidth onClick={stopInference}>Stop Inference</Button>
              )}
            </Box>
            <Box mt={2}>
              <Typography variant="body1">Status: {status}</Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default LiveInference;

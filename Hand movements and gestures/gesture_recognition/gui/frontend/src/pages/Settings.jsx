import React, { useState, useEffect } from 'react';
import { Typography, Box, Button, Grid, Paper, Select, MenuItem } from '@mui/material';
import axios from 'axios';

const Settings = () => {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(0);

  useEffect(() => {
    // Ideally fetch available cameras from backend or use navigator.mediaDevices
    // For now, simple manual selection or hardcoded
    setCameras([0, 1, 2]);
  }, []);

  const saveSettings = () => {
    // Send settings to backend
    axios.post('/api/settings', { camera: selectedCamera })
      .then(() => alert("Settings saved"))
      .catch(err => alert("Error saving settings"));
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Settings</Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6">Hardware</Typography>
            <Select
              fullWidth
              value={selectedCamera}
              onChange={(e) => setSelectedCamera(e.target.value)}
              sx={{ mt: 2 }}
            >
              {cameras.map(c => <MenuItem key={c} value={c}>Camera {c}</MenuItem>)}
            </Select>
            <Button variant="contained" color="primary" onClick={saveSettings} sx={{ mt: 2 }}>Save</Button>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Settings;

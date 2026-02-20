import React, { useState, useEffect } from 'react';
import { Typography, Box, TextField, Button, Grid, Paper } from '@mui/material';
import axios from 'axios';

const DataCollection = () => {
  const [actionName, setActionName] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState("Idle");

  const startRecording = () => {
    if (!actionName) return alert("Please enter an action name");
    // Send label_index as 0 for now, or manage it properly
    axios.post('/api/dataset/record', { action: actionName, start: true, label_index: 0 })
      .then(() => {
        setIsRecording(true);
        setStatus(`Recording ${actionName}...`);
      })
      .catch(err => alert("Error starting recording: " + err.message));
  };

  const stopRecording = () => {
    axios.post('/api/dataset/record', { start: false })
      .then(() => {
        setIsRecording(false);
        setStatus("Idle");
      })
      .catch(err => alert("Error stopping recording: " + err.message));
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Data Collection</Typography>
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
            <Typography variant="h6" gutterBottom>Controls</Typography>
            <TextField
              label="Action Name"
              fullWidth
              margin="normal"
              value={actionName}
              onChange={(e) => setActionName(e.target.value)}
              disabled={isRecording}
            />
            <Box mt={2}>
              {!isRecording ? (
                <Button variant="contained" color="primary" fullWidth onClick={startRecording}>Start Recording</Button>
              ) : (
                <Button variant="contained" color="secondary" fullWidth onClick={stopRecording}>Stop Recording</Button>
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

export default DataCollection;

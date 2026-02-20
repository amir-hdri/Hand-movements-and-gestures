import React, { useEffect, useState } from 'react';
import { Typography, Grid, Paper, Box } from '@mui/material';
import axios from 'axios';

const Dashboard = () => {
  const [status, setStatus] = useState("Loading...");

  useEffect(() => {
    axios.get('/api/status')
      .then(res => setStatus(res.data.status))
      .catch(err => setStatus("Error connecting to backend"));
  }, []);

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Dashboard</Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6} lg={4}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6">System Status</Typography>
            <Typography variant="body1">{status}</Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;

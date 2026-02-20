import React, { useState, useEffect } from 'react';
import { Typography, Box, Button, Grid, Paper, Checkbox, List, ListItem, ListItemText, ListItemButton, ListItemIcon } from '@mui/material';
import axios from 'axios';

const Training = () => {
  const [actions, setActions] = useState([]);
  const [selectedActions, setSelectedActions] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [status, setStatus] = useState("Idle");

  useEffect(() => {
    fetchActions();
  }, []);

  const fetchActions = () => {
    axios.get('/api/dataset/actions')
      .then(res => {
        setActions(res.data.actions);
      })
      .catch(err => console.error(err));
  };

  const handleToggle = (value) => () => {
    const currentIndex = selectedActions.indexOf(value);
    const newChecked = [...selectedActions];

    if (currentIndex === -1) {
      newChecked.push(value);
    } else {
      newChecked.splice(currentIndex, 1);
    }
    setSelectedActions(newChecked);
  };

  const startTraining = () => {
    if (selectedActions.length === 0) return alert("Select at least one action");
    setIsTraining(true);
    setStatus("Training started...");

    axios.post('/api/train', { actions: selectedActions })
      .then(res => {
        setIsTraining(false);
        setStatus(`Training complete. Accuracy: ${res.data.accuracy}`);
      })
      .catch(err => {
        setIsTraining(false);
        setStatus("Training failed: " + err.message);
      });
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Model Training</Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6">Select Actions</Typography>
            <List>
              {actions.map((action) => (
                <ListItem key={action} disablePadding>
                  <ListItemButton onClick={handleToggle(action)} dense>
                    <ListItemIcon>
                      <Checkbox
                        edge="start"
                        checked={selectedActions.indexOf(action) !== -1}
                        tabIndex={-1}
                        disableRipple
                      />
                    </ListItemIcon>
                    <ListItemText primary={action} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6">Controls</Typography>
            <Button
              variant="contained"
              color="primary"
              onClick={startTraining}
              disabled={isTraining}
              fullWidth
            >
              {isTraining ? "Training..." : "Start Training"}
            </Button>
            <Box mt={2}>
              <Typography variant="body1">Status: {status}</Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Training;

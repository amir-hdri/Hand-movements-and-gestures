import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Paper, Typography, Box, Button, TextField, Select, MenuItem,
  InputLabel, FormControl, Dialog, DialogTitle, DialogContent,
  DialogActions, IconButton, Tooltip, Chip, LinearProgress
} from '@mui/material';
import {
  Add, Delete, PlayArrow, Stop, School, FiberManualRecord,
  Check, Close, Edit, Save, RestartAlt
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import {
  fetchGestures, addGesture, deleteGesture, startRecording, 
  stopRecording, startTraining, resetDataset
} from '../api';

function ControlPanel({ status, gestures, onGesturesChange, isConnected }) {
  const { enqueueSnackbar } = useSnackbar();
  const [selectedGesture, setSelectedGesture] = useState('');
  const [openAddDialog, setOpenAddDialog] = useState(false);
  const [newGesture, setNewGesture] = useState('');
  const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
  const [gestureToDelete, setGestureToDelete] = useState('');
  const [localGestures, setLocalGestures] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // Load gestures
  const loadGestures = async () => {
    try {
      const data = await fetchGestures();
      setLocalGestures(data.gestures || []);
      if (onGesturesChange) onGesturesChange(data.gestures || []);
    } catch (e) {
      console.error("Failed to load gestures", e);
      enqueueSnackbar('Failed to load gestures: ' + e.message, { variant: 'error' });
    }
  };

  useEffect(() => {
    loadGestures();
  }, []);

  useEffect(() => {
    if (gestures && gestures.length > 0) {
      setLocalGestures(gestures);
    }
  }, [gestures]);

  // Handlers
  const handleAddGesture = async () => {
    if (!newGesture.trim()) {
      enqueueSnackbar('Gesture name cannot be empty', { variant: 'warning' });
      return;
    }
    
    if (localGestures.includes(newGesture.trim())) {
      enqueueSnackbar('Gesture already exists', { variant: 'warning' });
      return;
    }

    try {
      setIsLoading(true);
      await addGesture(newGesture.trim());
      setNewGesture('');
      setOpenAddDialog(false);
      loadGestures();
      enqueueSnackbar(`Gesture '${newGesture.trim()}' added successfully`, { variant: 'success' });
    } catch (e) {
      enqueueSnackbar('Failed to add gesture: ' + e.message, { variant: 'error' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteGesture = async () => {
    try {
      setIsLoading(true);
      await deleteGesture(gestureToDelete);
      setOpenDeleteDialog(false);
      setGestureToDelete('');
      loadGestures();
      
      // If we're deleting the currently selected gesture, clear selection
      if (selectedGesture === gestureToDelete) {
        setSelectedGesture('');
      }
      
      enqueueSnackbar(`Gesture '${gestureToDelete}' deleted`, { variant: 'success' });
    } catch (e) {
      enqueueSnackbar('Failed to delete gesture: ' + e.message, { variant: 'error' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleRecord = async () => {
    if (!isConnected) {
      enqueueSnackbar('Not connected to backend', { variant: 'error' });
      return;
    }

    if (status.mode === 'recording') {
      try {
        await stopRecording();
        enqueueSnackbar('Recording stopped', { variant: 'info' });
      } catch (e) {
        enqueueSnackbar('Failed to stop recording: ' + e.message, { variant: 'error' });
      }
    } else {
      if (!selectedGesture) {
        enqueueSnackbar('Please select a gesture first', { variant: 'warning' });
        return;
      }
      try {
        await startRecording(selectedGesture);
        enqueueSnackbar(`Started recording for '${selectedGesture}'`, { variant: 'success' });
      } catch (e) {
        enqueueSnackbar('Failed to start recording: ' + e.message, { variant: 'error' });
      }
    }
  };

  const handleTrain = async () => {
    if (!isConnected) {
      enqueueSnackbar('Not connected to backend', { variant: 'error' });
      return;
    }

    if (localGestures.length === 0) {
      enqueueSnackbar('No gestures available. Add gestures first.', { variant: 'warning' });
      return;
    }

    try {
      setIsLoading(true);
      await startTraining();
      enqueueSnackbar('Training started. This may take a while...', { variant: 'info', autoHideDuration: 5000 });
    } catch (e) {
      enqueueSnackbar('Training failed: ' + e.message, { variant: 'error' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetDataset = async () => {
    if (!isConnected) {
      enqueueSnackbar('Not connected to backend', { variant: 'error' });
      return;
    }

    try {
      await resetDataset();
      enqueueSnackbar('Dataset reset successfully', { variant: 'success' });
      loadGestures();
    } catch (e) {
      enqueueSnackbar('Failed to reset dataset: ' + e.message, { variant: 'error' });
    }
  };

  const openDeleteConfirmation = (gesture) => {
    setGestureToDelete(gesture);
    setOpenDeleteDialog(true);
  };

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Box display="flex" alignItems="center" gap={1} mb={2}>
        <FiberManualRecord color={status.mode === 'recording' ? 'error' : 'action'} />
        <Typography variant="h6" flex={1}>
          Controls
        </Typography>
        {isLoading && <Chip label="Processing..." size="small" color="info" />}
      </Box>

      {/* Gesture Selection */}
      <FormControl fullWidth margin="normal" disabled={!isConnected}>
        <InputLabel>Select Gesture</InputLabel>
        <Select
          value={selectedGesture}
          label="Select Gesture"
          onChange={(e) => setSelectedGesture(e.target.value)}
          disabled={!isConnected}
        >
          {localGestures.length === 0 ? (
            <MenuItem disabled>
              <Typography color="text.secondary">No gestures available</Typography>
            </MenuItem>
          ) : (
            localGestures.map((g) => (
              <MenuItem key={g} value={g}>
                <Box display="flex" justifyContent="space-between" width="100%">
                  {g}
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      openDeleteConfirmation(g);
                    }}
                    color="error"
                    sx={{ p: 0.5 }}
                  >
                    <Delete fontSize="small" />
                  </IconButton>
                </Box>
              </MenuItem>
            ))
          )}
        </Select>
      </FormControl>

      {/* Add Gesture Button */}
      <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
        <Button
          variant="outlined"
          color="primary"
          onClick={() => setOpenAddDialog(true)}
          fullWidth
          disabled={!isConnected || isLoading}
          startIcon={<Add />}
        >
          Add Gesture
        </Button>
      </motion.div>

      <Box display="flex" gap={2} mt={2}>
        {/* Recording Button */}
        <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
          <Button
            variant="contained"
            color={status.mode === 'recording' ? 'error' : 'primary'}
            onClick={handleRecord}
            disabled={!isConnected || !selectedGesture || status.mode === 'training' || isLoading}
            fullWidth
            startIcon={status.mode === 'recording' ? <Stop /> : <PlayArrow />}
            sx={{ py: 1.5 }}
          >
            {status.mode === 'recording' ? 'Stop Recording' : 'Start Recording'}
          </Button>
        </motion.div>
      </Box>

      {/* Training Button */}
      <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
        <Button
          variant="contained"
          color="warning"
          onClick={handleTrain}
          disabled={!isConnected || status.mode !== 'idle' || status.training_status === 'training' || isLoading}
          fullWidth
          startIcon={<School />}
          sx={{ py: 1.5, mt: 1 }}
        >
          {status.training_status === 'training' ? 'Training...' : 'Train Model'}
        </Button>
      </motion.div>

      {/* Reset Dataset Button */}
      <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
        <Button
          variant="outlined"
          color="error"
          onClick={handleResetDataset}
          disabled={!isConnected || isLoading}
          fullWidth
          startIcon={<RestartAlt />}
          sx={{ py: 1.5, mt: 1 }}
        >
          Reset Dataset
        </Button>
      </motion.div>

      {/* Connection Warning */}
      {!isConnected && (
        <Box mt={2} p={2} bgcolor="error.background" borderRadius={1} textAlign="center">
          <Typography variant="body2" color="error">
            Backend connection lost. Please check if the server is running.
          </Typography>
        </Box>
      )}

      {/* Add Gesture Dialog */}
      <Dialog open={openAddDialog} onClose={() => setOpenAddDialog(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Add New Gesture</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Gesture Name"
            fullWidth
            value={newGesture}
            onChange={(e) => setNewGesture(e.target.value)}
            helperText="Enter a unique name for the gesture"
            onKeyPress={(e) => e.key === 'Enter' && handleAddGesture()}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenAddDialog(false)}>Cancel</Button>
          <Button 
            onClick={handleAddGesture} 
            variant="contained" 
            color="primary"
            disabled={!newGesture.trim()}
          >
            Add
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Gesture Dialog */}
      <Dialog 
        open={openDeleteDialog} 
        onClose={() => setOpenDeleteDialog(false)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Delete Gesture</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the gesture "{gestureToDelete}"?
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            This will also remove all recorded data for this gesture.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDeleteDialog(false)}>Cancel</Button>
          <Button 
            onClick={handleDeleteGesture} 
            variant="contained" 
            color="error"
            startIcon={<Delete />}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Loading Progress */}
      {isLoading && (
        <Box sx={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}>
          <Box sx={{ 
            position: 'absolute', 
            top: '50%', 
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '80%'
          }}>
            <LinearProgress color="secondary" />
          </Box>
        </Box>
      )}
    </Paper>
  );
}

export default ControlPanel;

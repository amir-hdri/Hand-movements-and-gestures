import React, { useState, useEffect } from 'react';
import {
    Paper, Typography, Box, Button, TextField, Select, MenuItem, InputLabel, FormControl,
    Dialog, DialogTitle, DialogContent, DialogActions
} from '@mui/material';
import { fetchGestures, addGesture, startRecording, stopRecording, startTraining } from '../api';

function ControlPanel({ status, onGesturesChange }) {
    const [gestures, setGestures] = useState([]);
    const [selectedGesture, setSelectedGesture] = useState('');
    const [openAddDialog, setOpenAddDialog] = useState(false);
    const [newGesture, setNewGesture] = useState('');

    // Refresh gestures
    const loadGestures = async () => {
        try {
            const data = await fetchGestures();
            setGestures(data.gestures);
            if (onGesturesChange) onGesturesChange(data.gestures);
        } catch (e) {
            console.error("Failed to load gestures", e);
        }
    };

    useEffect(() => {
        loadGestures();
    }, []);

    // Handlers
    const handleAddGesture = async () => {
        if (!newGesture.trim()) return;
        try {
            await addGesture(newGesture.trim());
            setNewGesture('');
            setOpenAddDialog(false);
            loadGestures();
        } catch (e) {
            console.error("Failed to add gesture", e);
        }
    };

    const handleRecord = async () => {
        if (status.mode === 'recording') {
            await stopRecording();
        } else {
            if (!selectedGesture) return;
            await startRecording(selectedGesture);
        }
    };

    const handleTrain = async () => {
        try {
            await startTraining();
            // Training happens in background, UI will update via status polling
        } catch (e) {
            console.error("Training failed", e);
        }
    };

    return (
        <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                Controls
            </Typography>

            {/* Gesture Selection */}
            <FormControl fullWidth margin="normal">
                <InputLabel>Select Gesture</InputLabel>
                <Select
                    value={selectedGesture}
                    label="Select Gesture"
                    onChange={(e) => setSelectedGesture(e.target.value)}
                >
                    {gestures.map((g) => (
                        <MenuItem key={g} value={g}>{g}</MenuItem>
                    ))}
                </Select>
            </FormControl>

            <Box display="flex" gap={2} mt={2}>
                <Button
                    variant="contained"
                    color="primary"
                    onClick={() => setOpenAddDialog(true)}
                    fullWidth
                >
                    Add New Gesture
                </Button>
            </Box>

            {/* Recording */}
            <Box mt={2}>
                <Button
                    variant="contained"
                    color={status.mode === 'recording' ? 'error' : 'success'}
                    onClick={handleRecord}
                    disabled={!selectedGesture || status.mode === 'training'}
                    fullWidth
                >
                    {status.mode === 'recording' ? 'Stop Recording' : 'Start Recording'}
                </Button>
            </Box>

            {/* Training */}
            <Box mt={2}>
                <Button
                    variant="contained"
                    color="warning"
                    onClick={handleTrain}
                    disabled={status.mode !== 'idle' || status.training_status === 'training'}
                    fullWidth
                >
                    {status.training_status === 'training' ? 'Training...' : 'Train Model'}
                </Button>
            </Box>

            {/* Add Gesture Dialog */}
            <Dialog open={openAddDialog} onClose={() => setOpenAddDialog(false)}>
                <DialogTitle>Add New Gesture Label</DialogTitle>
                <DialogContent>
                    <TextField
                        autoFocus
                        margin="dense"
                        label="Gesture Name"
                        fullWidth
                        value={newGesture}
                        onChange={(e) => setNewGesture(e.target.value)}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setOpenAddDialog(false)}>Cancel</Button>
                    <Button onClick={handleAddGesture} variant="contained">Add</Button>
                </DialogActions>
            </Dialog>
        </Paper>
    );
}

export default ControlPanel;

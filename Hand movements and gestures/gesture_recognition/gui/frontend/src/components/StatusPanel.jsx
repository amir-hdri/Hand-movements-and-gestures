import React from 'react';
import { Paper, Typography, Box, Chip } from '@mui/material';

function StatusPanel({ status }) {
    const { mode, training_status, last_prediction } = status;

    return (
        <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                System Status
            </Typography>

            <Box mb={2}>
                <Typography variant="body1">
                    <strong>Mode:</strong> <Chip label={mode.toUpperCase()} color={mode === 'recording' ? 'error' : 'default'} />
                </Typography>

                <Typography variant="body1">
                    <strong>Training:</strong> {training_status}
                </Typography>
            </Box>

            <Box mt={3}>
                <Typography variant="h6" gutterBottom>
                    Latest Prediction
                </Typography>
                <Typography variant="body1">
                    <strong>Action:</strong> {last_prediction.action || 'None'}
                </Typography>
                <Typography variant="body1">
                    <strong>Confidence:</strong> {(last_prediction.confidence * 100).toFixed(2)}%
                </Typography>
            </Box>
        </Paper>
    );
}

export default StatusPanel;

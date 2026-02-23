import React from 'react';
import { Paper, Box, CircularProgress, Typography } from '@mui/material';

function CameraView({ mode }) {
    const isRecording = mode === 'recording';

    return (
        <Paper
            elevation={3}
            sx={{
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                border: isRecording ? '4px solid red' : 'none'
            }}
        >
            <Typography variant="h6" gutterBottom>
                Live Camera Feed
            </Typography>

            <Box
                sx={{
                    width: '100%',
                    height: 480,
                    backgroundColor: '#000',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    overflow: 'hidden'
                }}
            >
                <img
                    src="/api/video_feed"
                    alt="Camera Feed"
                    style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                    onError={(e) => { e.target.style.display='none'; }}
                />
            </Box>
        </Paper>
    );
}

export default CameraView;

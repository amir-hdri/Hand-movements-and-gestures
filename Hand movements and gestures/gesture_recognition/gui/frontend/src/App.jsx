import React, { useState, useEffect } from 'react';
import {
    Container, Grid, Typography, Box,
    ThemeProvider, createTheme
} from '@mui/material';
import CameraView from './components/CameraView';
import ControlPanel from './components/ControlPanel';
import StatusPanel from './components/StatusPanel';
import SettingsPanel from './components/SettingsPanel';
import { fetchStatus, fetchConfig, fetchGestures } from './api';

const darkTheme = createTheme({
    palette: {
        mode: 'dark',
    },
});

function App() {
    const [status, setStatus] = useState({
        mode: 'idle',
        training_status: 'idle',
        last_prediction: { action: null, confidence: 0 }
    });
    const [config, setConfig] = useState({ smart_thresholds: {} });
    const [gestures, setGestures] = useState([]);

    // Poll status every second
    useEffect(() => {
        const interval = setInterval(async () => {
            try {
                const data = await fetchStatus();
                setStatus(data);
            } catch (e) {
                console.error("Failed to fetch status", e);
            }
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    // Load config and gestures on mount
    useEffect(() => {
        const load = async () => {
            try {
                const cfg = await fetchConfig();
                setConfig(cfg);
                const g = await fetchGestures();
                setGestures(g.gestures);
            } catch (e) {
                console.error("Failed to load initial data", e);
            }
        };
        load();
    }, []);

    return (
        <ThemeProvider theme={darkTheme}>
            <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
                <Typography variant="h4" component="h1" gutterBottom align="center">
                    Gesture Recognition System
                </Typography>

                <Grid container spacing={3}>
                    {/* Camera Feed */}
                    <Grid item xs={12} md={8}>
                        <CameraView mode={status.mode} />
                    </Grid>

                    {/* Controls & Status */}
                    <Grid item xs={12} md={4}>
                        <Box display="flex" flexDirection="column" gap={3}>
                            <StatusPanel status={status} />

                            <ControlPanel
                                status={status}
                                onConfigChange={(c) => setConfig(c)}
                                onGesturesChange={(g) => setGestures(g)}
                            />

                            <SettingsPanel
                                config={config}
                                gestures={gestures}
                                onConfigUpdate={(c) => setConfig(c)}
                            />
                        </Box>
                    </Grid>
                </Grid>
            </Container>
        </ThemeProvider>
    );
}

export default App;

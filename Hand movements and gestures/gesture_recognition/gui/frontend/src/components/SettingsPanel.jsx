import React, { useState, useEffect } from 'react';
import {
    Paper, Typography, Box, Slider, Button
} from '@mui/material';
import { updateConfig } from '../api';

const DEFAULT_THRESHOLD = 0.9;

function SettingsPanel({ config, gestures, onConfigUpdate }) {
    const [thresholds, setThresholds] = useState({});

    // Sync with config and ensure all gestures are present
    useEffect(() => {
        const newThresholds = { ...thresholds };

        // Populate from gestures with default if missing
        if (gestures) {
            gestures.forEach(g => {
                if (newThresholds[g] === undefined) {
                    newThresholds[g] = DEFAULT_THRESHOLD;
                }
            });
        }

        // Override with config
        if (config && config.smart_thresholds) {
            Object.assign(newThresholds, config.smart_thresholds);
        }

        setThresholds(newThresholds);
    }, [config, gestures]);

    const handleChange = (key, value) => {
        setThresholds(prev => ({ ...prev, [key]: value }));
    };

    const handleSave = async () => {
        try {
            // Only save changed ones? Or all?
            // API expects `thresholds` dict.
            // We can save all.
            await updateConfig(thresholds);
            onConfigUpdate({ ...config, smart_thresholds: thresholds });
        } catch (e) {
            console.error("Failed to save config", e);
        }
    };

    if (!thresholds || Object.keys(thresholds).length === 0) {
        return null;
    }

    return (
        <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                Smart Thresholds
            </Typography>

            <Box display="flex" flexDirection="column" gap={2} maxHeight={300} overflow="auto">
                {Object.entries(thresholds).map(([key, value]) => (
                    <Box key={key} display="flex" alignItems="center" gap={2}>
                        <Typography minWidth={100}>
                            {key}: {(value * 100).toFixed(0)}%
                        </Typography>
                        <Slider
                            value={value}
                            min={0.0}
                            max={1.0}
                            step={0.01}
                            onChange={(e, val) => handleChange(key, val)}
                            valueLabelDisplay="auto"
                            sx={{ flexGrow: 1 }}
                        />
                    </Box>
                ))}
            </Box>

            <Box mt={2}>
                <Button variant="contained" onClick={handleSave} fullWidth>
                    Save Thresholds
                </Button>
            </Box>
        </Paper>
    );
}

export default SettingsPanel;

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Paper, Typography, Box, Divider, Switch, FormControlLabel,
  TextField, Button, Chip, Grid, Slider, Tooltip
} from '@mui/material';
import { Settings, Save, Tune, Info, HelpOutline } from '@mui/icons-material';

function SettingsPanel({ config, gestures, onConfigUpdate }) {
  const [localConfig, setLocalConfig] = useState({
    seq_length: config.seq_length || 30,
    threshold: config.threshold || 0.9,
    stable_count: config.stable_count || 3
  });
  const [expanded, setExpanded] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setLocalConfig(prev => ({
      ...prev,
      [name]: name === 'seq_length' || name === 'stable_count' ? parseInt(value) || 1 : parseFloat(value) || 0
    }));
  };

  const handleSliderChange = (name, value) => {
    setLocalConfig(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSave = () => {
    onConfigUpdate(localConfig);
  };

  const handleReset = () => {
    setLocalConfig({
      seq_length: 30,
      threshold: 0.9,
      stable_count: 3
    });
  };

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Box display="flex" alignItems="center" gap={1} mb={2}>
        <Settings color="primary" fontSize="small" />
        <Typography variant="h6" flex={1}>
          Settings
        </Typography>
        <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
          <Chip
            label={expanded ? 'Collapse' : 'Expand'}
            size="small"
            onClick={() => setExpanded(!expanded)}
            color={expanded ? 'primary' : 'default'}
          />
        </motion.div>
      </Box>

      <Divider sx={{ my: 1 }} />

      <motion.div
        initial={false}
        animate={{ height: expanded ? 'auto' : 0 }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        style={{ overflow: 'hidden' }}
      >
        <Box display={expanded ? 'block' : 'none'}>
          {/* Detection Settings */}
          <Typography variant="subtitle2" fontWeight={600} mb={2} display="flex" alignItems="center" gap={0.5}>
            <Tune fontSize="small" /> Detection Parameters
          </Typography>

          <Grid container spacing={2} mb={2}>
            {/* Sequence Length */}
            <Grid item xs={12}>
              <Box>
                <Typography variant="body2" color="text.secondary" mb={1}>
                  Sequence Length
                </Typography>
                <Tooltip title="Number of frames to consider for gesture recognition">
                  <Slider
                    name="seq_length"
                    value={localConfig.seq_length}
                    onChange={(e, value) => handleSliderChange('seq_length', value)}
                    min={10}
                    max={60}
                    step={5}
                    valueLabelDisplay="auto"
                    marks={[
                      { value: 10, label: '10' },
                      { value: 30, label: '30' },
                      { value: 60, label: '60' }
                    ]}
                  />
                </Tooltip>
                <Typography variant="caption" color="text.secondary">
                  Frames: {localConfig.seq_length}
                </Typography>
              </Box>
            </Grid>

            {/* Confidence Threshold */}
            <Grid item xs={12}>
              <Box>
                <Typography variant="body2" color="text.secondary" mb={1}>
                  Confidence Threshold
                </Typography>
                <Tooltip title="Minimum confidence score to trigger a prediction">
                  <Slider
                    name="threshold"
                    value={localConfig.threshold}
                    onChange={(e, value) => handleSliderChange('threshold', value)}
                    min={0.5}
                    max={1}
                    step={0.05}
                    valueLabelDisplay="auto"
                    marks={[
                      { value: 0.5, label: '50%' },
                      { value: 0.75, label: '75%' },
                      { value: 1, label: '100%' }
                    ]}
                  />
                </Tooltip>
                <Typography variant="caption" color="text.secondary">
                  Threshold: {(localConfig.threshold * 100).toFixed(0)}%
                </Typography>
              </Box>
            </Grid>

            {/* Stable Count */}
            <Grid item xs={12}>
              <Box>
                <Typography variant="body2" color="text.secondary" mb={1}>
                  Stable Count
                </Typography>
                <Tooltip title="Number of consecutive predictions needed to trigger a stable action">
                  <Slider
                    name="stable_count"
                    value={localConfig.stable_count}
                    onChange={(e, value) => handleSliderChange('stable_count', value)}
                    min={1}
                    max={10}
                    step={1}
                    valueLabelDisplay="auto"
                    marks={[
                      { value: 1, label: '1' },
                      { value: 5, label: '5' },
                      { value: 10, label: '10' }
                    ]}
                  />
                </Tooltip>
                <Typography variant="caption" color="text.secondary">
                  Count: {localConfig.stable_count}
                </Typography>
              </Box>
            </Grid>
          </Grid>

          <Divider sx={{ my: 2 }} />

          {/* Action Buttons */}
          <Box display="flex" gap={2}>
            <Button
              variant="outlined"
              color="primary"
              onClick={handleReset}
              startIcon={<HelpOutline />}
              fullWidth
            >
              Reset to Defaults
            </Button>
            <Button
              variant="contained"
              color="success"
              onClick={handleSave}
              startIcon={<Save />}
              fullWidth
              disabled={
                localConfig.seq_length === 30 &&
                localConfig.threshold === 0.9 &&
                localConfig.stable_count === 3
              }
            >
              Save Settings
            </Button>
          </Box>

          {/* Gesture Info */}
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle2" fontWeight={600} mb={1} display="flex" alignItems="center" gap={0.5}>
            <Info fontSize="small" /> Configured Gestures
          </Typography>
          
          <Box display="flex" flexWrap="wrap" gap={1}>
            {gestures.length > 0 ? (
              gestures.map((gesture) => (
                <Chip
                  key={gesture}
                  label={gesture}
                  variant="outlined"
                  size="small"
                  sx={{ m: 0.5 }}
                />
              ))
            ) : (
              <Typography variant="body2" color="text.secondary">
                No gestures configured
              </Typography>
            )}
          </Box>
        </Box>
      </motion.div>
    </Paper>
  );
}

export default SettingsPanel;

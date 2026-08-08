import React from 'react';
import { motion } from 'framer-motion';
import {
  Paper, Typography, Box, List, ListItem, ListItemText,
  Chip, Divider, IconButton, Tooltip
} from '@mui/material';
import { History as HistoryIcon, Delete, CheckCircle, Cancel } from '@mui/icons-material';

function GestureHistory({ history }) {
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'success';
    if (confidence >= 0.7) return 'info';
    if (confidence >= 0.5) return 'warning';
    return 'error';
  };

  // Native formatting (no external date-fns dependency required).
  const formatTime = (isoString) => {
    try {
      const date = new Date(isoString);
      if (Number.isNaN(date.getTime())) return 'N/A';
      return date.toLocaleTimeString([], { hour12: false });
    } catch {
      return 'N/A';
    }
  };

  const handleClear = () => {
    // Clear history logic would go here
    // For now, this is a placeholder
  };

  if (history.length === 0) {
    return (
      <Paper elevation={3} sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          <HistoryIcon color="action" sx={{ mr: 1, verticalAlign: 'middle' }} />
          Prediction History
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          No predictions yet. Start recording to see results.
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
        <Typography variant="h6" display="flex" alignItems="center" gap={1}>
          <HistoryIcon color="primary" />
          Prediction History
        </Typography>
        <Tooltip title="Clear history">
          <IconButton size="small" onClick={handleClear} color="inherit">
            <Delete fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>

      <List dense sx={{ maxHeight: 400, overflow: 'auto' }}>
        {history.map((entry, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            <ListItem
              secondaryAction={
                <Chip
                  label={`${(entry.confidence * 100).toFixed(1)}%`}
                  color={getConfidenceColor(entry.confidence)}
                  size="small"
                  icon={entry.confidence >= 0.7 ? <CheckCircle fontSize="small" /> : <Cancel fontSize="small" />}
                />
              }
            >
              <ListItemText
                primary={entry.action || 'Unknown'}
                secondary={formatTime(entry.timestamp)}
                primaryTypographyProps={{ fontWeight: 500 }}
              />
            </ListItem>
            {index < history.length - 1 && <Divider component="li" sx={{ mx: 2 }} />}
          </motion.div>
        ))}
      </List>
    </Paper>
  );
}

export default GestureHistory;

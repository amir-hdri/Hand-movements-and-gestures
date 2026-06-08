import React from 'react';
import { motion } from 'framer-motion';
import {
  Paper, Typography, Box, Chip, Divider, LinearProgress,
  Avatar, List, ListItem, ListItemAvatar, ListItemText
} from '@mui/material';
import {
  Circle, PlayCircleFilled, StopCircle, School, CheckCircle,
  Error, Warning, Info, Wifi, WifiOff, AutoGraph
} from '@mui/icons-material';

function StatusPanel({ status, isConnected }) {
  const { mode, training_status, last_prediction } = status;

  const getModeIcon = () => {
    switch (mode) {
      case 'recording':
        return <PlayCircleFilled color="error" />;
      case 'training':
        return <School color="warning" />;
      default:
        return <Circle color="success" />;
    }
  };

  const getModeColor = () => {
    switch (mode) {
      case 'recording':
        return 'error';
      case 'training':
        return 'warning';
      default:
        return 'success';
    }
  };

  const getTrainingStatusIcon = () => {
    switch (training_status) {
      case 'training':
        return <Box display="flex" alignItems="center" gap={1}>
          <Chip label="Training" size="small" color="warning" icon={<School fontSize="small" />} />
        </Box>;
      case 'completed':
        return <Chip label="Complete" size="small" color="success" icon={<CheckCircle fontSize="small" />} />;
      case 'failed':
        return <Chip label="Failed" size="small" color="error" icon={<Error fontSize="small" />} />;
      default:
        return <Chip label={training_status || "Idle"} size="small" color="default" />;
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'success';
    if (confidence >= 0.7) return 'info';
    if (confidence >= 0.5) return 'warning';
    return 'error';
  };

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Box display="flex" alignItems="center" gap={1} mb={2}>
        <Avatar sx={{ bgcolor: getModeColor() + '.main', width: 32, height: 32 }}>
          {getModeIcon()}
        </Avatar>
        <Typography variant="h6" flex={1}>
          System Status
        </Typography>
        <Chip 
          label={isConnected ? 'Connected' : 'Disconnected'}
          size="small"
          color={isConnected ? 'success' : 'error'}
          icon={isConnected ? <Wifi fontSize="small" /> : <WifiOff fontSize="small" />}
        />
      </Box>

      <List dense>
        {/* Mode */}
        <ListItem disablePadding>
          <ListItemAvatar>
            <Avatar sx={{ bgcolor: 'transparent', width: 32, height: 32 }}>
              {getModeIcon()}
            </Avatar>
          </ListItemAvatar>
          <ListItemText
            primary="Current Mode"
            secondary={mode.toUpperCase()}
            primaryTypographyProps={{ variant: 'subtitle2', color: 'text.secondary' }}
            secondaryTypographyProps={{ variant: 'body1', fontWeight: 500 }}
          />
        </ListItem>

        <Divider component="li" />

        {/* Training Status */}
        <ListItem disablePadding>
          <ListItemAvatar>
            <Avatar sx={{ bgcolor: 'transparent', width: 32, height: 32 }}>
              <School color={training_status === 'training' ? 'warning' : 'action'} />
            </Avatar>
          </ListItemAvatar>
          <ListItemText
            primary="Training Status"
            secondary={getTrainingStatusIcon()}
            primaryTypographyProps={{ variant: 'subtitle2', color: 'text.secondary' }}
          />
        </ListItem>

        <Divider component="li" />

        {/* Latest Prediction */}
        <ListItem disablePadding>
          <ListItemAvatar>
            <Avatar sx={{ bgcolor: 'transparent', width: 32, height: 32 }}>
              {last_prediction?.action ? (
                last_prediction.confidence >= 0.7 ? (
                  <CheckCircle color="success" />
                ) : (
                  <Warning color="warning" />
                )
              ) : (
                <Info color="action" />
              )}
            </Avatar>
          </ListItemAvatar>
          <ListItemText
            primary="Latest Prediction"
            secondary={
              last_prediction?.action ? (
                <Box display="flex" alignItems="center" gap={1}>
                  <Typography variant="body1" fontWeight={600}>
                    {last_prediction.action}
                  </Typography>
                  <Chip
                    label={`${(last_prediction.confidence * 100).toFixed(1)}%`}
                    size="small"
                    color={getConfidenceColor(last_prediction.confidence)}
                  />
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Awaiting prediction...
                </Typography>
              )
            }
            primaryTypographyProps={{ variant: 'subtitle2', color: 'text.secondary' }}
          />
        </ListItem>

        {/* Confidence Progress */}
        {last_prediction?.action && (
          <>
            <Divider component="li" />
            <ListItem disablePadding>
              <ListItemAvatar>
                <Avatar sx={{ bgcolor: 'transparent', width: 32, height: 32 }}>
                  <AutoGraph fontSize="small" color="primary" />
                </Avatar>
              </ListItemAvatar>
              <ListItemText
                primary="Confidence Level"
                primaryTypographyProps={{ variant: 'subtitle2', color: 'text.secondary' }}
                secondary={
                  <LinearProgress
                    variant="determinate"
                    value={last_prediction.confidence * 100}
                    sx={{ height: 8, borderRadius: 4, my: 0.5 }}
                    color={getConfidenceColor(last_prediction.confidence)}
                  />
                }
              />
            </ListItem>
          </>
        )}
      </List>
    </Paper>
  );
}

export default StatusPanel;

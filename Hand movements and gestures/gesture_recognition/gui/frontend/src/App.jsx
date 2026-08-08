import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Container, Grid, Typography, Box, IconButton, Tooltip,
  useTheme, useMediaQuery
} from '@mui/material';
import {
  Brightness4, Brightness7, CameraAlt, Gesture, History,
  Settings, PlayArrow, Stop, School, AutoGraph
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';

import CameraView from './components/CameraView';
import ControlPanel from './components/ControlPanel';
import StatusPanel from './components/StatusPanel';
import SettingsPanel from './components/SettingsPanel';
import GestureHistory from './components/GestureHistory';
import PredictionStats from './components/PredictionStats';
import { fetchStatus, fetchConfig, fetchGestures, fetchPredictionHistory } from './api';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: 'easeOut' },
  },
};

function App({ themeMode, onToggleTheme }) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const { enqueueSnackbar } = useSnackbar();

  const [status, setStatus] = useState({
    mode: 'idle',
    training_status: 'idle',
    last_prediction: { action: null, confidence: 0 }
  });
  const [config, setConfig] = useState({ smart_thresholds: {} });
  const [gestures, setGestures] = useState([]);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [isConnected, setIsConnected] = useState(true);
  const [showHistory, setShowHistory] = useState(false);
  const [showStats, setShowStats] = useState(false);

  // Poll status every second
  const pollStatus = useCallback(async () => {
    try {
      const data = await fetchStatus();
      setStatus(prev => ({
        ...prev,
        ...data,
        // Preserve existing values if new data is undefined
        mode: data.mode ?? prev.mode,
        training_status: data.training_status ?? prev.training_status,
        last_prediction: data.last_prediction ?? prev.last_prediction
      }));
      setIsConnected(true);
    } catch (e) {
      console.error("Failed to fetch status", e);
      setIsConnected(false);
      enqueueSnackbar('Connection lost. Reconnecting...', { variant: 'error' });
    }
  }, [enqueueSnackbar]);

  useEffect(() => {
    const interval = setInterval(pollStatus, 1000);
    return () => clearInterval(interval);
  }, [pollStatus]);

  // Load initial data
  useEffect(() => {
    const load = async () => {
      try {
        const [cfg, g, history] = await Promise.all([
          fetchConfig(),
          fetchGestures(),
          fetchPredictionHistory()
        ]);
        setConfig(cfg);
        setGestures(g.gestures || []);
        setPredictionHistory(history || []);
      } catch (e) {
        console.error("Failed to load initial data", e);
        enqueueSnackbar('Failed to load initial data', { variant: 'error' });
      }
    };
    load();
  }, [enqueueSnackbar]);

  // Connection recovery
  useEffect(() => {
    if (!isConnected) {
      const recoveryInterval = setInterval(pollStatus, 3000);
      return () => clearInterval(recoveryInterval);
    }
  }, [isConnected, pollStatus]);

  // Handle prediction updates.  status is polled every second and produces a
  // fresh last_prediction object each time, so we track the last *action* seen
  // to avoid adding duplicate history entries for an unchanged gesture.
  const lastHistoryAction = React.useRef(null);
  useEffect(() => {
    const action = status.last_prediction?.action;
    if (action && action !== lastHistoryAction.current) {
      lastHistoryAction.current = action;
      const newEntry = {
        action,
        confidence: status.last_prediction.confidence,
        timestamp: new Date().toISOString()
      };
      setPredictionHistory(prev => [newEntry, ...prev].slice(0, 20));
    }
  }, [status.last_prediction]);

  const toggleHistory = () => setShowHistory(prev => !prev);
  const toggleStats = () => setShowStats(prev => !prev);

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: theme.palette.mode === 'dark'
          ? `linear-gradient(135deg, ${theme.palette.background.default} 0%, ${theme.palette.background.paper} 100%)`
          : theme.palette.background.default,
        pb: 4
      }}
    >
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Container maxWidth="xl" sx={{ py: 3 }}>
          <Box display="flex" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={2}>
            <Box display="flex" alignItems="center" gap={2}>
              <motion.div
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <IconButton sx={{ p: 1.5 }} onClick={onToggleTheme} color="inherit">
                  {themeMode === 'dark' ? <Brightness7 /> : <Brightness4 />}
                </IconButton>
              </motion.div>
              <Box>
                <Typography variant="h4" component="h1" fontWeight={600} display="flex" alignItems="center" gap={1}>
                  <Gesture fontSize="large" />
                  Hand Gesture Recognition
                </Typography>
                <Typography variant="subtitle1" color="text.secondary">
                  Real-time gesture detection and classification
                </Typography>
              </Box>
            </Box>
            
            <Box display="flex" alignItems="center" gap={1}>
              <Tooltip title="Prediction History">
                <IconButton
                  onClick={toggleHistory}
                  color={showHistory ? 'primary' : 'inherit'}
                  sx={{ p: 1.5 }}
                >
                  <History />
                </IconButton>
              </Tooltip>
              <Tooltip title="Statistics">
                <IconButton
                  onClick={toggleStats}
                  color={showStats ? 'primary' : 'inherit'}
                  sx={{ p: 1.5 }}
                >
                  <AutoGraph />
                </IconButton>
              </Tooltip>
              <Tooltip title="Settings">
                <IconButton sx={{ p: 1.5 }} color="inherit">
                  <Settings />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </Container>
      </motion.div>

      {/* Main Content */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <Container maxWidth="xl" sx={{ mt: 2 }}>
          <Grid container spacing={3}>
            {/* Camera Feed - Full width on mobile, 2/3 on desktop */}
            <Grid item xs={12} md={8} lg={showHistory || showStats ? 7 : 8}>
              <motion.div variants={itemVariants}>
                <CameraView mode={status.mode} isConnected={isConnected} />
              </motion.div>
            </Grid>

            {/* Sidebar - Full width on mobile, 1/3 on desktop */}
            <Grid item xs={12} md={4} lg={showHistory || showStats ? 5 : 4}>
              <motion.div variants={itemVariants}>
                <Box display="flex" flexDirection="column" gap={3}>
                  {/* Status Panel */}
                  <motion.div
                    whileHover={{ scale: 1.01 }}
                    transition={{ type: 'spring', stiffness: 300 }}
                  >
                    <StatusPanel status={status} isConnected={isConnected} />
                  </motion.div>

                  {/* Controls */}
                  <motion.div
                    whileHover={{ scale: 1.01 }}
                    transition={{ type: 'spring', stiffness: 300 }}
                  >
                    <ControlPanel
                      status={status}
                      gestures={gestures}
                      onConfigChange={setConfig}
                      onGesturesChange={setGestures}
                      isConnected={isConnected}
                    />
                  </motion.div>

                  {/* Settings */}
                  <motion.div
                    whileHover={{ scale: 1.01 }}
                    transition={{ type: 'spring', stiffness: 300 }}
                  >
                    <SettingsPanel
                      config={config}
                      gestures={gestures}
                      onConfigUpdate={setConfig}
                    />
                  </motion.div>
                </Box>
              </motion.div>
            </Grid>

            {/* Additional Panels - Only shown when toggled */}
            <AnimatePresence>
              {showHistory && (
                <Grid item xs={12} lg={5}>
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <GestureHistory history={predictionHistory} />
                  </motion.div>
                </Grid>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {showStats && (
                <Grid item xs={12} lg={5}>
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <PredictionStats history={predictionHistory} gestures={gestures} />
                  </motion.div>
                </Grid>
              )}
            </AnimatePresence>
          </Grid>
        </Container>
      </motion.div>

      {/* Connection Status Indicator */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        style={{ position: 'fixed', bottom: 20, right: 20 }}
      >
        <Tooltip title={isConnected ? 'Connected to backend' : 'Disconnected from backend'}>
          <Box
            sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              backgroundColor: isConnected ? 'success.main' : 'error.main',
              boxShadow: '0 0 0 0 currentColor',
              animation: isConnected ? 'pulse 2s infinite' : 'none',
              '@keyframes pulse': {
                '0%': { boxShadow: '0 0 0 0 rgba(76, 175, 80, 0.7)' },
                '70%': { boxShadow: '0 0 0 10px rgba(76, 175, 80, 0)' },
                '100%': { boxShadow: '0 0 0 0 rgba(76, 175, 80, 0)' },
              }
            }}
          />
        </Tooltip>
      </motion.div>
    </Box>
  );
}

export default App;

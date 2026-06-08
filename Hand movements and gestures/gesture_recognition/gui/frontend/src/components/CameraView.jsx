import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Paper, Box, Typography, LinearProgress, Fade, IconButton, Tooltip
} from '@mui/material';
import {
  VideocamOff, RecordCircle, WifiOff, Wifi, Fullscreen, FullscreenExit
} from '@mui/icons-material';

function CameraView({ mode, isConnected }) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [error, setError] = useState(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  const isRecording = mode === 'recording';

  // Handle fullscreen toggle
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => {
        console.error('Error attempting to enable fullscreen:', err);
      });
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  // Handle fullscreen change events
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  // Handle image error
  const handleImageError = () => {
    setError('Camera feed unavailable');
  };

  const handleImageLoad = () => {
    setImageLoaded(true);
    setError(null);
  };

  return (
    <Paper
      elevation={3}
      sx={{
        p: 2,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        position: 'relative',
        border: isRecording ? '3px solid' : 'none',
        borderColor: isRecording ? 'error.main' : 'transparent',
        boxShadow: isRecording 
          ? `0 0 20px ${theme => theme.palette.error.main}`
          : 'none',
        overflow: 'hidden'
      }}
    >
      <Box
        display="flex"
        alignItems="center"
        justifyContent="space-between"
        width="100%"
        mb={2}
      >
        <Box display="flex" alignItems="center" gap={1}>
          <Typography variant="h6" display="flex" alignItems="center" gap={0.5}>
            <VideocamOff fontSize="small" />
            Live Camera Feed
          </Typography>
          
          {/* Connection Status */}
          <Fade in={!isConnected} unmountOnExit>
            <Tooltip title="Camera connection lost">
              <WifiOff color="error" fontSize="small" />
            </Tooltip>
          </Fade>
          
          <Fade in={isConnected} unmountOnExit>
            <Tooltip title="Camera connected">
              <Wifi color="success" fontSize="small" />
            </Tooltip>
          </Fade>
        </Box>

        <Box display="flex" alignItems="center" gap={1}>
          {/* Recording Indicator */}
          {isRecording && (
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: [0.8, 1.2, 0.8] }}
              transition={{ duration: 1, repeat: Infinity, ease: 'easeInOut' }}
            >
              <RecordCircle color="error" fontSize="small" />
            </motion.div>
          )}
          
          {/* Fullscreen Button */}
          <Tooltip title={isFullscreen ? 'Exit fullscreen' : 'Go fullscreen'}>
            <IconButton
              onClick={toggleFullscreen}
              size="small"
              sx={{ color: 'text.secondary' }}
            >
              {isFullscreen ? <FullscreenExit fontSize="small" /> : <Fullscreen fontSize="small" />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Loading Progress */}
      {!imageLoaded && !error && (
        <Box width="100%" sx={{ mb: 2 }}>
          <LinearProgress color="secondary" />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, textAlign: 'center', display: 'block' }}>
            Loading camera feed...
          </Typography>
        </Box>
      )}

      {/* Error State */}
      {error && (
        <Paper
          elevation={0}
          sx={{
            width: '100%',
            height: 480,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            backgroundColor: 'background.paper',
            p: 4
          }}
        >
          <VideocamOff color="disabled" sx={{ fontSize: 60, mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            {error}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Make sure the backend server is running and camera is accessible
          </Typography>
        </Paper>
      )}

      {/* Camera Feed */}
      {imageLoaded && !error && (
        <Box
          sx={{
            width: '100%',
            height: isFullscreen ? 'calc(100vh - 200px)' : 480,
            backgroundColor: '#000',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            overflow: 'hidden',
            borderRadius: 1,
            position: 'relative'
          }}
        >
          <img
            src="/api/video_feed"
            alt="Camera Feed"
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              display: 'block'
            }}
            onError={handleImageError}
            onLoad={handleImageLoad}
          />
        </Box>
      )}

      {/* Recording Overlay */}
      {isRecording && imageLoaded && !error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          style={{
            position: 'absolute',
            bottom: 20,
            left: '50%',
            transform: 'translateX(-50%)',
            backgroundColor: 'rgba(244, 67, 54, 0.9)',
            color: 'white',
            padding: '8px 20px',
            borderRadius: 20,
            fontSize: '0.875rem',
            fontWeight: 500
          }}
        >
          RECORDING
        </motion.div>
      )}
    </Paper>
  );
}

export default CameraView;

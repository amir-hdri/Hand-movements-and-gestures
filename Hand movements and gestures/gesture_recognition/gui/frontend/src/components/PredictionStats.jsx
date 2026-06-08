import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Paper, Typography, Box, Grid,
  LinearProgress, Divider, Tooltip
} from '@mui/material';
import { BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#A4DE6C', '#D0ED57', '#FF6B6B'];

function PredictionStats({ history, gestures }) {
  // Calculate statistics
  const stats = useMemo(() => {
    if (history.length === 0) {
      return {
        total: 0,
        byGesture: {},
        avgConfidence: 0,
        highConfidence: 0,
        distribution: []
      };
    }

    const byGesture = {};
    let totalConfidence = 0;
    let highConfidence = 0;

    history.forEach(entry => {
      const gesture = entry.action || 'Unknown';
      if (!byGesture[gesture]) {
        byGesture[gesture] = { count: 0, totalConfidence: 0 };
      }
      byGesture[gesture].count++;
      byGesture[gesture].totalConfidence += entry.confidence;
      totalConfidence += entry.confidence;
      if (entry.confidence >= 0.7) highConfidence++;
    });

    const avgConfidence = totalConfidence / history.length;
    const gestureDistribution = Object.entries(byGesture).map(([gesture, data]) => ({
      name: gesture,
      count: data.count,
      confidence: (data.totalConfidence / data.count).toFixed(2)
    }));

    return {
      total: history.length,
      byGesture,
      avgConfidence: (avgConfidence * 100).toFixed(1),
      highConfidence: ((highConfidence / history.length) * 100).toFixed(1),
      distribution: gestureDistribution
    };
  }, [history]);

  // Chart data
  const chartData = useMemo(() => {
    return stats.distribution.map(item => ({
      name: item.name,
      value: item.count
    }));
  }, [stats.distribution]);

  if (history.length === 0) {
    return (
      <Paper elevation={3} sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          Prediction Statistics
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          No prediction data available yet.
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Prediction Statistics
      </Typography>

      <Divider sx={{ my: 2 }} />

      {/* Summary Stats */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6}>
          <Box>
            <Typography variant="body2" color="text.secondary">
              Total Predictions
            </Typography>
            <Typography variant="h5" fontWeight={600}>
              {stats.total}
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={6}>
          <Box>
            <Typography variant="body2" color="text.secondary">
              Avg Confidence
            </Typography>
            <Typography variant="h5" fontWeight={600}>
              {stats.avgConfidence}%
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={6}>
          <Box>
            <Typography variant="body2" color="text.secondary">
              High Confidence
            </Typography>
            <Tooltip title="Predictions with confidence >= 70%">
              <Typography variant="h5" fontWeight={600}>
                {stats.highConfidence}%
              </Typography>
            </Tooltip>
          </Box>
        </Grid>
        <Grid item xs={6}>
          <Box>
            <Typography variant="body2" color="text.secondary">
              Unique Gestures
            </Typography>
            <Typography variant="h5" fontWeight={600}>
              {Object.keys(stats.byGesture).length}
            </Typography>
          </Box>
        </Grid>
      </Grid>

      <Divider sx={{ my: 2 }} />

      {/* Gesture Distribution Chart */}
      <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
        Gesture Distribution
      </Typography>
      
      <Box sx={{ height: 200, mb: 3 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 0, left: -10, bottom: 0 }}>
            <Bar dataKey="value" fill="#2196F3" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Confidence Distribution */}
      <Typography variant="subtitle2" gutterBottom>
        Confidence Distribution
      </Typography>
      
      <Box display="flex" flexDirection="column" gap={1}>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="body2" sx={{ minWidth: 100 }}>High (&gt;90%)</Typography>
          <LinearProgress
            variant="determinate"
            value={Math.min(100, (history.filter(h => h.confidence >= 0.9).length / history.length) * 100)}
            sx={{ flex: 1, height: 8, borderRadius: 4 }}
            color="success"
          />
          <Typography variant="body2" color="text.secondary">
            {((history.filter(h => h.confidence >= 0.9).length / history.length) * 100).toFixed(0)}%
          </Typography>
        </Box>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="body2" sx={{ minWidth: 100 }}>Good (70-90%)</Typography>
          <LinearProgress
            variant="determinate"
            value={Math.min(100, (history.filter(h => h.confidence >= 0.7 && h.confidence < 0.9).length / history.length) * 100)}
            sx={{ flex: 1, height: 8, borderRadius: 4 }}
            color="info"
          />
          <Typography variant="body2" color="text.secondary">
            {((history.filter(h => h.confidence >= 0.7 && h.confidence < 0.9).length / history.length) * 100).toFixed(0)}%
          </Typography>
        </Box>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="body2" sx={{ minWidth: 100 }}>Low (&lt;70%)</Typography>
          <LinearProgress
            variant="determinate"
            value={Math.min(100, (history.filter(h => h.confidence < 0.7).length / history.length) * 100)}
            sx={{ flex: 1, height: 8, borderRadius: 4 }}
            color="warning"
          />
          <Typography variant="body2" color="text.secondary">
            {((history.filter(h => h.confidence < 0.7).length / history.length) * 100).toFixed(0)}%
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
}

export default PredictionStats;

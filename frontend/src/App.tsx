import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Button,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Tabs,
  Tab,
  ThemeProvider,
  createTheme,
  CssBaseline
} from '@mui/material';
import {
  Visibility as EyeIcon,
  TrendingUp as RiskIcon,
  Assessment as StatsIcon,
  CloudUpload as UploadIcon,
  Speed as SpeedIcon
} from '@mui/icons-material';
import RiskPredictor from './components/RiskPredictor';
import DataGenerator from './components/DataGenerator';
import Dashboard from './components/Dashboard';
import ModelTraining from './components/ModelTraining';
import axios from 'axios';

// 创建主题
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    success: {
      main: '#4caf50',
    },
    warning: {
      main: '#ff9800',
    },
    error: {
      main: '#f44336',
    },
  },
  typography: {
    fontFamily: '"Microsoft YaHei", "微软雅黑", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "SimHei", "黑体", "Noto Sans SC", sans-serif',
  },
});

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://154.8.203.14:3000';

function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [healthStatus, setHealthStatus] = useState(null);
  const [stats, setStats] = useState(null);

  // 获取健康状态
  useEffect(() => {
    fetchHealthStatus();
    const interval = setInterval(fetchHealthStatus, 30000); // 每30秒更新一次
    return () => clearInterval(interval);
  }, []);

  // 获取统计信息
  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 60000); // 每分钟更新一次
    return () => clearInterval(interval);
  }, []);

  const fetchHealthStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/health`);
      setHealthStatus(response.data);
    } catch (err) {
      console.error('Failed to fetch health status:', err);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/stats`);
      setStats(response.data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const renderContent = () => {
    switch (activeTab) {
      case 0:
        return (
          <Dashboard
            stats={stats}
            healthStatus={healthStatus}
            onRefresh={() => {
              fetchHealthStatus();
              fetchStats();
            }}
          />
        );
      case 1:
        return <RiskPredictor />;
      case 2:
        return <DataGenerator />;
      case 3:
        return <ModelTraining />;
      default:
        return null;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', bgcolor: '#f5f5f5' }}>
        {/* Header */}
        <Box
          sx={{
            bgcolor: 'primary.main',
            color: 'white',
            py: 3,
            boxShadow: 3,
          }}
        >
          <Container maxWidth="xl">
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <EyeIcon sx={{ fontSize: 48 }} />
                <Box>
                  <Typography variant="h4" fontWeight="bold">
                    AI近视风险预测系统
                  </Typography>
                  <Typography variant="subtitle1">
                    AI-Powered Myopia Risk Prediction System
                  </Typography>
                </Box>
              </Box>
              {healthStatus && (
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                  <Alert
                    severity={healthStatus.modelStatus?.trained ? 'success' : 'warning'}
                    sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }}
                  >
                    {healthStatus.modelStatus?.trained ? '模型已就绪' : '模型未训练'}
                  </Alert>
                </Box>
              )}
            </Box>
          </Container>
        </Box>

        {/* Main Content */}
        <Container maxWidth="xl" sx={{ mt: 4, pb: 4 }}>
          {/* Tabs */}
          <Paper sx={{ mb: 3 }}>
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              variant="scrollable"
              scrollButtons="auto"
            >
              <Tab label="仪表盘" icon={<StatsIcon />} iconPosition="start" />
              <Tab label="风险预测" icon={<RiskIcon />} iconPosition="start" />
              <Tab label="数据生成" icon={<UploadIcon />} iconPosition="start" />
              <Tab label="模型训练" icon={<SpeedIcon />} iconPosition="start" />
            </Tabs>
          </Paper>

          {/* Error Alert */}
          {error && (
            <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
              {error}
            </Alert>
          )}

          {/* Content */}
          {renderContent()}
        </Container>

        {/* Footer */}
        <Box
          sx={{
            bgcolor: '#1976d2',
            color: 'white',
            py: 2,
            textAlign: 'center',
          }}
        >
          <Typography variant="body2">
            © 2026 AI近视风险预测系统 | 基于深度学习与医学研究
          </Typography>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;

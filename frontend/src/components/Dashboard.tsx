import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Chip
} from '@mui/material';
import {
  Visibility as EyeIcon,
  TrendingUp as RiskIcon,
  TrendingUp as TrendingIcon,
  CloudUpload as DataIcon,
  Speed as SpeedIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

interface DashboardProps {
  stats: any;
  healthStatus: any;
  onRefresh: () => void;
}

const Dashboard: React.FC<DashboardProps> = ({ stats, healthStatus, onRefresh }) => {
  const statCards = [
    {
      title: '总预测次数',
      value: stats?.totalPredictions || 0,
      icon: <EyeIcon />,
      color: 'primary',
      subtitle: '累计分析'
    },
    {
      title: '高风险人数',
      value: stats?.highRiskCount || 0,
      icon: <RiskIcon />,
      color: 'error',
      subtitle: '需重点关注'
    },
    {
      title: '中风险人数',
      value: stats?.mediumRiskCount || 0,
      icon: <TrendingIcon />,
      color: 'warning',
      subtitle: '需适当干预'
    },
    {
      title: '低风险人数',
      value: stats?.lowRiskCount || 0,
      icon: <RefreshIcon />,
      color: 'success',
      subtitle: '保持良好习惯'
    }
  ];

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" fontWeight="bold">
          系统概览
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={onRefresh}
        >
          刷新数据
        </Button>
      </Box>

      {/* 统计卡片 */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {statCards.map((card, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <Box>
                    <Typography variant="h4" fontWeight="bold" color={card.color + '.main'}>
                      {card.value.toLocaleString()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {card.title}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {card.subtitle}
                    </Typography>
                  </Box>
                  <Box sx={{ color: card.color + '.main' }}>
                    {card.icon}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* 系统状态 */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              模型状态
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography>训练状态:</Typography>
                <Chip
                  label={healthStatus?.modelStatus?.trained ? '已训练' : '未训练'}
                  color={healthStatus?.modelStatus?.trained ? 'success' : 'warning'}
                  size="small"
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography>模型准确率:</Typography>
                <Typography>
                  {healthStatus?.modelStatus?.accuracy
                    ? `${(healthStatus.modelStatus.accuracy * 100).toFixed(1)}%`
                    : 'N/A'}
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography>最后训练时间:</Typography>
                <Typography>
                  {healthStatus?.modelStatus?.lastTrainingTime
                    ? new Date(healthStatus.modelStatus.lastTrainingTime).toLocaleString()
                    : 'N/A'}
                </Typography>
              </Box>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              数据生成状态
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography>生成状态:</Typography>
                <Chip
                  label={healthStatus?.dataGenerationStatus?.generating ? '生成中' : '空闲'}
                  color={healthStatus?.dataGenerationStatus?.generating ? 'warning' : 'success'}
                  size="small"
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography>种子数据量:</Typography>
                <Typography>
                  {healthStatus?.dataGenerationStatus?.seedDataCount || 0}
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography>生成数据量:</Typography>
                <Typography>
                  {healthStatus?.dataGenerationStatus?.generatedDataCount || 0}
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography>验证通过:</Typography>
                <Chip
                  label={healthStatus?.dataGenerationStatus?.validationPassed ? '是' : '否'}
                  color={healthStatus?.dataGenerationStatus?.validationPassed ? 'success' : 'warning'}
                  size="small"
                />
              </Box>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* 快速操作 */}
      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          快速操作
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<RiskIcon />}
              onClick={() => window.location.hash = '#predict'}
            >
              开始预测
            </Button>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<DataIcon />}
              onClick={() => window.location.hash = '#generate'}
            >
              生成数据
            </Button>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<SpeedIcon />}
              onClick={() => window.location.hash = '#train'}
            >
              训练模型
            </Button>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default Dashboard;

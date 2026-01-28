import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  Slider,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  LinearProgress,
  Divider,
  Collapse
} from '@mui/material';
import {
  Visibility as EyeIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  TrendingUp as TrendingIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = 'http://154.8.203.14:3000';

const RiskPredictor = () => {
  const [formData, setFormData] = useState({
    age: 12,
    gender: 1,
    axial_length: 24.5,
    choroidal_thickness: 280,
    sphere: -2.5,
    cylinder: -0.5,
    parent_myopia: 0,
    outdoor_hours: 2,
    near_work_hours: 6,
    screen_time_hours: 3,
    genetic_risk_score: 50
  });

  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');
  const [showDetails, setShowDetails] = useState(true);

  const handleChange = (field: string) => (event: any, newValue?: any) => {
    const value = event?.target?.value !== undefined ? event.target.value : newValue;
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      console.log('发送预测请求:', formData);
      const response = await axios.post(`${API_BASE_URL}/api/predict`, formData);

      console.log('收到响应:', response.data);

      if (response.data.success) {
        setPrediction(response.data.prediction);
      } else {
        setError('预测失败,请稍后重试');
      }
    } catch (err: any) {
      console.error('预测错误:', err);
      setError(err.response?.data?.error || err.message || '预测失败,请检查输入数据');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level: number) => {
    switch (level) {
      case 2:
        return 'error';
      case 1:
        return 'warning';
      default:
        return 'success';
    }
  };

  const getRiskIcon = (level: number) => {
    switch (level) {
      case 2:
        return <WarningIcon sx={{ fontSize: 48 }} />;
      case 1:
        return <TrendingIcon sx={{ fontSize: 48 }} />;
      default:
        return <CheckIcon sx={{ fontSize: 48 }} />;
    }
  };

  return (
    <Grid container spacing={3}>
      {/* 输入表单 */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <EyeIcon /> 输入眼部参数
          </Typography>
          <Divider sx={{ mb: 3 }} />

          <Grid container spacing={2}>
            {/* 基本信息 */}
            <Grid item xs={6}>
              <Typography gutterBottom>年龄: {formData.age} 岁</Typography>
              <Slider
                value={formData.age}
                onChange={handleChange('age')}
                min={6}
                max={18}
                marks
                step={1}
                valueLabelDisplay="auto"
              />
            </Grid>

            <Grid item xs={6}>
              <Typography gutterBottom>性别</Typography>
              <Button
                variant={formData.gender === 0 ? 'contained' : 'outlined'}
                onClick={() => setFormData(prev => ({ ...prev, gender: 0 }))}
                sx={{ mr: 1 }}
              >
                女
              </Button>
              <Button
                variant={formData.gender === 1 ? 'contained' : 'outlined'}
                onClick={() => setFormData(prev => ({ ...prev, gender: 1 }))}
              >
                男
              </Button>
            </Grid>

            {/* 眼部结构参数 */}
            <Grid item xs={12}>
              <Typography gutterBottom>眼轴长度: {formData.axial_length.toFixed(1)} mm</Typography>
              <Slider
                value={formData.axial_length}
                onChange={handleChange('axial_length')}
                min={22}
                max={28}
                step={0.1}
                valueLabelDisplay="auto"
                marks={[
                  { value: 22, label: '22mm' },
                  { value: 24, label: '24mm' },
                  { value: 26, label: '26mm' },
                  { value: 28, label: '28mm' }
                ]}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>脉络膜厚度: {formData.choroidal_thickness.toFixed(0)} μm</Typography>
              <Slider
                value={formData.choroidal_thickness}
                onChange={handleChange('choroidal_thickness')}
                min={100}
                max={400}
                step={10}
                valueLabelDisplay="auto"
                marks={[
                  { value: 100, label: '100μm' },
                  { value: 250, label: '250μm' },
                  { value: 400, label: '400μm' }
                ]}
              />
            </Grid>

            {/* 屈光度 */}
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="球镜度数 (D)"
                type="number"
                value={formData.sphere}
                onChange={(e) => handleChange('sphere')(e)}
                inputProps={{ min: -10, max: 0, step: 0.25 }}
                helperText="近视度数"
              />
            </Grid>

            <Grid item xs={6}>
              <TextField
                fullWidth
                label="柱镜度数 (D)"
                type="number"
                value={formData.cylinder}
                onChange={(e) => handleChange('cylinder')(e)}
                inputProps={{ min: -3, max: 0, step: 0.25 }}
                helperText="散光度数"
              />
            </Grid>

            {/* 遗传因素 */}
            <Grid item xs={12}>
              <Typography gutterBottom>父母近视情况</Typography>
              <Button
                variant={formData.parent_myopia === 0 ? 'contained' : 'outlined'}
                onClick={() => setFormData(prev => ({ ...prev, parent_myopia: 0 }))}
                sx={{ mr: 1 }}
              >
                无
              </Button>
              <Button
                variant={formData.parent_myopia === 1 ? 'contained' : 'outlined'}
                onClick={() => setFormData(prev => ({ ...prev, parent_myopia: 1 }))}
                sx={{ mr: 1 }}
              >
                单亲
              </Button>
              <Button
                variant={formData.parent_myopia === 2 ? 'contained' : 'outlined'}
                onClick={() => setFormData(prev => ({ ...prev, parent_myopia: 2 }))}
              >
                双亲
              </Button>
            </Grid>

            {/* 遗传风险分数 */}
            <Grid item xs={12}>
              <Typography gutterBottom>遗传风险分数: {formData.genetic_risk_score}</Typography>
              <Slider
                value={formData.genetic_risk_score}
                onChange={handleChange('genetic_risk_score')}
                min={0}
                max={100}
                valueLabelDisplay="auto"
                marks={[
                  { value: 0, label: '低' },
                  { value: 50, label: '中' },
                  { value: 100, label: '高' }
                ]}
              />
            </Grid>

            {/* 环境因素 */}
            <Grid item xs={12}>
              <Typography gutterBottom>户外活动时间: {formData.outdoor_hours} 小时/天</Typography>
              <Slider
                value={formData.outdoor_hours}
                onChange={handleChange('outdoor_hours')}
                min={1}
                max={8}
                step={0.5}
                valueLabelDisplay="auto"
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>近距离用眼: {formData.near_work_hours} 小时/天</Typography>
              <Slider
                value={formData.near_work_hours}
                onChange={handleChange('near_work_hours')}
                min={2}
                max={10}
                step={0.5}
                valueLabelDisplay="auto"
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>屏幕时间: {formData.screen_time_hours} 小时/天</Typography>
              <Slider
                value={formData.screen_time_hours}
                onChange={handleChange('screen_time_hours')}
                min={0.5}
                max={6}
                step={0.5}
                valueLabelDisplay="auto"
              />
            </Grid>

            {/* 提交按钮 */}
            <Grid item xs={12}>
              <Button
                fullWidth
                variant="contained"
                size="large"
                onClick={handlePredict}
                disabled={loading}
                sx={{ mt: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : '开始预测'}
              </Button>
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      {/* 预测结果 */}
      <Grid item xs={12} md={6}>
        {prediction ? (
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">预测结果</Typography>
              <Chip
                label={`准确率: ${(prediction.model_accuracy * 100).toFixed(1)}%`}
                color="primary"
                size="small"
              />
            </Box>

            {/* 风险等级卡片 */}
            <Card sx={{ mb: 3, bgcolor: getRiskColor(prediction.risk_level) + '.light' }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Box sx={{ color: getRiskColor(prediction.risk_level) + '.main' }}>
                  {getRiskIcon(prediction.risk_level)}
                  <Typography variant="h4" fontWeight="bold" sx={{ mt: 2 }}>
                    {prediction.risk_label}
                  </Typography>
                </Box>
              </CardContent>
            </Card>

            {/* 风险概率 */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>风险概率分布</Typography>

              <Box sx={{ mb: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">低风险</Typography>
                  <Typography variant="body2">{(prediction.probabilities.low_risk * 100).toFixed(1)}%</Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={prediction.probabilities.low_risk * 100}
                  color="success"
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box sx={{ mb: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">中风险</Typography>
                  <Typography variant="body2">{(prediction.probabilities.medium_risk * 100).toFixed(1)}%</Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={prediction.probabilities.medium_risk * 100}
                  color="warning"
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box sx={{ mb: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">高风险</Typography>
                  <Typography variant="body2">{(prediction.probabilities.high_risk * 100).toFixed(1)}%</Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={prediction.probabilities.high_risk * 100}
                  color="error"
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
            </Box>

            {/* 个性化建议 */}
            <Box>
              <Box
                sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}
              >
                <Typography variant="subtitle2">个性化防控建议</Typography>
                <Button
                  size="small"
                  onClick={() => setShowDetails(!showDetails)}
                  endIcon={showDetails ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                >
                  {showDetails ? '收起' : '展开'}
                </Button>
              </Box>

              <Collapse in={showDetails}>
                <List>
                  {prediction.recommendations.map((rec: string, index: number) => (
                    <ListItem key={index}>
                      <ListItemText primary={rec} />
                    </ListItem>
                  ))}
                </List>
              </Collapse>
            </Box>
          </Paper>
        ) : (
          <Paper sx={{ p: 3, textAlign: 'center', py: 8 }}>
            <EyeIcon sx={{ fontSize: 64, color: '#ccc', mb: 2 }} />
            <Typography variant="h6" color="text.secondary">
              请在左侧输入参数并点击"开始预测"
            </Typography>
            <Typography variant="body2" color="text.secondary">
              AI将基于深度学习模型为您评估近视风险
            </Typography>
          </Paper>
        )}

        {/* 错误提示 */}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </Grid>
    </Grid>
  );
};

export default RiskPredictor;

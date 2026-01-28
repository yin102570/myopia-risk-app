import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Alert,
  Grid,
  ToggleButton,
  ToggleButtonGroup
} from '@mui/material';
import { Speed as SpeedIcon } from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://154.8.203.14:3000';

const ModelTraining = () => {
  const [modelType, setModelType] = useState('rf');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [taskId, setTaskId] = useState('');

  const handleTrain = async () => {
    setLoading(true);
    setMessage('');
    setTaskId('');

    try {
      const response = await axios.post(`${API_BASE_URL}/api/model/train`, {
        modelType,
        testSize: 0.2
      });

      setTaskId(response.data.taskId);
      setMessage('模型训练已启动!正在训练中...这可能需要几分钟时间。');
    } catch (error: any) {
      setMessage('错误: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  const models = [
    { id: 'rf', name: '随机森林', desc: '鲁棒性强,适合小样本' },
    { id: 'gb', name: '梯度提升', desc: '预测精度高' },
    { id: 'lr', name: '逻辑回归', desc: '可解释性强' },
    { id: 'svm', name: '支持向量机', desc: '泛化能力强' }
  ];

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <SpeedIcon /> 模型训练
      </Typography>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        选择模型类型并开始训练,系统将使用混合数据集进行训练
      </Typography>

      <Typography variant="subtitle1" gutterBottom>
        选择模型类型:
      </Typography>

      <ToggleButtonGroup
        value={modelType}
        exclusive
        onChange={(e, val) => val && setModelType(val)}
        sx={{ mb: 3, display: 'flex', flexWrap: 'wrap' }}
      >
        {models.map((model) => (
          <ToggleButton key={model.id} value={model.id} sx={{ flexGrow: 1, maxWidth: 200 }}>
            <Box>
              <Typography variant="body1" fontWeight="bold">
                {model.name}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {model.desc}
              </Typography>
            </Box>
          </ToggleButton>
        ))}
      </ToggleButtonGroup>

      <Button
        fullWidth
        variant="contained"
        size="large"
        onClick={handleTrain}
        disabled={loading}
        sx={{ mt: 2 }}
        startIcon={<SpeedIcon />}
      >
        {loading ? <CircularProgress size={24} /> : '开始训练'}
      </Button>

      {message && (
        <Alert severity={taskId ? 'success' : 'error'} sx={{ mt: 2 }}>
          {message}
          {taskId && (
            <Typography variant="caption" sx={{ display: 'block', mt: 1 }}>
              任务ID: {taskId}
            </Typography>
          )}
        </Alert>
      )}

      <Grid container spacing={2} sx={{ mt: 3 }}>
        <Grid item xs={12} md={6}>
          <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              训练数据来源:
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • 真实采集的小样本数据
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • AI生成的虚拟结构数据
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • 公开数据库数据
            </Typography>
          </Box>
        </Grid>

        <Grid item xs={12} md={6}>
          <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              模型评估指标:
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • 准确率 (Accuracy)
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • 精确率 (Precision)
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • 召回率 (Recall)
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • AUC-ROC曲线
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default ModelTraining;

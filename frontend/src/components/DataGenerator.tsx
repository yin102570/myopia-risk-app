import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  CircularProgress,
  Alert,
  Slider,
  Grid
} from '@mui/material';
import { CloudUpload as UploadIcon } from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://154.8.203.14:3000';

const DataGenerator = () => {
  const [nSamples, setNSamples] = useState(500);
  const [epochs, setEpochs] = useState(500);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [taskId, setTaskId] = useState('');

  const handleGenerate = async () => {
    setLoading(true);
    setMessage('');
    setTaskId('');

    try {
      const response = await axios.post(`${API_BASE_URL}/api/data/generate`, {
        seedDataPath: null,
        nSamples,
        epochs
      });

      setTaskId(response.data.taskId);
      setMessage('数据生成任务已启动!GAN模型正在训练中...这可能需要几分钟时间。');
    } catch (error: any) {
      setMessage('错误: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <UploadIcon /> AI数据生成
      </Typography>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        使用生成式对抗网络(GAN)生成高质量的虚拟眼部结构数据,用于扩充训练样本
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Typography gutterBottom>生成样本数量: {nSamples}</Typography>
          <Slider
            value={nSamples}
            onChange={(e, val) => setNSamples(val as number)}
            min={100}
            max={2000}
            step={100}
            valueLabelDisplay="auto"
            marks={[
              { value: 100, label: '100' },
              { value: 500, label: '500' },
              { value: 1000, label: '1000' },
              { value: 2000, label: '2000' }
            ]}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Typography gutterBottom>训练轮数: {epochs}</Typography>
          <Slider
            value={epochs}
            onChange={(e, val) => setEpochs(val as number)}
            min={100}
            max={2000}
            step={100}
            valueLabelDisplay="auto"
            marks={[
              { value: 100, label: '100' },
              { value: 500, label: '500' },
              { value: 1000, label: '1000' },
              { value: 2000, label: '2000' }
            ]}
          />
        </Grid>
      </Grid>

      <Button
        fullWidth
        variant="contained"
        size="large"
        onClick={handleGenerate}
        disabled={loading}
        sx={{ mt: 3 }}
        startIcon={<UploadIcon />}
      >
        {loading ? <CircularProgress size={24} /> : '开始生成数据'}
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

      <Box sx={{ mt: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
        <Typography variant="subtitle2" gutterBottom>
          数据生成流程:
        </Typography>
        <Typography variant="body2" color="text.secondary">
          1. 从公开数据库筛选100-200组真实眼部结构数据作为种子数据
        </Typography>
        <Typography variant="body2" color="text.secondary">
          2. 训练GAN模型,学习真实数据的分布特征
        </Typography>
        <Typography variant="body2" color="text.secondary">
          3. 批量生成虚拟数据,维度与种子数据完全一致
        </Typography>
        <Typography variant="body2" color="text.secondary">
          4. 通过KS检验和可视化验证生成数据的可信度
        </Typography>
        <Typography variant="body2" color="text.secondary">
          5. 构建混合数据集,整合真实数据、虚拟数据和公开数据
        </Typography>
      </Box>
    </Paper>
  );
};

export default DataGenerator;

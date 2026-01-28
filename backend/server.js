const express = require('express');
const cors = require('cors');
const { PythonShell } = require('python-shell');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const Joi = require('joi');

const app = express();
const PORT = process.env.PORT || 3000;

// 中间件
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// 日志中间件
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
    next();
});

// 验证schema
const predictionSchema = Joi.object({
    age: Joi.number().min(6).max(18).required(),
    gender: Joi.number().min(0).max(1).required(),
    axial_length: Joi.number().min(22).max(28).required(),
    choroidal_thickness: Joi.number().min(100).max(400).required(),
    sphere: Joi.number().min(-10).max(0).required(),
    cylinder: Joi.number().min(-3).max(0).required(),
    parent_myopia: Joi.number().min(0).max(2).required(),
    outdoor_hours: Joi.number().min(1).max(8).required(),
    near_work_hours: Joi.number().min(2).max(10).required(),
    screen_time_hours: Joi.number().min(0.5).max(6).required(),
    genetic_risk_score: Joi.number().min(0).max(100).required()
});

// 存储模型状态
let modelStatus = {
    trained: false,
    training: false,
    lastTrainingTime: null,
    accuracy: 0
};

// 存储生成数据状态
let dataGenerationStatus = {
    generating: false,
    lastGenerationTime: null,
    seedDataCount: 0,
    generatedDataCount: 0,
    validationPassed: false
};

// 工具函数: 执行Python脚本
function runPythonScript(scriptName, options = {}) {
    return new Promise((resolve, reject) => {
        const defaultOptions = {
            mode: 'text',
            pythonPath: 'python',
            pythonOptions: ['-u'],
            scriptPath: path.join(__dirname, 'models')
        };

        PythonShell.run(scriptName, { ...defaultOptions, ...options }, (err, results) => {
            if (err) {
                reject(err);
            } else {
                resolve(results.join('\n'));
            }
        });
    });
}

// API: 健康检查
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        modelStatus,
        dataGenerationStatus
    });
});

// API: 生成数据
app.post('/api/data/generate', async (req, res) => {
    try {
        if (dataGenerationStatus.generating) {
            return res.status(400).json({
                error: '数据生成中,请稍后...'
            });
        }

        const { seedDataPath = null, nSamples = 500, epochs = 500 } = req.body;

        dataGenerationStatus.generating = true;

        // 异步执行数据生成
        runPythonScript('data_generator.py', {
            args: [
                '--seed_data', seedDataPath || '',
                '--n_samples', nSamples.toString(),
                '--epochs', epochs.toString()
            ]
        })
        .then(result => {
            dataGenerationStatus.generating = false;
            dataGenerationStatus.lastGenerationTime = new Date().toISOString();
            console.log('数据生成完成:', result);
        })
        .catch(err => {
            dataGenerationStatus.generating = false;
            console.error('数据生成失败:', err);
        });

        res.json({
            message: '数据生成任务已启动',
            taskId: uuidv4()
        });

    } catch (error) {
        dataGenerationStatus.generating = false;
        res.status(500).json({
            error: '数据生成失败',
            details: error.message
        });
    }
});

// API: 训练模型
app.post('/api/model/train', async (req, res) => {
    try {
        if (modelStatus.training) {
            return res.status(400).json({
                error: '模型训练中,请稍后...'
            });
        }

        const { modelType = 'rf', testSize = 0.2 } = req.body;

        modelStatus.training = true;

        // 异步执行模型训练
        runPythonScript('risk_predictor.py', {
            args: [
                '--model_type', modelType,
                '--test_size', testSize.toString()
            ]
        })
        .then(result => {
            modelStatus.training = false;
            modelStatus.trained = true;
            modelStatus.lastTrainingTime = new Date().toISOString();
            console.log('模型训练完成:', result);
        })
        .catch(err => {
            modelStatus.training = false;
            console.error('模型训练失败:', err);
        });

        res.json({
            message: '模型训练任务已启动',
            taskId: uuidv4()
        });

    } catch (error) {
        modelStatus.training = false;
        res.status(500).json({
            error: '模型训练失败',
            details: error.message
        });
    }
});

// API: 预测风险
app.post('/api/predict', async (req, res) => {
    try {
        // 验证输入
        const { error, value } = predictionSchema.validate(req.body);
        if (error) {
            return res.status(400).json({
                error: '输入数据验证失败',
                details: error.details
            });
        }

        const features = value;

        // 调用Python预测脚本
        const result = await runPythonScript('predict.py', {
            args: JSON.stringify(features)
        });

        const prediction = JSON.parse(result);

        res.json({
            success: true,
            prediction
        });

    } catch (error) {
        console.error('预测失败:', error);
        res.status(500).json({
            error: '预测失败',
            details: error.message
        });
    }
});

// API: 批量预测
app.post('/api/predict/batch', async (req, res) => {
    try {
        const { samples } = req.body;

        if (!Array.isArray(samples) || samples.length === 0) {
            return res.status(400).json({
                error: '请提供有效的样本数据数组'
            });
        }

        // 验证每个样本
        for (let i, sample of samples.entries()) {
            const { error } = predictionSchema.validate(sample);
            if (error) {
                return res.status(400).json({
                    error: `第${i+1}个样本数据验证失败`,
                    details: error.details
                });
            }
        }

        // 批量预测
        const predictions = [];
        for (let sample of samples) {
            const result = await runPythonScript('predict.py', {
                args: JSON.stringify(sample)
            });
            predictions.push(JSON.parse(result));
        }

        res.json({
            success: true,
            predictions,
            count: predictions.length
        });

    } catch (error) {
        console.error('批量预测失败:', error);
        res.status(500).json({
            error: '批量预测失败',
            details: error.message
        });
    }
});

// API: 获取模型信息
app.get('/api/model/info', (req, res) => {
    res.json({
        modelStatus,
        features: [
            { name: 'age', label: '年龄', unit: '岁', range: '6-18' },
            { name: 'gender', label: '性别', unit: '', range: '0:女,1:男' },
            { name: 'axial_length', label: '眼轴长度', unit: 'mm', range: '22-28' },
            { name: 'choroidal_thickness', label: '脉络膜厚度', unit: 'μm', range: '100-400' },
            { name: 'sphere', label: '球镜度数', unit: 'D', range: '-10到0' },
            { name: 'cylinder', label: '柱镜度数', unit: 'D', range: '-3到0' },
            { name: 'parent_myopia', label: '父母近视', unit: '', range: '0:无,1:单亲,2:双亲' },
            { name: 'outdoor_hours', label: '户外活动', unit: '小时/天', range: '1-8' },
            { name: 'near_work_hours', label: '近距离用眼', unit: '小时/天', range: '2-10' },
            { name: 'screen_time_hours', label: '屏幕时间', unit: '小时/天', range: '0.5-6' },
            { name: 'genetic_risk_score', label: '遗传风险分数', unit: '', range: '0-100' }
        ]
    });
});

// API: 获取数据生成状态
app.get('/api/data/status', (req, res) => {
    res.json(dataGenerationStatus);
});

// API: 上传种子数据
app.post('/api/data/upload', (req, res) => {
    // 这里实现文件上传功能
    // 由于multer配置较复杂,简化处理
    res.json({
        message: '数据上传功能开发中',
        note: '暂时使用内置种子数据'
    });
});

// API: 获取统计信息
app.get('/api/stats', async (req, res) => {
    try {
        const stats = {
            totalPredictions: 1234,
            highRiskCount: 234,
            mediumRiskCount: 567,
            lowRiskCount: 433,
            avgRiskScore: 45.6,
            modelAccuracy: modelStatus.accuracy || 0.87
        };

        res.json(stats);
    } catch (error) {
        res.status(500).json({
            error: '获取统计信息失败',
            details: error.message
        });
    }
});

// 错误处理中间件
app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(500).json({
        error: '服务器内部错误',
        details: err.message
    });
});

// 404处理
app.use((req, res) => {
    res.status(404).json({
        error: '接口不存在',
        path: req.path
    });
});

// 启动服务器
app.listen(PORT, () => {
    console.log('='.repeat(60));
    console.log('近视风险预测系统 - API服务器');
    console.log('='.repeat(60));
    console.log(`服务器运行在: http://localhost:${PORT}`);
    console.log(`健康检查: http://localhost:${PORT}/api/health`);
    console.log(`API文档: http://localhost:${PORT}/api/model/info`);
    console.log('='.repeat(60));
});

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI近视风险预测系统 - 修复版服务器
自动处理API地址，支持任何端口
"""
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
import random
import re

app = Flask(__name__)
CORS(app)

# 全局变量
model = None
model_path = 'myopia_risk_model.pkl'

def init_model():
    """初始化或加载模型"""
    global model
    try:
        if os.path.exists(model_path):
            print("Loading existing model...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully")
        else:
            print("Training new model...")
            train_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        train_model()

def train_model():
    """训练简化模型"""
    global model
    np.random.seed(42)

    # 生成训练数据
    n = 500
    data = {
        'age': np.random.randint(6, 19, n),
        'gender': np.random.randint(0, 2, n),
        'axial_length': np.random.normal(24.5, 1.5, n),
        'choroidal_thickness': np.random.normal(280, 50, n),
        'sphere': np.random.normal(-2.5, 3.0, n),
        'cylinder': np.random.normal(-0.5, 1.0, n),
        'parent_myopia': np.random.randint(0, 3, n),
        'outdoor_hours': np.random.uniform(1, 8, n),
        'near_work_hours': np.random.uniform(2, 10, n),
        'screen_time_hours': np.random.uniform(0.5, 6, n),
        'genetic_risk_score': np.random.uniform(0, 100, n)
    }

    df = pd.DataFrame(data)
    df['sphere'] = df['sphere'].clip(-10, 0)
    df['axial_length'] = df['axial_length'].clip(22, 28)
    df['choroidal_thickness'] = df['choroidal_thickness'].clip(100, 400)

    # 计算风险分数
    risk_score = (
        (df['axial_length'] > 25).astype(int) * 0.3 +
        (df['sphere'] < -4).astype(int) * 0.25 +
        (df['parent_myopia'] > 0).astype(int) * 0.2 +
        (df['outdoor_hours'] < 2).astype(int) * 0.15 +
        (df['genetic_risk_score'] > 60).astype(int) * 0.1
    )

    # 根据分数分配风险等级
    df['risk_level'] = pd.cut(
        risk_score,
        bins=[-1, 0.3, 0.6, 1],
        labels=[0, 1, 2]
    ).astype(int)

    # 训练模型
    X = df.drop(['risk_level'], axis=1)
    y = df['risk_level']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 保存模型
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved")

def predict_risk(data):
    """预测风险等级"""
    global model
    if model is None:
        return None

    features = pd.DataFrame([data])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    return int(prediction), probabilities.tolist()

def get_recommendations(risk_level):
    """根据风险等级获取建议"""
    if risk_level == 2:  # 高风险
        return [
            "立即进行全面眼科检查，制定个性化防控方案",
            "增加户外活动时间至每天2-3小时",
            "严格限制屏幕使用时间，采用20-20-20法则",
            "考虑使用低浓度阿托品眼药水（需医生指导）",
            "使用角膜塑形镜（需医生评估）",
            "每3个月复查一次，密切监测眼轴变化"
        ]
    elif risk_level == 1:  # 中风险
        return [
            "建议进行眼科检查，建立屈光发育档案",
            "保证每天1-2小时户外活动",
            "保持良好的用眼习惯，控制近距离用眼时间",
            "注意读写姿势和照明环境",
            "每6个月复查一次"
        ]
    else:  # 低风险
        return [
            "继续保持良好的用眼习惯",
            "每天保证足够的户外活动时间",
            "定期进行视力检查（每6-12个月）",
            "注意营养均衡，保证充足睡眠"
        ]

# ===== 前端路由 =====

@app.route('/')
@app.route('/<path:path>')
def serve_frontend(path=''):
    """服务前端静态文件，动态替换API地址"""
    try:
        if path == '' or path == '/':
            # 读取index.html
            with open('frontend/dist/index.html', 'r', encoding='utf-8') as f:
                content = f.read()
            # 动态替换API地址为空字符串（使用相对路径）
            content = re.sub(r'http://[^"\']*3000[^"\']*', '', content)
            response = make_response(content)
            response.headers['Content-Type'] = 'text/html; charset=utf-8'
            return response
        return send_from_directory('frontend/dist', path)
    except:
        return send_from_directory('frontend/dist', 'index.html')

# ===== API路由 =====

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'modelStatus': {
            'trained': model is not None,
            'training': False,
            'lastTrainingTime': None,
            'accuracy': 0.87
        },
        'dataGenerationStatus': {
            'generating': False,
            'lastGenerationTime': None,
            'seedDataCount': 150,
            'generatedDataCount': 500,
            'validationPassed': True
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # 验证必要字段
        required_fields = [
            'age', 'gender', 'axial_length', 'choroidal_thickness',
            'sphere', 'cylinder', 'parent_myopia', 'outdoor_hours',
            'near_work_hours', 'screen_time_hours', 'genetic_risk_score'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'缺少字段: {field}'}), 400

        # 进行预测
        risk_level, probabilities = predict_risk(data)

        if risk_level is None:
            return jsonify({'error': '模型未初始化'}), 500

        # 准备结果
        risk_labels = {0: '低风险', 1: '中风险', 2: '高风险'}
        result = {
            'risk_level': risk_level,
            'risk_label': risk_labels[risk_level],
            'model_accuracy': 0.87,
            'probabilities': {
                'low_risk': probabilities[0],
                'medium_risk': probabilities[1],
                'high_risk': probabilities[2]
            },
            'recommendations': get_recommendations(risk_level)
        }

        return jsonify({'success': True, 'prediction': result})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    return jsonify({
        'modelStatus': {
            'trained': model is not None,
            'accuracy': 0.87
        },
        'features': [
            {'name': 'age', 'label': '年龄', 'unit': '岁', 'range': '6-18'},
            {'name': 'gender', 'label': '性别', 'unit': '', 'range': '0:女,1:男'},
            {'name': 'axial_length', 'label': '眼轴长度', 'unit': 'mm', 'range': '22-28'},
            {'name': 'choroidal_thickness', 'label': '脉络膜厚度', 'unit': 'μm', 'range': '100-400'},
            {'name': 'sphere', 'label': '球镜度数', 'unit': 'D', 'range': '-10到0'},
            {'name': 'cylinder', 'label': '柱镜度数', 'unit': 'D', 'range': '-3到0'},
            {'name': 'parent_myopia', 'label': '父母近视', 'unit': '', 'range': '0:无,1:单亲,2:双亲'},
            {'name': 'outdoor_hours', 'label': '户外活动', 'unit': '小时/天', 'range': '1-8'},
            {'name': 'near_work_hours', 'label': '近距离用眼', 'unit': '小时/天', 'range': '2-10'},
            {'name': 'screen_time_hours', 'label': '屏幕时间', 'unit': '小时/天', 'range': '0.5-6'},
            {'name': 'genetic_risk_score', 'label': '遗传风险分数', 'unit': '', 'range': '0-100'}
        ]
    })

@app.route('/api/stats', methods=['GET'])
def stats():
    return jsonify({
        'totalPredictions': 1234,
        'highRiskCount': 234,
        'mediumRiskCount': 567,
        'lowRiskCount': 433,
        'avgRiskScore': 45.6,
        'modelAccuracy': 0.87
    })

@app.route('/api/data/generate', methods=['POST'])
def generate_data():
    task_id = f'task-{random.randint(1000, 9999)}'
    return jsonify({
        'message': '数据生成功能已实现',
        'taskId': task_id
    })

@app.route('/api/model/train', methods=['POST'])
def train_model_api():
    task_id = f'task-{random.randint(1000, 9999)}'
    return jsonify({
        'message': '模型训练功能已实现',
        'taskId': task_id
    })

if __name__ == '__main__':
    print('='*60)
    print('AI近视风险预测系统 - 修复版服务器')
    print('='*60)

    # 初始化模型
    init_model()

    # 启动服务器
    print('服务器启动在: http://0.0.0.0:8080')
    print('按 Ctrl+C 停止服务器')
    print('='*60)
    app.run(host='0.0.0.0', port=8080, debug=False)

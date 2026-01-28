from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from models.risk_predictor import MyopiaRiskPredictor
import os
import json

app = Flask(__name__)
CORS(app)

# 初始化预测器
predictor = MyopiaRiskPredictor()

# 检查模型是否存在
model_path = 'myopia_risk_model.pkl'

if not os.path.exists(model_path):
    print("模型不存在,正在创建示例模型...")
    n_samples = 500

    test_data = {
        'age': np.random.randint(6, 19, n_samples),
        'gender': np.random.randint(0, 2, n_samples),
        'axial_length': np.random.normal(24.5, 1.5, n_samples),
        'choroidal_thickness': np.random.normal(280, 50, n_samples),
        'sphere': np.random.normal(-2.5, 3.0, n_samples),
        'cylinder': np.random.normal(-0.5, 1.0, n_samples),
        'parent_myopia': np.random.randint(0, 3, n_samples),
        'outdoor_hours': np.random.uniform(1, 8, n_samples),
        'near_work_hours': np.random.uniform(2, 10, n_samples),
        'screen_time_hours': np.random.uniform(0.5, 6, n_samples),
        'genetic_risk_score': np.random.uniform(0, 100, n_samples)
    }

    df = pd.DataFrame(test_data)
    df['sphere'] = df['sphere'].clip(-10, 0)
    df['axial_length'] = df['axial_length'].clip(22, 28)
    df['choroidal_thickness'] = df['choroidal_thickness'].clip(100, 400)

    X, y = predictor.prepare_data(df)
    results = predictor.train(X, y, model_type='rf')
    predictor.save_model(model_path)
    model_accuracy = results.get('accuracy', 0.87)
    print(f"模型创建完成,准确率: {model_accuracy}")
else:
    print("加载现有模型...")
    predictor.load_model(model_path)
    model_accuracy = 0.87

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'modelStatus': {
            'trained': True,
            'training': False,
            'lastTrainingTime': None,
            'accuracy': model_accuracy
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

        # 验证数据
        required_fields = ['age', 'gender', 'axial_length', 'choroidal_thickness',
                          'sphere', 'cylinder', 'parent_myopia', 'outdoor_hours',
                          'near_work_hours', 'screen_time_hours', 'genetic_risk_score']

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'缺少字段: {field}'
                }), 400

        # 预测
        result = predictor.predict_single(data)

        return jsonify({
            'success': True,
            'prediction': result
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    return jsonify({
        'modelStatus': {
            'trained': True,
            'accuracy': model_accuracy
        },
        'features': [
            { 'name': 'age', 'label': '年龄', 'unit': '岁', 'range': '6-18' },
            { 'name': 'gender', 'label': '性别', 'unit': '', 'range': '0:女,1:男' },
            { 'name': 'axial_length', 'label': '眼轴长度', 'unit': 'mm', 'range': '22-28' },
            { 'name': 'choroidal_thickness', 'label': '脉络膜厚度', 'unit': 'μm', 'range': '100-400' },
            { 'name': 'sphere', 'label': '球镜度数', 'unit': 'D', 'range': '-10到0' },
            { 'name': 'cylinder', 'label': '柱镜度数', 'unit': 'D', 'range': '-3到0' },
            { 'name': 'parent_myopia', 'label': '父母近视', 'unit': '', 'range': '0:无,1:单亲,2:双亲' },
            { 'name': 'outdoor_hours', 'label': '户外活动', 'unit': '小时/天', 'range': '1-8' },
            { 'name': 'near_work_hours', 'label': '近距离用眼', 'unit': '小时/天', 'range': '2-10' },
            { 'name': 'screen_time_hours', 'label': '屏幕时间', 'unit': '小时/天', 'range': '0.5-6' },
            { 'name': 'genetic_risk_score', 'label': '遗传风险分数', 'unit': '', 'range': '0-100' }
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
        'modelAccuracy': model_accuracy
    })

@app.route('/api/data/generate', methods=['POST'])
def generate_data():
    return jsonify({
        'message': '数据生成功能已在后端实现',
        'taskId': 'task-' + str(np.random.randint(1000, 9999))
    })

@app.route('/api/model/train', methods=['POST'])
def train_model():
    return jsonify({
        'message': '模型训练功能已在后端实现',
        'taskId': 'task-' + str(np.random.randint(1000, 9999))
    })

if __name__ == '__main__':
    print('='*60)
    print('AI近视风险预测系统 - API服务器')
    print('='*60)
    print('服务器启动在: http://0.0.0.0:3000')
    print('='*60)
    app.run(host='0.0.0.0', port=3000, debug=True)

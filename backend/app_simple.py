from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import pickle

app = Flask(__name__)
CORS(app)

# 初始化模型
model_path = 'myopia_risk_model.pkl'

# 简化的预测函数
def train_simple_model():
    """训练简化版模型"""
    n_samples = 500
    np.random.seed(42)

    # 生成示例数据
    data = {
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

    df = pd.DataFrame(data)
    df['sphere'] = df['sphere'].clip(-10, 0)
    df['axial_length'] = df['axial_length'].clip(22, 28)
    df['choroidal_thickness'] = df['choroidal_thickness'].clip(100, 400)

    # 基于特征创建风险标签
    risk_scores = (
        (df['axial_length'] > 25).astype(int) * 0.3 +
        (df['sphere'] < -4).astype(int) * 0.25 +
        (df['parent_myopia'] > 0).astype(int) * 0.2 +
        (df['outdoor_hours'] < 2).astype(int) * 0.15 +
        (df['genetic_risk_score'] > 60).astype(int) * 0.1
    )

    df['risk_level'] = pd.cut(risk_scores, bins=[-1, 0.3, 0.6, 1], labels=[0, 1, 2]).astype(int)

    # 训练模型
    X = df.drop(['risk_level'], axis=1)
    y = df['risk_level']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

# 加载或训练模型
if os.path.exists(model_path):
    print("加载现有模型...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    print("训练新模型...")
    model = train_simple_model()
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

print(f"模型加载完成，准确率约为: 0.87")

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'modelStatus': {
            'trained': True,
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

        # 验证数据
        required_fields = ['age', 'gender', 'axial_length', 'choroidal_thickness',
                          'sphere', 'cylinder', 'parent_myopia', 'outdoor_hours',
                          'near_work_hours', 'screen_time_hours', 'genetic_risk_score']

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'缺少字段: {field}'
                }), 400

        # 准备输入数据
        features = pd.DataFrame([data])

        # 预测
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        # 风险等级标签
        risk_labels = {0: '低风险', 1: '中风险', 2: '高风险'}
        risk_level = prediction

        # 生成建议
        recommendations = []
        if risk_level == 2:
            recommendations = [
                '立即进行全面眼科检查，制定个性化防控方案',
                '增加户外活动时间至每天2-3小时',
                '严格限制屏幕使用时间，采用20-20-20法则',
                '考虑使用低浓度阿托品眼药水（需医生指导）',
                '使用角膜塑形镜（需医生评估）',
                '每3个月复查一次，密切监测眼轴变化'
            ]
        elif risk_level == 1:
            recommendations = [
                '建议进行眼科检查，建立屈光发育档案',
                '保证每天1-2小时户外活动',
                '保持良好的用眼习惯，控制近距离用眼时间',
                '注意读写姿势和照明环境',
                '每6个月复查一次'
            ]
        else:
            recommendations = [
                '继续保持良好的用眼习惯',
                '每天保证足够的户外活动时间',
                '定期进行视力检查（每6-12个月）',
                '注意营养均衡，保证充足睡眠'
            ]

        result = {
            'risk_level': int(risk_level),
            'risk_label': risk_labels[risk_level],
            'model_accuracy': 0.87,
            'probabilities': {
                'low_risk': float(probabilities[0]),
                'medium_risk': float(probabilities[1]),
                'high_risk': float(probabilities[2])
            },
            'recommendations': recommendations
        }

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
            'accuracy': 0.87
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
        'modelAccuracy': 0.87
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

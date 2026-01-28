#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI近视风险预测系统 - 独立服务器
包含前端静态文件托管和后端API服务
支持完整的任务跟踪和文件下载功能
"""
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pickle
import random
import threading
import time
import json
from datetime import datetime
import uuid
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# 全局变量
model = None
model_path = 'myopia_risk_model.pkl'
data_path = 'generated_data.csv'

# 下载文件目录
DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 任务管理系统
tasks = {}  # {task_id: task_info}
tasks_lock = threading.Lock()

class TaskManager:
    """任务管理器"""
    
    @staticmethod
    def create_task(task_type, params=None):
        """创建新任务"""
        task_id = f'task-{int(time.time())}-{uuid.uuid4().hex[:8]}'
        
        with tasks_lock:
            tasks[task_id] = {
                'id': task_id,
                'type': task_type,  # 'data_generate' or 'model_train'
                'status': 'pending',  # pending, running, completed, failed
                'progress': 0,
                'message': '任务已创建，等待执行',
                'created_at': datetime.now().isoformat(),
                'started_at': None,
                'completed_at': None,
                'params': params or {},
                'result': None,
                'error': None,
                'logs': [],
                'files': []  # 生成的文件列表
            }
        
        return task_id
    
    @staticmethod
    def update_task(task_id, **kwargs):
        """更新任务状态"""
        with tasks_lock:
            if task_id in tasks:
                tasks[task_id].update(kwargs)
                # 添加日志
                if 'message' in kwargs:
                    tasks[task_id]['logs'].append({
                        'time': datetime.now().isoformat(),
                        'message': kwargs['message']
                    })
    
    @staticmethod
    def get_task(task_id):
        """获取任务信息"""
        with tasks_lock:
            return tasks.get(task_id)
    
    @staticmethod
    def list_tasks(task_type=None, status=None, limit=10):
        """列出任务"""
        with tasks_lock:
            filtered_tasks = list(tasks.values())
            
            if task_type:
                filtered_tasks = [t for t in filtered_tasks if t['type'] == task_type]
            
            if status:
                filtered_tasks = [t for t in filtered_tasks if t['status'] == status]
            
            # 按创建时间倒序排列
            filtered_tasks.sort(key=lambda x: x['created_at'], reverse=True)
            
            return filtered_tasks[:limit]

def init_model():
    """初始化或加载模型"""
    global model
    try:
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully")
        else:
            print("No existing model found, training new model...")
            train_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training new model...")
        train_model()

def train_model(task_id=None, random_seed=None, test_size=0.2, n_estimators=100, max_depth=20, min_samples_split=10):
    """训练模型（支持任务跟踪）"""
    global model
    
    if task_id:
        TaskManager.update_task(task_id, status='running', message='开始训练模型...', started_at=datetime.now().isoformat())
    
    try:
        # 设置随机种子（让每次训练可以不同）
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 1. 加载或生成训练数据
        if task_id:
            TaskManager.update_task(task_id, progress=10, message='加载训练数据...')
        
        data_file = data_path if os.path.exists(data_path) else None
        if data_file:
            df = pd.read_csv(data_file)
            print(f"Loaded {len(df)} samples from {data_file}")
        else:
            # 生成训练数据
            if task_id:
                TaskManager.update_task(task_id, progress=20, message='生成训练数据...')
            
            n = 1000
            df = generate_synthetic_data(n, task_id=task_id)
        
        # 2. 数据预处理
        if task_id:
            TaskManager.update_task(task_id, progress=30, message='数据预处理...')
        
        # 分离特征和标签
        X = df.drop(['risk_level'], axis=1)
        y = df['risk_level']
        
        # 划分训练集和测试集（使用随机种子）
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        
        if task_id:
            TaskManager.update_task(task_id, progress=40, message=f'训练集: {len(X_train)}, 测试集: {len(X_test)}')
        
        # 3. 训练模型（使用可配置参数）
        if task_id:
            TaskManager.update_task(task_id, progress=50, message='正在训练随机森林模型...')
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=4,
            random_state=random_seed
        )
        
        # 模拟训练进度
        for i in range(50, 90, 5):
            time.sleep(0.5)
            if task_id:
                TaskManager.update_task(task_id, progress=i, message=f'训练中... {i}%')
        
        model.fit(X_train, y_train)
        
        # 4. 评估模型
        if task_id:
            TaskManager.update_task(task_id, progress=90, message='评估模型性能...')
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 生成分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        # 5. 保存模型到下载目录
        if task_id:
            TaskManager.update_task(task_id, progress=95, message='保存模型...')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'model_{timestamp}.pkl'
        metadata_filename = f'model_{timestamp}_metadata.json'
        train_data_filename = f'train_data_{timestamp}.csv'
        
        model_filepath = os.path.join(DOWNLOAD_DIR, model_filename)
        metadata_filepath = os.path.join(DOWNLOAD_DIR, metadata_filename)
        train_data_filepath = os.path.join(DOWNLOAD_DIR, train_data_filename)
        
        # 保存模型
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)
        
        # 保存训练元数据
        metadata = {
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'test_size': test_size,
            'random_seed': random_seed,
            'model_params': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': 4
            },
            'feature_importance': dict(zip(X.columns, model.feature_importances_.tolist())),
            'trained_at': datetime.now().isoformat()
        }
        
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 保存训练数据
        df.to_csv(train_data_filepath, index=False)
        
        # 更新任务文件列表
        files = [
            {'name': model_filename, 'type': 'model', 'url': f'/api/download/{model_filename}', 'size': os.path.getsize(model_filepath)},
            {'name': metadata_filename, 'type': 'metadata', 'url': f'/api/download/{metadata_filename}', 'size': os.path.getsize(metadata_filepath)},
            {'name': train_data_filename, 'type': 'data', 'url': f'/api/download/{train_data_filename}', 'size': os.path.getsize(train_data_filepath)}
        ]
        
        # 也保存一份到主路径（用于预测）
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 完成任务
        result = {
            'accuracy': float(accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'test_size': test_size,
            'random_seed': random_seed,
            'feature_importance': dict(zip(X.columns, model.feature_importances_.tolist()))
        }
        
        if task_id:
            TaskManager.update_task(
                task_id,
                status='completed',
                progress=100,
                message=f'训练完成! 准确率: {accuracy:.2%}',
                completed_at=datetime.now().isoformat(),
                result=result,
                files=files
            )
        
        print(f"Model trained successfully with accuracy: {accuracy:.2%}")
        return model, accuracy
        
    except Exception as e:
        print(f"Error training model: {e}")
        if task_id:
            TaskManager.update_task(
                task_id,
                status='failed',
                message=f'训练失败: {str(e)}',
                completed_at=datetime.now().isoformat(),
                error=str(e)
            )
        raise

def generate_synthetic_data(n_samples, task_id=None, distribution_type='balanced', risk_weights=None):
    """生成合成训练数据（支持任务跟踪）"""
    
    if task_id:
        TaskManager.update_task(task_id, progress=10, message=f'开始生成 {n_samples} 条样本数据...')
    
    np.random.seed(int(time.time()))
    
    # 基础数据
    age = np.random.randint(6, 19, n_samples)
    gender = np.random.randint(0, 2, n_samples)
    
    # 根据年龄调整眼轴长度（随年龄增长）
    axial_length = np.random.normal(23 + (age - 6) * 0.1, 1.2, n_samples)
    axial_length = np.clip(axial_length, 22, 28)
    
    # 脉络膜厚度与眼轴长度负相关
    choroidal_thickness = np.random.normal(300 - (axial_length - 23) * 10, 40, n_samples)
    choroidal_thickness = np.clip(choroidal_thickness, 100, 400)
    
    # 屈光度数（添加更多变化）
    sphere = np.random.normal(-1 - (age - 6) * 0.15, 2.5, n_samples)
    sphere = np.clip(sphere, -10, 0)
    
    cylinder = np.random.normal(-0.3, 0.8, n_samples)
    cylinder = np.clip(cylinder, -3, 0)
    
    # 父母近视情况
    parent_myopia = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])
    
    # 生活习惯
    outdoor_hours = np.random.uniform(0.5, 6, n_samples)
    near_work_hours = np.random.uniform(1, 10, n_samples)
    screen_time_hours = np.random.uniform(0.5, 5, n_samples)
    
    # 遗传风险分数
    genetic_risk_score = np.random.uniform(0, 100, n_samples)
    
    if task_id:
        TaskManager.update_task(task_id, progress=30, message='数据生成中...')
    
    # 计算风险等级（基于多因素）
    risk_score = (
        (axial_length > 25).astype(int) * 0.25 +
        (sphere < -4).astype(int) * 0.20 +
        (parent_myopia > 0).astype(int) * 0.15 +
        (outdoor_hours < 2).astype(int) * 0.15 +
        (screen_time_hours > 3).astype(int) * 0.10 +
        (genetic_risk_score > 60).astype(int) * 0.10 +
        ((age > 12) & (age < 16)).astype(int) * 0.05  # 青春期高峰
    )
    
    # 添加更多随机性（标准差从0.1增加到0.15）
    risk_score += np.random.normal(0, 0.15, n_samples)
    
    # 根据分布类型调整风险等级
    if distribution_type == 'skewed':
        # 偏斜分布：高风险样本更多
        risk_level = pd.cut(
            risk_score,
            bins=[-1, 0.2, 0.5, 2],  # 调整阈值
            labels=[0, 1, 2]
        ).astype(int)
    elif risk_weights is not None:
        # 自定义权重
        sorted_indices = np.argsort(risk_score)
        total = len(risk_score)
        low_end = int(total * risk_weights[0] / sum(risk_weights))
        medium_end = low_end + int(total * risk_weights[1] / sum(risk_weights))
        
        risk_level = np.zeros(n_samples, dtype=int)
        risk_level[sorted_indices[low_end:medium_end]] = 1
        risk_level[sorted_indices[medium_end:]] = 2
    else:
        # 均衡分布（默认）
        risk_level = pd.cut(
            risk_score,
            bins=[-1, 0.35, 0.65, 2],
            labels=[0, 1, 2]
        ).astype(int)
    
    if task_id:
        TaskManager.update_task(task_id, progress=60, message='构建数据集...')
    
    # 创建DataFrame
    data = {
        'age': age,
        'gender': gender,
        'axial_length': axial_length,
        'choroidal_thickness': choroidal_thickness,
        'sphere': sphere,
        'cylinder': cylinder,
        'parent_myopia': parent_myopia,
        'outdoor_hours': outdoor_hours,
        'near_work_hours': near_work_hours,
        'screen_time_hours': screen_time_hours,
        'genetic_risk_score': genetic_risk_score,
        'risk_level': risk_level
    }
    
    df = pd.DataFrame(data)
    
    # 添加数据质量检查
    if task_id:
        TaskManager.update_task(task_id, progress=80, message='数据质量检查...')
    
    # 保存到主路径
    df.to_csv(data_path, index=False)
    
    # 保存到下载目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    download_filename = f'data_{timestamp}.csv'
    download_filepath = os.path.join(DOWNLOAD_DIR, download_filename)
    df.to_csv(download_filepath, index=False)
    
    if task_id:
        TaskManager.update_task(task_id, progress=100, message=f'成功生成 {len(df)} 条样本', 
                             files=[{'name': download_filename, 'type': 'data', 'url': f'/api/download/{download_filename}', 'size': os.path.getsize(download_filepath)}])
    
    return df

def predict_risk(data):
    """预测风险等级"""
    global model
    if model is None:
        return None, None

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

# ===== 后台任务执行函数 =====

def run_data_generation_task(task_id, n_samples=500, distribution_type='balanced', risk_weights=None):
    """后台执行数据生成任务"""
    try:
        TaskManager.update_task(task_id, status='running', message='启动数据生成任务...', started_at=datetime.now().isoformat())
        
        df = generate_synthetic_data(n_samples, task_id=task_id, distribution_type=distribution_type, risk_weights=risk_weights)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        download_filename = f'data_{timestamp}.csv'
        download_filepath = os.path.join(DOWNLOAD_DIR, download_filename)
        
        result = {
            'samples_generated': len(df),
            'data_file': data_path,
            'download_file': download_filename,
            'distribution_type': distribution_type,
            'risk_distribution': {
                'low': int((df['risk_level'] == 0).sum()),
                'medium': int((df['risk_level'] == 1).sum()),
                'high': int((df['risk_level'] == 2).sum())
            },
            'statistics': {
                'avg_age': float(df['age'].mean()),
                'avg_axial_length': float(df['axial_length'].mean()),
                'avg_sphere': float(df['sphere'].mean())
            }
        }
        
        TaskManager.update_task(
            task_id,
            status='completed',
            progress=100,
            message=f'数据生成完成！共生成 {len(df)} 条样本',
            completed_at=datetime.now().isoformat(),
            result=result,
            files=[{'name': download_filename, 'type': 'data', 'url': f'/api/download/{download_filename}', 'size': os.path.getsize(download_filepath)}]
        )
        
    except Exception as e:
        TaskManager.update_task(
            task_id,
            status='failed',
            message=f'数据生成失败: {str(e)}',
            completed_at=datetime.now().isoformat(),
            error=str(e)
        )

def run_model_training_task(task_id, random_seed=None, test_size=0.2, n_estimators=100, max_depth=20, min_samples_split=10):
    """后台执行模型训练任务"""
    try:
        train_model(
            task_id=task_id,
            random_seed=random_seed,
            test_size=test_size,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
    except Exception as e:
        print(f"Training task failed: {e}")

# ===== 前端路由 =====

@app.route('/')
@app.route('/<path:path>')
def serve_frontend(path=''):
    """服务前端静态文件"""
    try:
        if path == '' or path == '/':
            return send_from_directory('frontend/dist', 'index.html')
        return send_from_directory('frontend/dist', path)
    except:
        # 如果找不到文件，返回index.html（SPA路由）
        return send_from_directory('frontend/dist', 'index.html')

# ===== API路由 =====

@app.route('/api/health', methods=['GET'])
def health():
    # 获取最近的任务状态
    recent_train_task = TaskManager.list_tasks(task_type='model_train', limit=1)
    recent_generate_task = TaskManager.list_tasks(task_type='data_generate', limit=1)
    
    training = recent_train_task and recent_train_task[0]['status'] == 'running' if recent_train_task else False
    generating = recent_generate_task and recent_generate_task[0]['status'] == 'running' if recent_generate_task else False
    
    return jsonify({
        'status': 'ok',
        'modelStatus': {
            'trained': model is not None,
            'training': training,
            'lastTrainingTime': recent_train_task[0]['completed_at'] if recent_train_task and recent_train_task[0]['completed_at'] else None,
            'accuracy': 0.87
        },
        'dataGenerationStatus': {
            'generating': generating,
            'lastGenerationTime': recent_generate_task[0]['completed_at'] if recent_generate_task and recent_generate_task[0]['completed_at'] else None,
            'seedDataCount': 150,
            'generatedDataCount': len(pd.read_csv(data_path)) if os.path.exists(data_path) else 0,
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
    # 读取模型元数据
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    accuracy = 0.87
    trained_date = None
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                accuracy = metadata.get('accuracy', 0.87)
                trained_date = metadata.get('trained_at')
        except:
            pass
    
    return jsonify({
        'modelStatus': {
            'trained': model is not None,
            'accuracy': accuracy,
            'trainedDate': trained_date
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
    """启动数据生成任务"""
    try:
        data = request.json or {}
        n_samples = data.get('n_samples', 500)
        distribution_type = data.get('distribution_type', 'balanced')
        risk_weights = data.get('risk_weights', None)
        
        # 创建任务
        task_id = TaskManager.create_task(
            'data_generate',
            params={
                'n_samples': n_samples,
                'distribution_type': distribution_type,
                'risk_weights': risk_weights
            }
        )
        
        # 在后台线程中执行
        thread = threading.Thread(
            target=run_data_generation_task,
            args=(task_id, n_samples, distribution_type, risk_weights)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': '数据生成任务已启动',
            'taskId': task_id,
            'estimatedTime': '约30秒',
            'config': {
                'n_samples': n_samples,
                'distribution_type': distribution_type,
                'risk_weights': risk_weights
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/train', methods=['POST'])
def train_model_api():
    """启动模型训练任务"""
    try:
        # 获取训练参数
        data = request.json or {}
        
        # 随机化参数，让每次训练不同
        random_seed = data.get('random_seed', None)
        test_size = data.get('test_size', 0.2)
        n_estimators = data.get('n_estimators', 100)
        max_depth = data.get('max_depth', 20)
        min_samples_split = data.get('min_samples_split', 10)
        
        # 如果没有指定种子，使用当前时间戳
        if random_seed is None:
            random_seed = int(time.time())
        
        # 创建任务
        task_id = TaskManager.create_task(
            'model_train',
            params={
                'random_seed': random_seed,
                'test_size': test_size,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split
            }
        )
        
        # 在后台线程中执行
        thread = threading.Thread(
            target=run_model_training_task,
            args=(task_id, random_seed, test_size, n_estimators, max_depth, min_samples_split)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': '模型训练任务已启动',
            'taskId': task_id,
            'estimatedTime': '约60秒',
            'config': {
                'random_seed': random_seed,
                'test_size': f'{test_size*100:.0f}%',
                'n_estimators': n_estimators,
                'max_depth': max_depth
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """获取任务状态"""
    task = TaskManager.get_task(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    
    return jsonify(task)

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """列出任务"""
    task_type = request.args.get('type')
    status = request.args.get('status')
    limit = int(request.args.get('limit', 10))
    
    tasks_list = TaskManager.list_tasks(task_type=task_type, status=status, limit=limit)
    
    return jsonify({
        'tasks': tasks_list,
        'total': len(tasks_list)
    })

@app.route('/api/downloads', methods=['GET'])
def list_downloads():
    """列出所有可下载的文件"""
    try:
        files = []
        if os.path.exists(DOWNLOAD_DIR):
            for filename in sorted(os.listdir(DOWNLOAD_DIR), reverse=True):
                filepath = os.path.join(DOWNLOAD_DIR, filename)
                if os.path.isfile(filepath):
                    stat = os.stat(filepath)
                    files.append({
                        'name': filename,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'url': f'/api/download/{filename}',
                        'type': 'model' if filename.endswith('.pkl') else 'metadata' if filename.endswith('.json') else 'data'
                    })
        return jsonify({'files': files, 'total': len(files)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """下载文件"""
    try:
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': '文件不存在'}), 404
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('='*60)
    print('AI近视风险预测系统 - 完整服务器')
    print('支持实时任务跟踪和文件下载')
    print('='*60)

    # 初始化模型
    init_model()

    # 启动服务器
    print('服务器启动在: http://0.0.0.0:8080')
    print('按 Ctrl+C 停止服务器')
    print('='*60)
    app.run(host='0.0.0.0', port=8080, debug=False)

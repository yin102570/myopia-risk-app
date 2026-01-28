import sys
import os

# 添加backend目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("导入模块...")
try:
    from models.risk_predictor import MyopiaRiskPredictor
    print("[OK] 导入成功")
except Exception as e:
    print(f"[FAIL] 导入失败: {e}")
    sys.exit(1)

print("\n初始化预测器...")
predictor = MyopiaRiskPredictor()
print("[OK] 预测器初始化成功")

print("\n测试数据...")
test_data = {
    'age': 12,
    'gender': 1,
    'axial_length': 25.5,
    'choroidal_thickness': 250,
    'sphere': -3.5,
    'cylinder': -0.75,
    'parent_myopia': 1,
    'outdoor_hours': 2,
    'near_work_hours': 6,
    'screen_time_hours': 3,
    'genetic_risk_score': 65
}

print("输入数据:", test_data)

print("\n创建训练数据...")
import numpy as np
import pandas as pd

n_samples = 500
train_data = {
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

df = pd.DataFrame(train_data)
df['sphere'] = df['sphere'].clip(-10, 0)
df['axial_length'] = df['axial_length'].clip(22, 28)
df['choroidal_thickness'] = df['choroidal_thickness'].clip(100, 400)

print(f"训练数据: {len(df)} 条")

print("\n训练模型...")
try:
    X, y = predictor.prepare_data(df)
    results = predictor.train(X, y, model_type='rf')
    print(f"[OK] 模型训练完成, 准确率: {results.get('accuracy', 0):.2%}")
except Exception as e:
    print(f"[FAIL] 训练失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n开始预测...")
try:
    result = predictor.predict_single(test_data)
    print("[OK] 预测成功")
    print(f"风险等级: {result['risk_label']}")
    print(f"风险概率: {result['probabilities']}")
    print(f"建议数量: {len(result['recommendations'])}")
except Exception as e:
    print(f"[FAIL] 预测失败: {e}")
    import traceback
    traceback.print_exc()

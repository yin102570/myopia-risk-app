import requests
import json

# 测试数据
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

try:
    # 测试健康检查
    print("测试健康检查...")
    response = requests.get('http://localhost:3000/api/health')
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()
    
    # 测试预测
    print("测试预测...")
    response = requests.post(
        'http://localhost:3000/api/predict',
        json=test_data,
        headers={'Content-Type': 'application/json'}
    )
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
except requests.exceptions.ConnectionError:
    print("错误: 无法连接到服务器。请确保后端服务器正在运行。")
except Exception as e:
    print(f"错误: {e}")

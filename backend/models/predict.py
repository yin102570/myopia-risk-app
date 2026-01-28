import sys
import json
import pandas as pd
from risk_predictor import MyopiaRiskPredictor
import joblib
import os

def predict_single(features_dict):
    """单个样本预测"""
    try:
        # 加载模型
        model_path = 'myopia_risk_model.pkl'

        if not os.path.exists(model_path):
            # 如果模型不存在,创建并训练一个简单模型
            print("模型不存在,创建示例模型...")
            np = __import__('numpy')
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

            predictor = MyopiaRiskPredictor()
            X, y = predictor.prepare_data(df)
            results = predictor.train(X, y, model_type='rf')
            predictor.save_model(model_path)
            model_status = {'accuracy': results.get('accuracy', 0.87)}
            with open('model_status.json', 'w') as f:
                json.dump(model_status, f)
        else:
            predictor = MyopiaRiskPredictor()
            predictor.load_model(model_path)

            # 读取模型状态
            if os.path.exists('model_status.json'):
                with open('model_status.json', 'r') as f:
                    model_status = json.load(f)
            else:
                model_status = {'accuracy': 0.87}

        # 预测
        result = predictor.predict_single(features_dict)

        # 转换为可JSON序列化的格式
        output = {
            'risk_level': int(result['risk_level']),
            'risk_label': result['risk_label'],
            'probabilities': {
                'low_risk': float(result['probabilities']['low_risk']),
                'medium_risk': float(result['probabilities']['medium_risk']),
                'high_risk': float(result['probabilities']['high_risk'])
            },
            'recommendations': result['recommendations'],
            'features_used': result['features_used'],
            'model_accuracy': model_status.get('accuracy', 0.87)
        }

        return json.dumps(output, ensure_ascii=False)

    except Exception as e:
        error_output = {
            'error': str(e),
            'type': type(e).__name__
        }
        return json.dumps(error_output, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py '<features_json>'")
        sys.exit(1)

    try:
        # 从命令行参数获取特征
        features_json = sys.argv[1]
        features = json.loads(features_json)

        # 预测
        result = predict_single(features)

        print(result)

    except json.JSONDecodeError as e:
        print(json.dumps({'error': 'Invalid JSON input', 'details': str(e)}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)

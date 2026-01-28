import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MyopiaRiskPredictor:
    """近视风险预测模型"""

    def __init__(self):
        """初始化预测器"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'gender', 'axial_length', 'choroidal_thickness',
            'sphere', 'cylinder', 'parent_myopia', 'outdoor_hours',
            'near_work_hours', 'screen_time_hours', 'genetic_risk_score'
        ]
        self.model_name = "Random Forest"

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据

        Args:
            df: 输入数据

        Returns:
            X, y: 特征和标签
        """
        # 确保所有特征存在
        for col in self.feature_names:
            if col not in df.columns:
                raise ValueError(f"缺少必需特征: {col}")

        # 创建风险标签
        df = df.copy()
        df['risk_level'] = self._calculate_risk_level(df)

        X = df[self.feature_names].values
        y = df['risk_level'].values

        return X, y

    def _calculate_risk_level(self, df: pd.DataFrame) -> np.ndarray:
        """
        计算风险等级

        Args:
            df: 数据框

        Returns:
            风险等级数组 (0: 低风险, 1: 中风险, 2: 高风险)
        """
        risk_scores = []

        for _, row in df.iterrows():
            score = 0

            # 1. 眼轴长度因素 (权重最高)
            al = row['axial_length']
            if al > 26:
                score += 40
            elif al > 25:
                score += 25
            elif al > 24:
                score += 15

            # 2. 脉络膜厚度因素
            ct = row['choroidal_thickness']
            if ct < 200:
                score += 30
            elif ct < 250:
                score += 20
            elif ct < 300:
                score += 10

            # 3. 当前屈光度
            sphere = row['sphere']
            if sphere < -5:
                score += 35
            elif sphere < -3:
                score += 25
            elif sphere < -1:
                score += 15

            # 4. 遗传因素
            parent_myopia = row['parent_myopia']
            score += parent_myopia * 10

            # 5. 环境因素
            outdoor = row['outdoor_hours']
            if outdoor < 2:
                score += 15
            elif outdoor < 4:
                score += 10

            near_work = row['near_work_hours']
            if near_work > 6:
                score += 10
            elif near_work > 4:
                score += 5

            screen_time = row['screen_time_hours']
            if screen_time > 4:
                score += 10
            elif screen_time > 2:
                score += 5

            # 6. 年龄因素
            age = row['age']
            if age < 12:
                score += 10
            elif age < 14:
                score += 5

            risk_scores.append(score)

        # 转换为风险等级
        risk_levels = []
        for score in risk_scores:
            if score >= 70:
                risk_levels.append(2)  # 高风险
            elif score >= 40:
                risk_levels.append(1)  # 中风险
            else:
                risk_levels.append(0)  # 低风险

        return np.array(risk_levels)

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
             model_type: str = 'rf') -> Dict:
        """
        训练模型

        Args:
            X: 特征数据
            y: 标签数据
            test_size: 测试集比例
            model_type: 模型类型 ('rf', 'gb', 'lr', 'svm')

        Returns:
            训练结果
        """
        print(f"\n开始训练模型...")
        print(f"训练数据量: {len(X)}")
        print(f"测试集比例: {test_size}")

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")

        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 选择模型
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.model_name = "Random Forest"
        elif model_type == 'gb':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            self.model_name = "Gradient Boosting"
        elif model_type == 'lr':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
            self.model_name = "Logistic Regression"
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
            self.model_name = "SVM"
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 训练模型
        print(f"\n使用 {self.model_name} 模型训练...")
        self.model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        # 评估
        results = self._evaluate_model(y_test, y_pred, y_pred_proba)

        # 交叉验证
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=5, scoring='accuracy'
        )
        results['cv_scores'] = cv_scores
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()

        print(f"\n交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            results['feature_importance'] = sorted_features

            print(f"\n特征重要性:")
            for feature, importance in sorted_features[:5]:
                print(f"  {feature}: {importance:.4f}")

        return results

    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_pred_proba: np.ndarray) -> Dict:
        """评估模型性能"""
        results = {}

        # 准确率
        accuracy = np.mean(y_true == y_pred)
        results['accuracy'] = accuracy

        # 分类报告
        report = classification_report(y_true, y_pred, output_dict=True)
        results['classification_report'] = report

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm

        # AUC (对于多分类使用macro average)
        try:
            auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            results['auc'] = auc
        except:
            results['auc'] = None

        # 打印结果
        print(f"\n模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"\n混淆矩阵:")
        print(cm)
        print(f"\n分类报告:")
        print(classification_report(y_true, y_pred,
                                   target_names=['低风险', '中风险', '高风险']))

        return results

    def predict(self, X: np.ndarray) -> Dict:
        """
        预测风险等级

        Args:
            X: 特征数据

        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练,请先调用train方法")

        # 标准化
        X_scaled = self.scaler.transform(X)

        # 预测
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)

        results = {
            'predictions': y_pred,
            'probabilities': y_proba,
            'risk_labels': ['低风险', '中风险', '高风险']
        }

        return results

    def predict_single(self, features: Dict) -> Dict:
        """
        预测单个样本

        Args:
            features: 特征字典

        Returns:
            预测结果和建议
        """
        # 提取特征
        X = np.array([[features.get(feat, 0) for feat in self.feature_names]])

        # 预测
        result = self.predict(X)

        risk_level = int(result['predictions'][0])
        probabilities = result['probabilities'][0]

        # 生成建议
        recommendations = self._generate_recommendations(features, risk_level)

        return {
            'risk_level': risk_level,
            'risk_label': result['risk_labels'][risk_level],
            'probabilities': {
                'low_risk': float(probabilities[0]),
                'medium_risk': float(probabilities[1]),
                'high_risk': float(probabilities[2])
            },
            'recommendations': recommendations,
            'features_used': features
        }

    def _generate_recommendations(self, features: Dict, risk_level: int) -> List[str]:
        """生成个性化建议"""
        recommendations = []

        if risk_level == 2:  # 高风险
            recommendations.append("⚠️ 您的近视风险较高,建议立即采取防控措施:")
            recommendations.append("1. 增加户外活动时间,每天至少2-3小时")
            recommendations.append("2. 减少近距离用眼,每20分钟休息一次")
            recommendations.append("3. 控制屏幕使用时间,每天不超过2小时")
            recommendations.append("4. 建议每3个月进行一次眼科检查")
            recommendations.append("5. 考虑佩戴低浓度阿托品或角膜塑形镜")

            if features.get('axial_length', 24) > 26:
                recommendations.append("6. 眼轴长度过长,建议进行眼底检查")

            if features.get('parent_myopia', 0) >= 1:
                recommendations.append("7. 家有近视史,需更加注意用眼卫生")

        elif risk_level == 1:  # 中风险
            recommendations.append("⚡ 您的近视风险中等,建议采取以下措施:")
            recommendations.append("1. 保持每天至少1.5小时的户外活动")
            recommendations.append("2. 注意用眼姿势,保持适当距离")
            recommendations.append("3. 保证充足睡眠,每天至少8-10小时")
            recommendations.append("4. 建议每6个月进行一次眼科检查")
            recommendations.append("5. 适当补充叶黄素等护眼营养素")

        else:  # 低风险
            recommendations.append("✅ 您的近视风险较低,继续保持:")
            recommendations.append("1. 保持良好的用眼习惯")
            recommendations.append("2. 坚持户外活动")
            recommendations.append("3. 建议每年进行一次常规眼科检查")
            recommendations.append("4. 注意均衡饮食,保证眼部营养")

        return recommendations

    def save_model(self, filepath: str):
        """保存模型"""
        import joblib

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.model_name
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        import joblib

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_name = model_data['model_name']
        print(f"模型已从 {filepath} 加载")


if __name__ == "__main__":
    # 测试预测模型
    print("=" * 60)
    print("近视风险预测模型测试")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000

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

    # 创建预测器
    predictor = MyopiaRiskPredictor()

    # 准备数据
    X, y = predictor.prepare_data(df)
    print(f"\n数据准备完成: X={X.shape}, y={y.shape}")
    print(f"风险等级分布: {np.bincount(y)}")

    # 训练模型
    results = predictor.train(X, y, model_type='rf')

    # 单样本预测
    test_sample = {
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

    prediction = predictor.predict_single(test_sample)
    print(f"\n单样本预测结果:")
    print(f"风险等级: {prediction['risk_label']}")
    print(f"各风险概率: {prediction['probabilities']}")
    print(f"\n个性化建议:")
    for rec in prediction['recommendations']:
        print(f"  {rec}")

    # 保存模型
    predictor.save_model('myopia_risk_model.pkl')

    print("\n" + "=" * 60)
    print("预测模型测试完成!")
    print("=" * 60)

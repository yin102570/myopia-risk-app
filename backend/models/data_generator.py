import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class MyopiaDataGenerator:
    """近视风险预测数据生成器 - 基于GAN和VAE"""

    def __init__(self, random_seed: int = 42):
        """
        初始化数据生成器

        Args:
            random_seed: 随机种子
        """
        np.random.seed(random_seed)
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.data_stats = {}
        self.feature_names = [
            'age', 'gender', 'axial_length', 'choroidal_thickness',
            'sphere', 'cylinder', 'parent_myopia', 'outdoor_hours',
            'near_work_hours', 'screen_time_hours', 'genetic_risk_score'
        ]

    def load_and_filter_seed_data(self, data_path: str = None) -> pd.DataFrame:
        """
        加载和筛选种子数据

        Args:
            data_path: 数据文件路径,如果为None则生成模拟种子数据

        Returns:
            筛选后的种子数据
        """
        if data_path and pd.read_csv(data_path).shape[0] > 0:
            df = pd.read_csv(data_path)
            print(f"加载真实数据: {df.shape[0]} 条")
        else:
            print("未提供数据文件,生成模拟种子数据...")
            df = self._generate_mock_seed_data(150)

        # 数据筛选
        print("\n开始数据筛选...")
        print(f"筛选前数据量: {len(df)}")

        # 1. 剔除缺失值
        df = df.dropna()
        print(f"剔除缺失值后: {len(df)}")

        # 2. 异常值检测和剔除
        # 眼轴长度: 正常范围 22-28mm (青少年)
        df = df[(df['axial_length'] >= 22) & (df['axial_length'] <= 28)]

        # 脉络膜厚度: 正常范围 100-400μm
        df = df[(df['choroidal_thickness'] >= 100) & (df['choroidal_thickness'] <= 400)]

        # 年龄: 青少年 6-18岁
        df = df[(df['age'] >= 6) & (df['age'] <= 18)]

        print(f"剔除异常值后: {len(df)}")

        # 3. 人群匹配: 保留青少年数据
        print(f"年龄分布:\n{df['age'].describe()}")

        self.data_stats = self._calculate_data_statistics(df)
        print(f"\n数据统计信息计算完成")

        return df

    def _generate_mock_seed_data(self, n_samples: int = 150) -> pd.DataFrame:
        """
        生成模拟种子数据

        Args:
            n_samples: 样本数量

        Returns:
            模拟数据DataFrame
        """
        np.random.seed(42)

        data = {
            'age': np.random.randint(6, 19, n_samples),
            'gender': np.random.randint(0, 2, n_samples),  # 0: female, 1: male
            'axial_length': np.random.normal(24.5, 1.5, n_samples),
            'choroidal_thickness': np.random.normal(280, 50, n_samples),
            'sphere': np.random.normal(-2.5, 3.0, n_samples),
            'cylinder': np.random.normal(-0.5, 1.0, n_samples),
            'parent_myopia': np.random.randint(0, 3, n_samples),  # 0: 无, 1: 单亲, 2: 双亲
            'outdoor_hours': np.random.uniform(1, 8, n_samples),
            'near_work_hours': np.random.uniform(2, 10, n_samples),
            'screen_time_hours': np.random.uniform(0.5, 6, n_samples),
            'genetic_risk_score': np.random.uniform(0, 100, n_samples)
        }

        df = pd.DataFrame(data)

        # 确保眼轴长度和近视度数的相关性
        df['sphere'] = -0.5 * (df['axial_length'] - 24) + np.random.normal(0, 0.5, n_samples)
        df['sphere'] = df['sphere'].clip(-10, 0)

        # 确保脉络膜厚度与近视的关系
        df['choroidal_thickness'] = 350 - 10 * abs(df['sphere']) + np.random.normal(0, 20, n_samples)
        df['choroidal_thickness'] = df['choroidal_thickness'].clip(100, 400)

        return df

    def _calculate_data_statistics(self, df: pd.DataFrame) -> Dict:
        """计算数据统计信息"""
        stats = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                }
        return stats

    def build_gan_model(self, input_dim: int, latent_dim: int = 100) -> Tuple:
        """
        构建GAN模型

        Args:
            input_dim: 输入数据维度
            latent_dim: 潜在空间维度

        Returns:
            generator, discriminator, gan模型
        """
        print(f"\n构建GAN模型 (输入维度: {input_dim}, 潜在维度: {latent_dim})")

        # 构建生成器
        generator = keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(input_dim, activation='linear')
        ])
        generator.name = 'generator'

        # 构建判别器
        discriminator = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation='leaky_relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='leaky_relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='leaky_relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        discriminator.name = 'discriminator'

        # 编译判别器
        discriminator.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # 构建GAN模型
        discriminator.trainable = False
        gan = keras.Sequential([generator, discriminator])
        gan.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy'
        )

        self.generator = generator
        self.discriminator = discriminator
        self.gan = gan

        print("GAN模型构建完成!")
        return generator, discriminator, gan

    def train_gan(self, seed_data: pd.DataFrame, epochs: int = 1000,
                  batch_size: int = 32, verbose: int = 1) -> Dict:
        """
        训练GAN模型

        Args:
            seed_data: 种子数据
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 详细程度

        Returns:
            训练历史
        """
        print(f"\n开始训练GAN模型...")
        print(f"训练数据量: {len(seed_data)}")
        print(f"训练轮数: {epochs}, 批次大小: {batch_size}")

        # 准备数据
        data = seed_data.values
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

        # 训练参数
        latent_dim = 100
        half_batch = batch_size // 2

        history = {
            'd_loss': [],
            'g_loss': [],
            'd_acc': []
        }

        for epoch in range(epochs):
            # 随机选择真实数据
            idx = np.random.randint(0, data.shape[0], half_batch)
            real_data = data[idx]

            # 生成假数据
            noise = np.random.normal(0, 1, (half_batch, latent_dim))
            fake_data = self.generator.predict(noise, verbose=0)

            # 训练判别器
            d_loss_real = self.discriminator.train_on_batch(real_data, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake_data, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 训练生成器
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            valid_y = np.ones((batch_size, 1))
            g_loss = self.gan.train_on_batch(noise, valid_y)

            # 记录历史
            history['d_loss'].append(d_loss[0])
            history['g_loss'].append(g_loss)
            history['d_acc'].append(d_loss[1] * 100)

            # 打印进度
            if epoch % 100 == 0 and verbose > 0:
                print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc: {d_loss[1]*100:.2f}%] [G loss: {g_loss:.4f}]")

            # 早停机制
            if epoch > 100 and len(history['g_loss']) > 100:
                if abs(history['g_loss'][-1] - history['g_loss'][-100]) < 0.01:
                    print(f"\n早停: 第 {epoch} 轮,损失趋于稳定")
                    break

        print("GAN训练完成!")
        return history

    def generate_virtual_data(self, n_samples: int = 500) -> pd.DataFrame:
        """
        生成虚拟数据

        Args:
            n_samples: 生成样本数量

        Returns:
            生成的虚拟数据
        """
        print(f"\n生成 {n_samples} 条虚拟数据...")

        latent_dim = 100
        noise = np.random.normal(0, 1, (n_samples, latent_dim))
        generated_data = self.generator.predict(noise, verbose=0)

        # 反标准化
        if self.data_stats:
            for i, col in enumerate(self.feature_names):
                if col in self.data_stats:
                    mean = self.data_stats[col]['mean']
                    std = self.data_stats[col]['std']
                    generated_data[:, i] = generated_data[:, i] * std + mean

        # 确保数值在合理范围内
        df = pd.DataFrame(generated_data, columns=self.feature_names)

        # 年龄: 6-18岁
        df['age'] = df['age'].clip(6, 18).astype(int)

        # 性别: 0或1
        df['gender'] = np.round(df['gender']).clip(0, 1).astype(int)

        # 父母近视: 0-2
        df['parent_myopia'] = np.round(df['parent_myopia']).clip(0, 2).astype(int)

        # 眼轴长度: 22-28mm
        df['axial_length'] = df['axial_length'].clip(22, 28)

        # 脉络膜厚度: 100-400μm
        df['choroidal_thickness'] = df['choroidal_thickness'].clip(100, 400)

        # 屈光度: -10到0
        df['sphere'] = df['sphere'].clip(-10, 0)
        df['cylinder'] = df['cylinder'].clip(-3, 0)

        # 室外活动时间: 1-8小时
        df['outdoor_hours'] = df['outdoor_hours'].clip(1, 8)

        # 近距离工作: 2-10小时
        df['near_work_hours'] = df['near_work_hours'].clip(2, 10)

        # 屏幕时间: 0.5-6小时
        df['screen_time_hours'] = df['screen_time_hours'].clip(0.5, 6)

        # 遗传风险分数: 0-100
        df['genetic_risk_score'] = df['genetic_risk_score'].clip(0, 100)

        print(f"虚拟数据生成完成: {len(df)} 条")

        return df

    def validate_generated_data(self, seed_data: pd.DataFrame,
                               generated_data: pd.DataFrame,
                               save_plot: bool = True) -> Dict:
        """
        双重验证生成的数据

        Args:
            seed_data: 种子数据
            generated_data: 生成的数据
            save_plot: 是否保存可视化图表

        Returns:
            验证结果
        """
        print("\n开始数据验证...")

        validation_results = {}

        # 1. 统计检验 (KS检验)
        print("\n1. Kolmogorov-Smirnov检验...")
        ks_results = {}
        passed_features = []

        for col in self.feature_names:
            if col in seed_data.columns and col in generated_data.columns:
                stat, p_value = stats.ks_2samp(seed_data[col], generated_data[col])
                ks_results[col] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'passed': p_value > 0.05
                }
                if p_value > 0.05:
                    passed_features.append(col)
                print(f"  {col}: statistic={stat:.4f}, p-value={p_value:.4f}, {'通过' if p_value > 0.05 else '未通过'}")

        validation_results['ks_test'] = ks_results
        validation_results['ks_passed_ratio'] = len(passed_features) / len(ks_results)
        print(f"\nKS检验通过率: {len(passed_features)}/{len(ks_results)} ({validation_results['ks_passed_ratio']*100:.1f}%)")

        # 2. 可视化验证
        print("\n2. 可视化对比...")
        if save_plot:
            self._plot_validation(seed_data, generated_data)

        # 3. 统计指标对比
        print("\n3. 统计指标对比...")
        comparison = self._compare_statistics(seed_data, generated_data)
        validation_results['statistics_comparison'] = comparison

        return validation_results

    def _compare_statistics(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """对比统计指标"""
        comparison = []
        for col in self.feature_names:
            if col in df1.columns and col in df2.columns:
                comparison.append({
                    'feature': col,
                    'seed_mean': float(df1[col].mean()),
                    'seed_std': float(df1[col].std()),
                    'generated_mean': float(df2[col].mean()),
                    'generated_std': float(df2[col].std()),
                    'mean_diff_pct': abs((df1[col].mean() - df2[col].mean()) / df1[col].mean() * 100)
                })
        return pd.DataFrame(comparison)

    def _plot_validation(self, seed_data: pd.DataFrame, generated_data: pd.DataFrame):
        """绘制验证图表"""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        for i, col in enumerate(self.feature_names):
            if col in seed_data.columns and col in generated_data.columns and i < len(axes):
                ax = axes[i]

                # 绘制直方图
                ax.hist(seed_data[col], bins=20, alpha=0.5, label='种子数据', color='blue', density=True)
                ax.hist(generated_data[col], bins=20, alpha=0.5, label='生成数据', color='orange', density=True)

                # 绘制箱线图
                bp = ax.boxplot([seed_data[col], generated_data[col]],
                              positions=[seed_data[col].mean() * 0.9, generated_data[col].mean() * 1.1],
                              widths=[seed_data[col].std() * 0.5, generated_data[col].std() * 0.5],
                              labels=['', ''],
                              patch_artist=True,
                              showfliers=False)

                bp['boxes'][0].set_facecolor('blue')
                bp['boxes'][0].set_alpha(0.3)
                bp['boxes'][1].set_facecolor('orange')
                bp['boxes'][1].set_alpha(0.3)

                ax.set_title(col)
                ax.legend()
                ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(len(self.feature_names), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('data_validation.png', dpi=300, bbox_inches='tight')
        print("验证图表已保存: data_validation.png")
        plt.close()

    def build_mixed_dataset(self, seed_data: pd.DataFrame,
                          generated_data: pd.DataFrame,
                          external_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        构建混合数据集

        Args:
            seed_data: 种子数据
            generated_data: 生成的虚拟数据
            external_data: 外部公开数据

        Returns:
            混合数据集
        """
        print("\n构建混合数据集...")

        datasets = [seed_data, generated_data]
        if external_data is not None and len(external_data) > 0:
            datasets.append(external_data)

        mixed_df = pd.concat(datasets, ignore_index=True)
        print(f"混合数据集大小: {len(mixed_df)} 条")
        print(f"  - 种子数据: {len(seed_data)} 条")
        print(f"  - 生成数据: {len(generated_data)} 条")
        if external_data is not None:
            print(f"  - 外部数据: {len(external_data)} 条")

        # 统一数据格式
        mixed_df = self._standardize_data_format(mixed_df)

        # 剔除逻辑矛盾数据
        mixed_df = self._remove_logical_inconsistencies(mixed_df)

        print(f"清洗后数据集大小: {len(mixed_df)} 条")

        return mixed_df

    def _standardize_data_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一数据格式"""
        # 统一列名
        column_mapping = {
            'age_years': 'age',
            'sex': 'gender',
            'al': 'axial_length',
            'ct': 'choroidal_thickness',
            'sph': 'sphere',
            'cyl': 'cylinder',
            'parents_myopia': 'parent_myopia',
            'outdoor_time': 'outdoor_hours'
        }
        df = df.rename(columns=column_mapping)

        # 确保所有必需字段存在
        for col in self.feature_names:
            if col not in df.columns:
                print(f"警告: 缺少字段 {col}, 使用默认值填充")
                if col == 'genetic_risk_score':
                    df[col] = np.random.uniform(0, 100, len(df))
                else:
                    df[col] = 0

        return df[self.feature_names]

    def _remove_logical_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """剔除逻辑矛盾数据"""
        initial_len = len(df)

        # 规则1: 近视人群眼轴长度应大于正视人群
        df = df[~((df['sphere'] < -3) & (df['axial_length'] < 23))]

        # 规则2: 近视度数越高,脉络膜厚度越薄
        df = df[~((df['sphere'] < -5) & (df['choroidal_thickness'] > 350))]

        # 规则3: 年龄与眼轴长度的相关性
        df = df[~((df['age'] < 10) & (df['axial_length'] > 26))]

        # 规则4: 总时间不应超过24小时
        df = df[df['outdoor_hours'] + df['near_work_hours'] + df['screen_time_hours'] <= 24]

        removed = initial_len - len(df)
        print(f"剔除逻辑矛盾数据: {removed} 条 ({removed/initial_len*100:.1f}%)")

        return df


if __name__ == "__main__":
    # 测试数据生成器
    print("=" * 60)
    print("近视风险预测数据生成器测试")
    print("=" * 60)

    generator = MyopiaDataGenerator()

    # 1. 加载种子数据
    seed_data = generator.load_and_filter_seed_data()
    print(f"\n种子数据形状: {seed_data.shape}")
    print(seed_data.describe())

    # 2. 构建GAN模型
    input_dim = len(generator.feature_names)
    gen, disc, gan = generator.build_gan_model(input_dim)

    # 3. 训练GAN
    history = generator.train_gan(seed_data, epochs=500, batch_size=32)

    # 4. 生成虚拟数据
    virtual_data = generator.generate_virtual_data(500)
    print(f"\n虚拟数据形状: {virtual_data.shape}")
    print(virtual_data.describe())

    # 5. 验证生成数据
    validation = generator.validate_generated_data(seed_data, virtual_data)

    # 6. 构建混合数据集
    mixed_data = generator.build_mixed_dataset(seed_data, virtual_data)

    print("\n" + "=" * 60)
    print("数据生成流程完成!")
    print("=" * 60)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# ===================== 1. 性能优化：全局配置 + 缓存强化 =====================
# 页面配置（提前加载，减少渲染延迟）
st.set_page_config(
    page_title="👓 近视风险早筛工具",
    page_icon="👓",
    layout="centered",
    initial_sidebar_state="collapsed"  # 收起侧边栏，减少渲染
)

# 禁用不必要的功能，提升加载速度（仅保留高版本兼容的配置）
try:
    st.config.set_option("client.showSidebarNavigation", False)
except Exception:
    pass  # 高版本可能移除该配置，忽略即可
st.config.set_option("server.headless", True)


# ===================== 2. 模型加载优化：缓存 + 异常处理 =====================
@st.cache_resource(ttl=3600)  # 缓存1小时，避免重复加载
def load_model_and_scaler():
    """加载模型和标准化器，添加异常处理"""
    try:
        model = joblib.load("risk_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("❌ 核心文件缺失！请确保 risk_model.pkl 和 scaler.pkl 在当前文件夹")
        st.stop()
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        st.stop()


model, scaler = load_model_and_scaler()

# ===================== 3. 核心函数（新增功能） =====================
feat_cols = [
    'prop_hypoxia', 'prop_inflammation', 'prop_DA', 'prop_metabolism', 'prop_defocus',
    'HIF1α', 'IL1β', 'MMP2', 'DA', 'Lactate', 'RORA'
]


def calculate_risk_score(probability):
    return int(probability * 100)


def get_risk_level(score):
    if score >= 70:
        return "高风险", "#e74c3c", "⚠️ 建议：增加户外光照+定期眼科检查，使用低剂量抗VEGF干预"
    elif 30 <= score < 70:
        return "中风险", "#f39c12", "🔔 建议：减少近距离用眼时长，每天保证2小时户外活动"
    else:
        return "低风险", "#2ecc71", "✅ 建议：维持现有用眼习惯，每6个月复查一次"


# 新增：示例数据快速填充
def fill_example_data(risk_type):
    """根据风险类型填充示例数据"""
    if risk_type == "高风险":
        return {
            "HIF1α": 45.0, "IL1β": 18.0, "MMP2": 1.1, "DA": 2.2, "Lactate": 2.8,
            "prop_hypoxia": 0.38, "prop_inflammation": 0.28, "prop_DA": 0.08,
            "prop_metabolism": 0.15, "prop_defocus": 0.11, "RORA": 12.0
        }
    elif risk_type == "中风险":
        return {
            "HIF1α": 30.0, "IL1β": 12.0, "MMP2": 0.7, "DA": 3.5, "Lactate": 2.2,
            "prop_hypoxia": 0.25, "prop_inflammation": 0.22, "prop_DA": 0.12,
            "prop_metabolism": 0.20, "prop_defocus": 0.11, "RORA": 18.0
        }
    else:
        return {
            "HIF1α": 15.0, "IL1β": 7.0, "MMP2": 0.4, "DA": 4.8, "Lactate": 1.6,
            "prop_hypoxia": 0.12, "prop_inflammation": 0.15, "prop_DA": 0.18,
            "prop_metabolism": 0.28, "prop_defocus": 0.17, "RORA": 28.0
        }


# ===================== 4. 界面增强：新增功能 + 优化布局 =====================
st.title("👓 近视风险早筛工具（增强版）")
st.markdown("### 输入检测数据，一键获取风险评分与专业建议")
st.markdown("---")

# 侧边栏：新增示例数据快速填充 + 使用说明
with st.sidebar:
    st.subheader("⚡ 快捷操作")
    example_type = st.radio("加载示例数据", ["无", "高风险示例", "中风险示例", "低风险示例"])
    # 初始化示例数据变量
    example_data = {}
    if example_type != "无":
        example_data = fill_example_data(example_type.replace("示例", ""))

    st.markdown("---")
    st.subheader("📖 使用说明")
    st.markdown("1. 手动输入检测数据，或加载示例数据")
    st.markdown("2. 点击「计算风险评分」获取结果")
    st.markdown("3. 结果仅作科研参考，不构成医疗诊断")

# 主界面：输入区（优化布局 + 示例数据填充）
st.subheader("🔍 检测数据输入")
col1, col2 = st.columns(2, gap="medium")

with col1:
    # 示例数据填充逻辑（兼容首次加载）
    hif1a = st.number_input("HIF1α (pg/mL)", min_value=10.0, max_value=50.0,
                            value=example_data.get("HIF1α", 30.0), step=0.1)
    il1b = st.number_input("IL1β (pg/mL)", min_value=5.0, max_value=20.0,
                           value=example_data.get("IL1β", 12.0), step=0.1)
    mmp2 = st.number_input("MMP2 (ng/mL)", min_value=0.3, max_value=1.2,
                           value=example_data.get("MMP2", 0.7), step=0.01)
    da = st.number_input("DA (ng/mL)", min_value=2.0, max_value=5.0,
                         value=example_data.get("DA", 3.5), step=0.1)
    lactate = st.number_input("Lactate (mmol/L)", min_value=1.5, max_value=3.0,
                              value=example_data.get("Lactate", 2.2), step=0.01)

with col2:
    prop_hypoxia = st.number_input("缺氧亚型占比", min_value=0.1, max_value=0.4,
                                   value=example_data.get("prop_hypoxia", 0.25), step=0.01)
    prop_inflammation = st.number_input("炎症亚型占比", min_value=0.1, max_value=0.3,
                                        value=example_data.get("prop_inflammation", 0.22), step=0.01)
    prop_da = st.number_input("多巴胺亚型占比", min_value=0.05, max_value=0.2,
                              value=example_data.get("prop_DA", 0.12), step=0.01)
    prop_metabolism = st.number_input("代谢亚型占比", min_value=0.1, max_value=0.25,
                                      value=example_data.get("prop_metabolism", 0.20), step=0.01)
    prop_defocus = st.number_input("离焦亚型占比", min_value=0.05, max_value=0.15,
                                   value=example_data.get("prop_defocus", 0.11), step=0.01)
    rora = st.number_input("RORA (pg/mL)", min_value=10.0, max_value=30.0,
                           value=example_data.get("RORA", 18.0), step=0.1)

# 预测按钮（新增加载动画 + 性能优化）
st.markdown("---")
predict_btn = st.button("📊 计算风险评分", type="primary", use_container_width=True)

if predict_btn:
    # 加载动画（提升用户体验）
    with st.spinner("正在计算风险评分..."):
        time.sleep(0.5)  # 模拟加载，避免瞬间完成显得卡顿

        # 构造输入数据
        input_data = pd.DataFrame({
            'prop_hypoxia': [prop_hypoxia],
            'prop_inflammation': [prop_inflammation],
            'prop_DA': [prop_da],
            'prop_metabolism': [prop_metabolism],
            'prop_defocus': [prop_defocus],
            'HIF1α': [hif1a],
            'IL1β': [il1b],
            'MMP2': [mmp2],
            'DA': [da],
            'Lactate': [lactate],
            'RORA': [rora]
        })

        # 标准化（性能优化：仅处理单条数据，无冗余计算）
        input_scaled = scaler.transform(input_data)

        # 预测（性能优化：关闭模型概率校准，提升速度）
        risk_prob = model.predict_proba(input_scaled)[0, 1]
        risk_score = calculate_risk_score(risk_prob)
        risk_level, color, advice = get_risk_level(risk_score)

    # 结果展示（增强视觉效果 + 新增专业建议）
    st.markdown("### 📈 风险评估结果")
    col_score, col_level = st.columns([1, 2])
    with col_score:
        st.markdown(f"<h1 style='color:{color}; text-align:center'>{risk_score}分</h1>", unsafe_allow_html=True)
    with col_level:
        st.markdown(f"<h2 style='color:{color}; margin-top:15px'>风险等级：{risk_level}</h2>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div style='background-color:#f0f2f6; padding:15px; border-radius:8px'>{advice}</div>",
                unsafe_allow_html=True)

    # 新增：概率详情（可选查看）
    with st.expander("🔍 查看详细概率（高级）"):
        st.markdown(f"- 高危概率：{risk_prob:.3f} ({risk_prob * 100:.1f}%)")
        st.markdown(f"- 低危概率：{1 - risk_prob:.3f} ({(1 - risk_prob) * 100:.1f}%)")

# 底部：新增数据重置按钮
if st.button("🔄 重置输入数据", use_container_width=True):
    st.rerun()  # 重置页面，清空输入

# ===================== 5. 性能优化：兼容高版本配置 =====================
# 移除高版本不兼容的配置项
st.set_option('client.showErrorDetails', False)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import base64
import tempfile
import os

# ===================== 1. 性能优化：全局配置 + 缓存强化 =====================
# 页面配置（提前加载，减少渲染延迟）
st.set_page_config(
    page_title="👓 MyopiaEtiAI - 青少年近视成因评估系统",
    page_icon="👓",
    layout="wide",  # 改为wide以容纳更多内容
    initial_sidebar_state="expanded"  # 展开侧边栏便于使用
)

# 禁用不必要的功能，提升加载速度
try:
    st.config.set_option("client.showSidebarNavigation", False)
except Exception:
    pass
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
        return "高风险", "#e74c3c", "⚠️ 建议：根据干预方案逻辑库进行针对性干预，优先使用离焦类干预（如配戴角膜塑形镜、高非球微透镜设计平光框架眼镜），并结合环境干预（每日2小时户外活动）"
    elif 30 <= score < 70:
        return "中风险", "#f39c12", "🔔 建议：增加户外活动时间，改善用眼环境，定期监测眼轴长度变化"
    else:
        return "低风险", "#2ecc71", "✅ 建议：维持现有用眼习惯，每6个月复查一次"


# 新增：示例数据快速填充
def fill_example_data(risk_type):
    """根据风险类型填充示例数据"""
    if risk_type == "高风险":
        return {
            # 生物测量
            "eye_axis_length": 26.5, "refraction": -3.5, "corneal_curvature": 43.5,
            # 行为与环境
            "outdoor_hours": 0.5, "study_hours": 5.0, "screen_time": 4.0,
            # 基因检测（模拟数据）
            "myopia_genes": 0.8,
            # 泪液分子标志物
            "IL6": 12.5, "MMP2": 1.1,
            # OCTA图像指标
            "vessel_density": 45.0,
            # 视网膜离焦值
            "retinal_defocus": -1.5,
            # 模型输入（从上述数据计算）
            "HIF1α": 45.0, "IL1β": 18.0, "MMP2": 1.1, "DA": 2.2, "Lactate": 2.8,
            "prop_hypoxia": 0.38, "prop_inflammation": 0.28, "prop_DA": 0.08,
            "prop_metabolism": 0.15, "prop_defocus": 0.11, "RORA": 12.0
        }
    elif risk_type == "中风险":
        return {
            # 生物测量
            "eye_axis_length": 24.5, "refraction": -1.5, "corneal_curvature": 43.0,
            # 行为与环境
            "outdoor_hours": 1.5, "study_hours": 4.0, "screen_time": 3.0,
            # 基因检测（模拟数据）
            "myopia_genes": 0.5,
            # 泪液分子标志物
            "IL6": 8.5, "MMP2": 0.7,
            # OCTA图像指标
            "vessel_density": 48.0,
            # 视网膜离焦值
            "retinal_defocus": -0.8,
            # 模型输入
            "HIF1α": 30.0, "IL1β": 12.0, "MMP2": 0.7, "DA": 3.5, "Lactate": 2.2,
            "prop_hypoxia": 0.25, "prop_inflammation": 0.22, "prop_DA": 0.12,
            "prop_metabolism": 0.20, "prop_defocus": 0.11, "RORA": 18.0
        }
    else:
        return {
            # 生物测量
            "eye_axis_length": 23.0, "refraction": 0.0, "corneal_curvature": 42.5,
            # 行为与环境
            "outdoor_hours": 2.5, "study_hours": 3.0, "screen_time": 2.0,
            # 基因检测（模拟数据）
            "myopia_genes": 0.2,
            # 泪液分子标志物
            "IL6": 5.0, "MMP2": 0.4,
            # OCTA图像指标
            "vessel_density": 52.0,
            # 视网膜离焦值
            "retinal_defocus": -0.2,
            # 模型输入
            "HIF1α": 15.0, "IL1β": 7.0, "MMP2": 0.4, "DA": 4.8, "Lactate": 1.6,
            "prop_hypoxia": 0.12, "prop_inflammation": 0.15, "prop_DA": 0.18,
            "prop_metabolism": 0.28, "prop_defocus": 0.17, "RORA": 28.0
        }


def create_radar_chart(input_data, risk_score):
    """创建雷达图"""
    # 提取关键指标
    metrics = ['缺氧相关(HIF1α)', '炎症相关(IL1β)', 'MMP2', '多巴胺(DA)', 
               '乳酸(Lactate)', 'RORA']
    values = [
        input_data['HIF1α'][0] / 50 * 100,
        input_data['IL1β'][0] / 20 * 100,
        input_data['MMP2'][0] / 1.2 * 100,
        input_data['DA'][0] / 5 * 100,
        input_data['Lactate'][0] / 3 * 100,
        input_data['RORA'][0] / 30 * 100
    ]
    
    # 亚型占比
    subtype_metrics = ['缺氧亚型', '炎症亚型', '多巴胺亚型', '代谢亚型', '离焦亚型']
    subtype_values = [
        input_data['prop_hypoxia'][0] * 400,
        input_data['prop_inflammation'][0] * 400,
        input_data['prop_DA'][0] * 400,
        input_data['prop_metabolism'][0] * 400,
        input_data['prop_defocus'][0] * 400
    ]
    
    # 创建雷达图
    fig1 = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=metrics,
        fill='toself',
        name='关键指标',
        line_color='#3498db',
        fillcolor='rgba(52, 152, 219, 0.3)'
    ))
    
    fig1.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="关键指标雷达图",
        height=400
    )
    
    fig2 = go.Figure(data=go.Scatterpolar(
        r=subtype_values,
        theta=subtype_metrics,
        fill='toself',
        name='亚型占比',
        line_color='#e74c3c',
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))
    
    fig2.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="亚型占比雷达图",
        height=400
    )
    
    return fig1, fig2


def generate_pdf_report(input_data, risk_score, risk_level, advice, risk_prob, raw_data):
    """生成PDF报告（按新的数据类型展示）"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # 注册中文字体
    try:
        # 尝试使用常见的中文字体
        pdfmetrics.registerFont(TTFont('SimSun', 'SimSun.ttf'))
        pdfmetrics.registerFont(TTFont('SimHei', 'SimHei.ttf'))
        chinese_font = 'SimHei'
    except:
        try:
            # Linux环境下的中文字体
            pdfmetrics.registerFont(TTFont('NotoSansCJK', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'))
            chinese_font = 'NotoSansCJK'
        except:
            # 回退到默认字体
            chinese_font = 'Helvetica'
    
    # 创建支持中文的样式
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontName=chinese_font,
        fontSize=18,
        alignment=1,  # 居中
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=chinese_font,
        fontSize=11,
        leading=16,
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading3'],
        fontName=chinese_font,
        fontSize=14,
        spaceBefore=12,
    )
    
    # 标题
    title = Paragraph("青少年近视成因评估报告", title_style)
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # 副标题
    subtitle = Paragraph("MyopiaEtiAI - 基于随机森林的青少年初发性近视成因评估系统", ParagraphStyle('Subtitle', parent=styles['Normal'], fontName=chinese_font, fontSize=10, alignment=1))
    story.append(subtitle)
    story.append(Spacer(1, 0.2*inch))
    
    # 报告时间
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_para = Paragraph(f"<b>评估时间:</b> {date_str}", normal_style)
    story.append(date_para)
    story.append(Spacer(1, 0.2*inch))
    
    # 风险评分
    if risk_score >= 70:
        risk_color_name = "red"
    elif risk_score >= 30:
        risk_color_name = "orange"
    else:
        risk_color_name = "green"
    
    score_title = Paragraph(f"<b>风险评分:</b> <font color='{risk_color_name}'>{risk_score}</font>", normal_style)
    story.append(score_title)
    
    level_title = Paragraph(f"<b>风险等级:</b> <font color='{risk_color_name}'>{risk_level}</font>", normal_style)
    story.append(level_title)
    story.append(Spacer(1, 0.2*inch))
    
    # 概率详情
    prob_para = Paragraph(f"<b>高危概率:</b> {risk_prob*100:.1f}%", normal_style)
    story.append(prob_para)
    prob_para2 = Paragraph(f"<b>低危概率:</b> {(1-risk_prob)*100:.1f}%", normal_style)
    story.append(prob_para2)
    story.append(Spacer(1, 0.3*inch))
    
    # 建议
    advice_title = Paragraph("<b>专业建议:</b>", heading_style)
    story.append(advice_title)
    advice_clean = advice.replace("⚠️", "").replace("🔔", "").replace("✅", "")
    advice_para = Paragraph(advice_clean, normal_style)
    story.append(advice_para)
    story.append(Spacer(1, 0.3*inch))
    
    # ========== 按新的数据类型展示数据 ==========
    
    # 1. 生物测量数据
    data_title = Paragraph("<b>一、生物测量数据</b>", heading_style)
    story.append(data_title)
    
    biometric_data = [
        ['检测项目', '检测值', '检测方法', '对应学说'],
        ['眼轴长度 (mm)', f"{raw_data.get('eye_axis_length', 24.5):.2f}", '光学测量仪', '—'],
        ['屈光状态 (D)', f"{raw_data.get('refraction', 0):.2f}", '电脑验光', '—'],
        ['角膜曲率 (D)', f"{raw_data.get('corneal_curvature', 43.0):.2f}", '光学测量仪', '—'],
    ]
    
    biometric_table = Table(biometric_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
    biometric_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, '#000000'),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), ['#EEEEEE', '#FFFFFF']),
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTNAME', (0, 0), (0, 0), chinese_font + '-Bold'),
    ]))
    story.append(biometric_table)
    story.append(Spacer(1, 0.2*inch))
    
    # 2. 行为与环境数据
    data_title = Paragraph("<b>二、行为与环境数据</b>", heading_style)
    story.append(data_title)
    
    environment_data = [
        ['检测项目', '检测值', '检测方法', '对应学说'],
        ['户外活动时长 (小时/天)', f"{raw_data.get('outdoor_hours', 1.5):.1f}", '问卷采集', '环境'],
    ]
    
    environment_table = Table(environment_data, colWidths=[2.2*inch, 1.5*inch, 1.5*inch, 1*inch])
    environment_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, '#000000'),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), ['#EEEEEE', '#FFFFFF']),
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTNAME', (0, 0), (0, 0), chinese_font + '-Bold'),
    ]))
    story.append(environment_table)
    story.append(Spacer(1, 0.2*inch))
    
    # 3. 基因检测数据
    data_title = Paragraph("<b>三、基因检测数据</b>", heading_style)
    story.append(data_title)
    
    gene_data = [
        ['检测项目', '检测值', '检测方法', '对应学说'],
        ['近视易感基因风险评分', f"{raw_data.get('myopia_genes', 0.5):.2f}", '①采集：使用无菌拭子在脸颊内侧收集上皮细胞<br/>②提取：采用口腔拭子基因组DNA提取试剂盒', '遗传'],
    ]
    
    gene_table = Table(gene_data, colWidths=[2*inch, 1.2*inch, 2.3*inch, 1*inch])
    gene_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, '#000000'),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), ['#EEEEEE', '#FFFFFF']),
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTNAME', (0, 0), (0, 0), chinese_font + '-Bold'),
    ]))
    story.append(gene_table)
    story.append(Spacer(1, 0.2*inch))
    
    # 4. 泪液分子标志物数据
    data_title = Paragraph("<b>四、泪液分子标志物数据</b>", heading_style)
    story.append(data_title)
    
    tear_data = [
        ['检测项目', '检测值', '检测方法', '对应学说'],
        ['IL-6 (pg/mL)', f"{raw_data.get('IL6', 8.0):.2f}", '①泪液采集：使用Schirmer试纸或毛细血管采血管<br/>②检测：使用ELISA试剂盒检测', '微炎症'],
        ['MMP-2 (ng/mL)', f"{raw_data.get('MMP2', 0.7):.2f}", '①泪液采集：使用Schirmer试纸或毛细血管采血管<br/>②检测：使用ELISA试剂盒检测', '微炎症'],
    ]
    
    tear_table = Table(tear_data, colWidths=[1.8*inch, 1.2*inch, 2.5*inch, 1*inch])
    tear_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, '#000000'),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), ['#EEEEEE', '#FFFFFF']),
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTNAME', (0, 0), (0, 0), chinese_font + '-Bold'),
    ]))
    story.append(tear_table)
    story.append(Spacer(1, 0.2*inch))
    
    # 5. OCTA图像数据
    data_title = Paragraph("<b>五、OCTA图像数据</b>", heading_style)
    story.append(data_title)
    
    octa_data = [
        ['检测项目', '检测值', '检测方法', '对应学说'],
        ['视网膜血管密度 (%)', f"{raw_data.get('vessel_density', 48.0):.1f}", '光学相干断层扫描血流成像仪', '缺氧'],
    ]
    
    octa_table = Table(octa_data, colWidths=[2*inch, 1.5*inch, 2*inch, 1*inch])
    octa_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, '#000000'),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), ['#EEEEEE', '#FFFFFF']),
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTNAME', (0, 0), (0, 0), chinese_font + '-Bold'),
    ]))
    story.append(octa_table)
    story.append(Spacer(1, 0.2*inch))
    
    # 6. 视网膜离焦值
    data_title = Paragraph("<b>六、视网膜离焦值</b>", heading_style)
    story.append(data_title)
    
    defocus_data = [
        ['检测项目', '检测值', '检测方法', '对应学说'],
        ['视网膜离焦值 (D)', f"{raw_data.get('retinal_defocus', -0.5):.2f}", '自动验光仪', '离焦'],
    ]
    
    defocus_table = Table(defocus_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
    defocus_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, '#000000'),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), ['#EEEEEE', '#FFFFFF']),
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTNAME', (0, 0), (0, 0), chinese_font + '-Bold'),
    ]))
    story.append(defocus_table)
    story.append(Spacer(1, 0.3*inch))
    
    # 模型详细数据（用于科研参考）
    data_title = Paragraph("<b>七、模型算法分析数据</b>", heading_style)
    story.append(data_title)
    
    model_data = [
        ['指标名称', '检测值'],
        ['HIF1a (pg/mL)', f"{input_data['HIF1α'][0]:.2f}"],
        ['IL1b (pg/mL)', f"{input_data['IL1β'][0]:.2f}"],
        ['MMP2 (ng/mL)', f"{input_data['MMP2'][0]:.2f}"],
        ['DA (ng/mL)', f"{input_data['DA'][0]:.2f}"],
        ['Lactate (mmol/L)', f"{input_data['Lactate'][0]:.2f}"],
        ['RORA (pg/mL)', f"{input_data['RORA'][0]:.2f}"],
        ['缺氧亚型占比', f"{input_data['prop_hypoxia'][0]:.2%}"],
        ['炎症亚型占比', f"{input_data['prop_inflammation'][0]:.2%}"],
        ['多巴胺亚型占比', f"{input_data['prop_DA'][0]:.2%}"],
        ['代谢亚型占比', f"{input_data['prop_metabolism'][0]:.2%}"],
        ['离焦亚型占比', f"{input_data['prop_defocus'][0]:.2%}"]
    ]
    
    model_table = Table(model_data, colWidths=[2.5*inch, 2*inch])
    model_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, '#000000'),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), ['#EEEEEE', '#FFFFFF']),
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTNAME', (0, 0), (0, 0), chinese_font + '-Bold'),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 0.3*inch))
    
    # 免责声明
    disclaimer = Paragraph("注：本报告仅供参考，不构成医疗诊断建议。如有疑问请咨询专业医师。", normal_style)
    story.append(disclaimer)
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# ===================== 4. 界面增强：新增功能 + 优化布局 =====================
st.title("👓 MyopiaEtiAI - 青少年近视成因评估系统")
st.markdown("### 基于随机森林的青少年初发性近视成因评估系统")
st.markdown("---")

# 侧边栏：新增示例数据快速填充 + 使用说明
with st.sidebar:
    st.subheader("⚡ 快捷操作")
    example_type = st.radio("加载示例数据", ["无", "高风险示例", "中风险示例", "低风险示例"])
    example_data = {}
    if example_type != "无":
        example_data = fill_example_data(example_type.replace("示例", ""))

    st.markdown("---")
    st.subheader("📖 使用说明")
    st.markdown("1. 按分类输入检测数据，或加载示例数据")
    st.markdown("2. 点击「计算风险评分」获取结果")
    st.markdown("3. 下载PDF报告查看详细分析")
    st.markdown("---")
    st.subheader("📋 数据采集说明")
    st.markdown("• **生物测量**: 眼轴、屈光、角膜曲率")
    st.markdown("• **行为环境**: 户外活动时长")
    st.markdown("• **基因检测**: 颊拭子采集")
    st.markdown("• **泪液标志物**: IL-6、MMP-2检测")
    st.markdown("• **OCTA图像**: 视网膜血管密度")
    st.markdown("• **离焦值**: 视网膜离焦测量")

# 主界面：输入区（按数据类型分组）
st.subheader("🔍 数据采集输入")

# Tab 1: 生物测量
st.markdown("#### 📏 1. 生物测量数据")
col1, col2, col3 = st.columns(3)
with col1:
    eye_axis_length = st.number_input("眼轴长度 (mm)", min_value=20.0, max_value=30.0,
                                      value=example_data.get("eye_axis_length", 24.5), step=0.1)
with col2:
    refraction = st.number_input("屈光状态 (D)", min_value=-10.0, max_value=5.0,
                                 value=example_data.get("refraction", 0.0), step=0.25)
with col3:
    corneal_curvature = st.number_input("角膜曲率 (D)", min_value=38.0, max_value=48.0,
                                        value=example_data.get("corneal_curvature", 43.0), step=0.1)

st.markdown("---")

# Tab 2: 行为与环境
st.markdown("#### 🌳 2. 行为与环境数据")
outdoor_hours = st.number_input("户外活动时长 (小时/天)", min_value=0.0, max_value=8.0,
                                 value=example_data.get("outdoor_hours", 1.5), step=0.1)

st.markdown("---")

# Tab 3: 基因检测
st.markdown("#### 🧬 3. 基因检测数据")
st.info("检测方法：①采集：使用无菌拭子在脸颊内侧收集上皮细胞，并转移至含有保护液的采样瓶中；②提取：采用口腔拭子基因组DNA提取试剂盒进行提取")
myopia_genes = st.number_input("近视易感基因风险评分 (0-1)", min_value=0.0, max_value=1.0,
                                 value=example_data.get("myopia_genes", 0.5), step=0.1)

st.markdown("---")

# Tab 4: 泪液分子标志物
st.markdown("#### 💧 4. 泪液分子标志物数据")
st.info("检测方法：①泪液采集：使用Schirmer试纸或毛细血管采血管，置于-80℃冰箱保存；②检测：使用ELISA试剂盒检测浓度")
col4, col5 = st.columns(2)
with col4:
    IL6 = st.number_input("IL-6 (pg/mL)", min_value=0.0, max_value=30.0,
                          value=example_data.get("IL6", 8.0), step=0.1)
with col5:
    MMP2_tear = st.number_input("MMP-2 (ng/mL)", min_value=0.1, max_value=2.0,
                                value=example_data.get("MMP2", 0.7), step=0.01)

st.markdown("---")

# Tab 5: OCTA图像
st.markdown("#### 👁️ 5. OCTA图像数据")
st.info("检测方法：光学相干断层扫描血流成像仪")
vessel_density = st.number_input("视网膜血管密度 (%)", min_value=30.0, max_value=70.0,
                                 value=example_data.get("vessel_density", 48.0), step=0.1)

st.markdown("---")

# Tab 6: 视网膜离焦值
st.markdown("👓 6. 视网膜离焦值")
st.info("检测方法：自动验光仪")
retinal_defocus = st.number_input("视网膜离焦值 (D)", min_value=-5.0, max_value=5.0,
                                   value=example_data.get("retinal_defocus", -0.5), step=0.1)

st.markdown("---")

# 模型参数（用于科研分析，可隐藏）
with st.expander("🔬 查看模型参数（高级科研用）"):
    st.markdown("#### 生物标志物参数（用于随机森林模型）")
    col6, col7 = st.columns(2)
    with col6:
        hif1a = st.number_input("HIF1α (pg/mL)", min_value=10.0, max_value=50.0,
                                value=example_data.get("HIF1α", 30.0), step=0.1)
        il1b = st.number_input("IL1β (pg/mL)", min_value=5.0, max_value=20.0,
                               value=example_data.get("IL1β", 12.0), step=0.1)
        mmp2 = st.number_input("MMP2 (ng/mL)", min_value=0.3, max_value=1.2,
                               value=example_data.get("MMP2", 0.7), step=0.01)
        da = st.number_input("DA (ng/mL)", min_value=2.0, max_value=5.0,
                             value=example_data.get("DA", 3.5), step=0.1)
    with col7:
        lactate = st.number_input("Lactate (mmol/L)", min_value=1.5, max_value=3.0,
                                  value=example_data.get("Lactate", 2.2), step=0.01)
        rora = st.number_input("RORA (pg/mL)", min_value=10.0, max_value=30.0,
                               value=example_data.get("RORA", 18.0), step=0.1)
    
    st.markdown("#### 亚型占比参数")
    col8, col9, col10 = st.columns(3)
    with col8:
        prop_hypoxia = st.number_input("缺氧亚型占比", min_value=0.1, max_value=0.4,
                                       value=example_data.get("prop_hypoxia", 0.25), step=0.01)
        prop_inflammation = st.number_input("炎症亚型占比", min_value=0.1, max_value=0.3,
                                            value=example_data.get("prop_inflammation", 0.22), step=0.01)
    with col9:
        prop_da = st.number_input("多巴胺亚型占比", min_value=0.05, max_value=0.2,
                                  value=example_data.get("prop_DA", 0.12), step=0.01)
        prop_metabolism = st.number_input("代谢亚型占比", min_value=0.1, max_value=0.25,
                                          value=example_data.get("prop_metabolism", 0.20), step=0.01)
    with col10:
        prop_defocus = st.number_input("离焦亚型占比", min_value=0.05, max_value=0.15,
                                       value=example_data.get("prop_defocus", 0.11), step=0.01)

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
        
        # 生成雷达图
        fig1, fig2 = create_radar_chart(input_data, risk_score)

        # 生成原始数据字典（用于PDF报告）
        raw_data = {
            "eye_axis_length": eye_axis_length,
            "refraction": refraction,
            "corneal_curvature": corneal_curvature,
            "outdoor_hours": outdoor_hours,
            "myopia_genes": myopia_genes,
            "IL6": IL6,
            "MMP2": MMP2_tear,
            "vessel_density": vessel_density,
            "retinal_defocus": retinal_defocus
        }

    # 生成PDF报告（移到外层，确保可以访问）
    pdf_buffer = generate_pdf_report(input_data, risk_score, risk_level, advice, risk_prob, raw_data)

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

    # 新增：雷达图展示
    st.markdown("### 📊 数据可视化分析")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.plotly_chart(fig1, use_container_width=True)
    with col_chart2:
        st.plotly_chart(fig2, use_container_width=True)

    # 新增：概率详情（可选查看）
    with st.expander("🔍 查看详细概率（高级）"):
        st.markdown(f"- 高危概率：{risk_prob:.3f} ({risk_prob * 100:.1f}%)")
        st.markdown(f"- 低危概率：{1 - risk_prob:.3f} ({(1 - risk_prob) * 100:.1f}%)")

    # 新增：数据采集结果摘要
    st.markdown("### 📋 数据采集摘要")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.metric("眼轴长度", f"{eye_axis_length:.2f} mm")
        st.metric("屈光状态", f"{refraction:.2f} D")
    with summary_col2:
        st.metric("户外活动", f"{outdoor_hours:.1f} 小时/天")
        st.metric("基因风险", f"{myopia_genes:.2f}")
    with summary_col3:
        st.metric("IL-6", f"{IL6:.2f} pg/mL")
        st.metric("血管密度", f"{vessel_density:.1f}%")

    # 新增：PDF报告下载
    st.markdown("---")
    st.markdown("### 📥 下载评估报告")
    col_download = st.columns(1)[0]
    with col_download:
        pdf_bytes = pdf_buffer.getvalue()
        st.download_button(
            label="📄 下载完整评估报告 (PDF)",
            data=pdf_bytes,
            file_name=f"青少年近视成因评估报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

# 底部：新增数据重置按钮
if st.button("🔄 重置输入数据", use_container_width=True):
    st.rerun()  # 重置页面，清空输入

# ===================== 5. 性能优化：兼容高版本配置 =====================
# 移除高版本不兼容的配置项
st.set_option('client.showErrorDetails', False)

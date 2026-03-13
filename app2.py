import streamlit as st
import pandas as pd  # 读数据、清洗、整理表格
import io
import numpy as np  # 对数组做快速数学计算
import tensorflow as tf
import os
import tempfile
import shutil
import time
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import warnings
from sklearn.mixture import GaussianMixture
from openai import OpenAI
import requests
# 页面配置（完全保留）
st.set_page_config(page_title="🧠 ANN训练器", layout="wide")
warnings.filterwarnings('ignore')

# ========== 初始化session state（统一整理，不删任何变量） ==========
def init_session_state():
    # 保留你所有的变量，只是统一初始化，避免重复定义
    state_vars = {
        'df': None,
        'raw_df': None,
        'current_file': None,
        'step': 1,
        'model_config': None,
        'model': None,
        'scaler': None,
        'history': None,
        'label_encoders': {},
        'encoding_log': [],
        'scores': {},
        'total_score': 0,
        'max_total': 100,
        'warnings': [],
        'detection_results': {},
        'original_df': None,
        'processed_df': None,
        'temp_df': None,
        'uploaded_model': None,  # 新增：上传的模型
        'uploaded_scaler': None,  # 新增：上传的标准化器
        'uploaded_config': None,  # 新增：上传的配置
        'ai_enabled': False,
        'ai_advice': None,
        'ai_messages': None,
        'ai_history': [],
        'ai_advice_generated': False,
        'user_choices': {
            'missing': None,
            'outlier': None,
            'balance': None,
            'features': None,
            # 新增：标记步骤是否已确认，修复无限刷新
            'missing_confirmed': False,
            'outlier_confirmed': False,
            'balance_confirmed': False,
            'features_confirmed': False
        }
    }
    for key, value in state_vars.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ========== 验证环境变量 ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if DEEPSEEK_API_KEY:
    st.session_state.ai_enabled = True
    st.success("你好，我是王嘉屿！")
    # 初始化AI对话历史
    if st.session_state.ai_messages is None:
        st.session_state.ai_messages = [
            {"role": "system", "content": "你是一个机器学习调参专家，请基于对话历史给出连贯的建议。记住之前给用户的建议，让每次建议都有连续性。"}
        ]
    if st.session_state.ai_history is None:
        st.session_state.ai_history = []
else:
    st.session_state.ai_enabled = False
    st.session_state.ai_advice = None

# ========== 新增：GMM聚类函数（用于回归问题的样本平衡性处理） ==========
def gmm_clustering_for_regression(y, max_components=10):
    """
    使用一维高斯混合模型对回归目标值进行聚类
    通过BIC自动选择最优成分数
    返回：聚类标签和最优模型
    """
    y_reshaped = y.reshape(-1, 1)
    # 如果样本太少，直接返回单类
    if len(y) < 10:
        return np.zeros(len(y), dtype=int), None
    best_bic = np.inf
    best_gmm = None
    best_n = 1
    # 尝试不同数量的成分
    n_components_range = range(1, min(max_components, len(y) // 5) + 1)
    for n in n_components_range:
        try:
            gmm = GaussianMixture(n_components=n, random_state=42, max_iter=100)
            gmm.fit(y_reshaped)
            bic = gmm.bic(y_reshaped)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_n = n
        except:
            continue
    if best_gmm is not None:
        labels = best_gmm.predict(y_reshaped)
        return labels, best_gmm
    else:
        return np.zeros(len(y), dtype=int), None

# ========== 提取重复函数（不修改功能，只精简重复代码） ==========
def data_quality_score(df, label_col, label_encoders):
    """完全保留你原版的评分逻辑，只是移到函数里避免重复"""
    scores = {}
    warnings_list = []
    detection_results = {}
    # 1. 缺失值检测（30分）
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0].index.tolist()
    rows_with_na = df[df.isnull().any(axis=1)].index.tolist()
    missing_ratio = missing_counts.sum() / (df.shape[0] * df.shape[1])
    detection_results['missing'] = {
        'missing_counts': missing_counts.to_dict(),
        'missing_cols': missing_cols,
        'rows_with_na': rows_with_na,
        'missing_ratio': missing_ratio
    }
    if missing_ratio == 0:
        missing_score = 30
        missing_status = "✅ 无缺失值"
    elif missing_ratio < 0.01:
        missing_score = 25
        missing_status = "⚠️ 轻微缺失"
    elif missing_ratio < 0.05:
        missing_score = 20
        missing_status = "⚠️ 中度缺失"
    elif missing_ratio < 0.1:
        missing_score = 15
        missing_status = "⚠️ 较多缺失"
    else:
        missing_score = 10
        missing_status = "❌ 严重缺失"
        warnings_list.append("缺失值比例过高，建议处理")
    scores['缺失值'] = {
        'score': missing_score,
        'max_score': 30,
        'status': missing_status,
        'detail': f"缺失比例: {missing_ratio:.2%}"
    }
    # 2. 异常值检测（25分）
    outlier_info = []
    outlier_counts = []
    for col in df.columns[:-1]:
        data = df[col].dropna()
        if len(data) > 3:
            # ===== 先判断是否是分类变量 =====
            if col in label_encoders:
                # 分类变量：直接用IQR
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((data < lower_bound) | (data > upper_bound)).sum()
                method = "IQR (分类变量)"
                is_normal = False  # 分类变量不判断正态性
            else:
                # 连续变量：判断正态性（考虑样本量）
                n = len(data)
                # 根据样本量选择不同的显著性水平
                if n > 1000:
                    alpha = 0.01  # 大样本更严格
                elif n > 500:
                    alpha = 0.05  # 中等样本
                else:
                    alpha = 0.1  # 小样本宽松

                # 正态性检验
                shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, n)))
                is_normal = shapiro_p > alpha

                if is_normal:
                    # 正态分布用Z-score
                    mean_val = data.mean()
                    std_val = data.std()
                    z_scores = np.abs((data - mean_val) / std_val)
                    outliers = (z_scores > 2.5).sum()
                    method = "Z-score"
                else:
                    # 非正态用IQR
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
                    method = "IQR"

            if outliers > 0:
                outlier_info.append({
                    '列名': col,
                    '异常值数量': outliers,
                    '比例': f"{outliers / len(data) * 100:.1f}%",
                    '方法': method
                })
            outlier_counts.append(outliers)
    detection_results['outlier'] = {
        'outlier_info': outlier_info,
        'outlier_cols': [item['列名'] for item in outlier_info]
    }
    if outlier_counts:
        total_outliers = sum(outlier_counts)
        total_values = len(df) * len(df.columns[:-1])
        outlier_ratio = total_outliers / total_values
    else:
        outlier_ratio = 0
    if outlier_ratio == 0:
        outlier_score = 25
        outlier_status = "✅ 无异常值"
    elif outlier_ratio < 0.01:
        outlier_score = 22
        outlier_status = "⚠️ 轻微异常"
    elif outlier_ratio < 0.03:
        outlier_score = 18
        outlier_status = "⚠️ 中度异常"
    elif outlier_ratio < 0.05:
        outlier_score = 15
        outlier_status = "⚠️ 较多异常"
    else:
        outlier_score = 10
        outlier_status = "❌ 严重异常"
        warnings_list.append("异常值比例较高，建议处理")
    scores['异常值'] = {
        'score': outlier_score,
        'max_score': 25,
        'status': outlier_status,
        'detail': f"异常值比例: {outlier_ratio:.2%}" if outlier_ratio > 0 else "无异常值"
    }
    # 3. 样本平衡性检测（25分）- 修改：支持分类和回归
    is_classification = label_col in label_encoders
    if is_classification:
        # 分类问题的平衡性检测
        value_counts = df[label_col].value_counts()
        detection_results['balance'] = {
            'is_classification': True,
            'value_counts': value_counts.to_dict(),
            'class_counts': len(value_counts)
        }
        if len(value_counts) > 1:
            max_ratio = value_counts.max() / len(df)
            min_ratio = value_counts.min() / len(df)
            imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
            if imbalance_ratio < 1.5:
                balance_score = 25
                balance_status = "✅ 样本平衡"
            elif imbalance_ratio < 3:
                balance_score = 20
                balance_status = "⚠️ 轻度不平衡"
            elif imbalance_ratio < 5:
                balance_score = 15
                balance_status = "⚠️ 中度不平衡"
            else:
                balance_score = 10
                balance_status = "❌ 严重不平衡"
                warnings_list.append("样本类别严重不平衡，可能影响模型训练")
            balance_detail = f"最大类占比: {max_ratio:.2%}, 最小类占比: {min_ratio:.2%}"
        else:
            balance_score = 25
            balance_status = "✅ 单类别"
            balance_detail = "只有一种类别"
    else:
        # 回归问题的平衡性检测 - 使用GMM聚类
        y = df[label_col].values
        labels, gmm = gmm_clustering_for_regression(y, max_components=8)
        # 统计聚类后的类别分布
        cluster_counts = pd.Series(labels).value_counts()
        n_clusters = len(cluster_counts)
        detection_results['balance'] = {
            'is_classification': False,
            'gmm_clusters': n_clusters,
            'cluster_counts': cluster_counts.to_dict(),
            'gmm_model': gmm  # 保存GMM模型供后续处理使用
        }
        if n_clusters <= 1:
            balance_score = 25
            balance_status = "✅ 单峰分布"
            balance_detail = "目标值呈现单峰分布"
        else:
            max_ratio = cluster_counts.max() / len(y)
            min_ratio = cluster_counts.min() / len(y)
            imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
            if imbalance_ratio < 1.5:
                balance_score = 25
                balance_status = "✅ 聚类平衡"
            elif imbalance_ratio < 3:
                balance_score = 20
                balance_status = "⚠️ 聚类轻度不平衡"
            elif imbalance_ratio < 5:
                balance_score = 15
                balance_status = "⚠️ 聚类中度不平衡"
            else:
                balance_score = 10
                balance_status = "❌ 聚类严重不平衡"
                warnings_list.append("基于GMM聚类的样本分布严重不平衡，可能影响模型训练")
            balance_detail = f"GMM聚类数: {n_clusters}, 最大类占比: {max_ratio:.2%}, 最小类占比: {min_ratio:.2%}"
    scores['样本平衡'] = {
        'score': balance_score,
        'max_score': 25,
        'status': balance_status,
        'detail': balance_detail
    }
    # 4. 特征重要性评分（20分）
    try:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        if len(df) < 30:
            feature_score = 5
            feature_status = "⚠️ 样本太少"
            feature_detail = f"当前{len(df)}行，无法准确评估"
            feature_importance = None
        elif X.isnull().any().any():
            feature_score = 6
            feature_status = "⚠️ 存在缺失值"
            feature_detail = "处理缺失值后可重新评估"
            feature_importance = None
        else:
            if label_col in label_encoders:
                rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                y_encoded = y.values
            else:
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                y_encoded = y.values
            rf_model.fit(X, y_encoded)
            importance = rf_model.feature_importances_
            importance_normalized = importance / importance.sum() * 100
            low_importance_ratio = (importance_normalized < 1).sum() / len(importance_normalized)
            feature_importance = {
                'features': X.columns.tolist(),
                'importance': importance_normalized.tolist()
            }
            if low_importance_ratio == 0:
                feature_score = 20
                feature_status = "✅ 所有特征都有贡献"
            elif low_importance_ratio < 0.2:
                feature_score = 18
                feature_status = "⚠️ 少量噪声特征"
            elif low_importance_ratio < 0.3:
                feature_score = 15
                feature_status = "⚠️ 部分特征为噪声"
            elif low_importance_ratio < 0.4:
                feature_score = 12
                feature_status = "⚠️ 较多噪声特征"
                warnings_list.append("存在较多噪声特征，建议筛选")
            else:
                feature_score = 8
                feature_status = "❌ 大量噪声特征"
                warnings_list.append("噪声特征过多，建议筛选")
            feature_detail = f"噪声特征比例: {low_importance_ratio:.1%}"
        detection_results['feature'] = {
            'importance': feature_importance,
            'feature_cols': X.columns.tolist()
        }
    except Exception as e:
        feature_score = 4
        feature_status = "❌ 无法评估"
        feature_detail = "请先处理数据后再试"
        detection_results['feature'] = {'importance': None, 'feature_cols': df.columns[:-1].tolist()}
    scores['特征重要性'] = {
        'score': feature_score,
        'max_score': 20,
        'status': feature_status,
        'detail': feature_detail
    }
    total_score = sum([s['score'] for s in scores.values()])
    max_total = sum([s['max_score'] for s in scores.values()])
    return scores, total_score, max_total, warnings_list, detection_results

# ========== 新增：处理回归样本不平衡的函数 ==========
def balance_regression_by_gmm(df, label_col, method='smote'):
    """
    使用GMM聚类结果为回归问题生成合成样本
    method: 'smote', 'adasyn', 'undersample', 'none'
    """
    y = df[label_col].values
    X = df.iloc[:, :-1].copy()
    # 使用GMM聚类
    labels, gmm = gmm_clustering_for_regression(y, max_components=8)
    if gmm is None or len(np.unique(labels)) <= 1:
        return df, labels  # 无法聚类或只有一类，返回原数据
    # 基于聚类标签进行平衡处理
    cluster_counts = pd.Series(labels).value_counts()
    if method == 'none' or len(cluster_counts) <= 1:
        return df, labels
    # 准备数据
    X_balanced = X.copy()
    y_balanced = pd.Series(y.copy(), name=label_col)
    labels_balanced = labels.copy()
    if method == 'smote' or method == 'adasyn':
        # 对于回归问题，我们不能直接使用SMOTE，而是对每个聚类内部进行过采样
        max_cluster_size = cluster_counts.max()
        for cluster_id in cluster_counts.index:
            current_size = cluster_counts[cluster_id]
            if current_size < max_cluster_size:
                # 需要过采样的聚类
                cluster_mask = labels == cluster_id
                X_cluster = X[cluster_mask]
                y_cluster = y[cluster_mask]
                # 计算需要生成的样本数
                n_synthesize = max_cluster_size - current_size
                # 简单过采样：随机复制并添加噪声
                for _ in range(n_synthesize):
                    # 随机选择一个样本
                    idx = np.random.randint(0, len(X_cluster))
                    X_sample = X_cluster.iloc[idx].copy()
                    y_sample = y_cluster[idx]
                    # 添加小噪声
                    noise_scale = 0.01 * X_sample.std() if X_sample.std() > 0 else 0.01
                    X_noise = np.random.normal(0, noise_scale, size=len(X_sample))
                    X_sample = X_sample + X_noise
                    # 添加到数据集
                    X_balanced = pd.concat([X_balanced, pd.DataFrame([X_sample])], ignore_index=True)
                    y_balanced = pd.concat([y_balanced, pd.Series([y_sample])], ignore_index=True)
                    labels_balanced = np.append(labels_balanced, cluster_id)
        # 重建DataFrame
        balanced_df = X_balanced.copy()
        balanced_df[label_col] = y_balanced.values
    elif method == 'undersample':
        # 欠采样：减少多数类的样本
        min_cluster_size = cluster_counts.min()
        indices_to_keep = []
        for cluster_id in cluster_counts.index:
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) > min_cluster_size:
                # 随机选择
                selected = np.random.choice(cluster_indices, min_cluster_size, replace=False)
                indices_to_keep.extend(selected)
            else:
                indices_to_keep.extend(cluster_indices)
        indices_to_keep = sorted(indices_to_keep)
        balanced_df = df.iloc[indices_to_keep].reset_index(drop=True)
        labels_balanced = labels[indices_to_keep]
    else:
        balanced_df = df.copy()
    return balanced_df, labels_balanced


# ========== AI调用函数（完整版，带历史记忆） ==========
def get_ai_advice(data):
    """调用DeepSeek API获取模型评价和优化建议（带历史记忆）"""

    try:
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )

        # 计算基准准确率
        if data['数据概况']['问题类型'] == '分类':
            n_classes = data['数据概况']['类别数']
            if n_classes != "N/A":
                baseline = 100 / n_classes
                baseline_text = f"随机猜测基准是 {baseline:.1f}%"
            else:
                baseline_text = "请根据问题类型合理评价"
        else:
            baseline_text = "这是回归问题，请用MAE等指标评价"

        # 构建当前对话的prompt
        current_prompt = f"""
【当前模型表现】
数据概况：
- 样本量：{data['数据概况']['样本量']}行
- 特征数：{data['数据概况']['特征数']}个
- 问题类型：{data['数据概况']['问题类型']}
- 类别数：{data['数据概况']['类别数']}
- {baseline_text}

当前配置：
- 隐藏层：{data['当前配置']['隐藏层']}层
- 神经元：{data['当前配置']['神经元']}个/层
- 激活函数：{data['当前配置']['激活函数']}
- 学习率：{data['当前配置']['学习率']}
- 批次大小：{data['当前配置']['批次大小']}
- 训练轮次：{data['当前配置']['训练轮次']}
- 测试集比例：{data['当前配置']['测试集比例']}%

当前效果：
- 准确率：{data['当前效果']['准确率']}

【历史建议回顾】
{st.session_state.ai_history[-1] if st.session_state.ai_history else '这是第一次咨询，还没有历史建议。'}

请基于以上信息和历史对话，给出有针对性的优化建议。
如果是第二次或更多次咨询，请参考之前给的建议，告诉用户这次调整后的效果如何，下一步该怎么优化。

请以JSON格式返回，只返回JSON：
{{
    "评价": "结合基准和历史表现评价当前模型",
    "优化建议": {{
        "隐藏层": 建议值,
        "神经元": 建议值,
        "激活函数": "建议值",
        "学习率": 建议值,
        "批次大小": 建议值,
        "训练轮次": 建议值,
        "测试集比例": 建议值
    }},
    "预期效果": "说明优化后可能达到的效果"
}}
"""

        # 添加用户消息到历史
        st.session_state.ai_messages.append({"role": "user", "content": current_prompt})

        # 调用API（传整个历史）
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=st.session_state.ai_messages,
            temperature=0.3,
            max_tokens=1000
        )

        # 获取AI回复
        advice_text = response.choices[0].message.content

        # 将AI回复添加到历史
        st.session_state.ai_messages.append({"role": "assistant", "content": advice_text})

        # 解析JSON
        import json
        import re

        # 提取JSON部分
        json_match = re.search(r'(\{.*\})', advice_text, re.DOTALL)
        if json_match:
            advice_json = json.loads(json_match.group(1))

            # 保存到历史记录
            history_entry = f"第{len(st.session_state.ai_history) + 1}次：准确率{data['当前效果']['准确率']} → {advice_json['评价']}"
            st.session_state.ai_history.append(history_entry)

            return advice_json
        else:
            # 解析失败时返回默认建议
            return {
                "评价": "模型表现不理想，建议调整参数",
                "优化建议": {
                    "隐藏层": 2,
                    "神经元": 64,
                    "激活函数": "relu",
                    "学习率": 0.001,
                    "批次大小": 32,
                    "训练轮次": 100,
                    "测试集比例": 20
                },
                "预期效果": "优化后准确率预计可提升10-20%"
            }

    except Exception as e:
        print(f"AI调用失败：{str(e)}")
        return {
            "评价": f"AI服务暂时不可用，使用默认建议",
            "优化建议": {
                "隐藏层": 2,
                "神经元": 64,
                "激活函数": "relu",
                "学习率": 0.001,
                "批次大小": 32,
                "训练轮次": 100,
                "测试集比例": 20
            },
            "预期效果": "优化后准确率预计可提升10-20%"
        }
# ========== 侧边栏（修改后逻辑） ==========
with st.sidebar:
    st.title("🧠 ANN训练器")
    st.markdown("---")
    # ===== 新增：警告提示 =====
    st.warning(
        "⚠️ **重要提示**：上传的文件不能提前编码！\n\n请保持原始数据（文本类别请保留为字符串），系统会自动进行编码处理。如果提前编码，分类问题可能会被误判为回归问题。")
    st.markdown("---")
    # 文件上传（保留在侧边栏）
    uploaded_file = st.file_uploader("上传数据文件", type=['csv', 'xlsx', 'xls'], key="data_uploader")
    # ===== 模型上传功能（修改核心逻辑：仅上传模型即可跳转） =====
    st.markdown("---")
    st.markdown("### 🤖 上传训练好的模型")
    st.info("上传模型文件后将直接进入预测步骤（无需同时上传数据文件）")
    uploaded_model_file = st.file_uploader("上传模型文件 (.h5)", type=['h5'], key="model_uploader")

    # 逻辑1：仅上传模型文件 → 跳至步骤4（预测）
    if uploaded_model_file is not None:
        # 加载模型（核心修复：确保临时文件100%删除）
        tmp_file_path = None
        model = None
        try:
            # 创建临时文件（delete=False：手动控制删除）
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(uploaded_model_file.getvalue())
                tmp_file.flush()
                tmp_file_path = tmp_file.name  # 保存文件路径，用于后续删除
                # 加载模型
                model = tf.keras.models.load_model(tmp_file_path)
        except Exception as e:
            st.error(f"❌ 模型加载失败：{str(e)}")
            st.stop()
        finally:
            # 关键：无论模型加载成功/失败，都强制删除临时文件
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    # Windows系统需先关闭文件句柄，再删除
                    os.unlink(tmp_file_path)
                except PermissionError:
                    # 若仍被占用，延迟1秒再试（极端情况）
                    time.sleep(1)
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                except Exception as e:
                    st.warning(f"⚠️ 临时文件清理警告：{str(e)}")

        # 保存模型到session state
        st.session_state.uploaded_model = model
        st.session_state.model = model

        # 初始化默认配置（避免后续报错）
        default_config = {
            'feature': [],  # 后续可通过数据文件补充或手动输入
            'target': 'label',
            'hidden_layers': 2,
            'neurons': 64,
            'activation': 'relu',
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'test_size': 0.2
        }
        st.session_state.model_config = default_config
        st.session_state.is_classification = False  # 默认回归，后续可自动识别
        st.session_state.num_classes = None

        # 直接跳转到步骤4（预测）
        st.session_state.step = 4
        st.success("✅ 模型已加载，进入预测步骤！")
        st.rerun()

    # 逻辑2：仅上传数据文件 → 正常处理（不跳转）
    elif uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='gbk')
        else:
            df = pd.read_excel(uploaded_file)
        # 只保存原始数据，不编码
        st.session_state.raw_df = df
        st.session_state.current_file = uploaded_file.name
        st.success(f"✅ 文件已上传：{df.shape[0]}行 × {df.shape[1]}列")

    st.markdown("---")
    # 步骤导航
    st.write("**当前步骤**")
    steps = ["📤 上传", "🧹 清洗", "🤖 训练", "🔮 预测"]
    current_step_idx = st.session_state.step - 1
    for i, step in enumerate(steps):
        if i == current_step_idx:
            st.markdown(f"🔵 **{step}**")
        elif i < current_step_idx:
            st.markdown(f"✅ ~~{step}~~")
        else:
            st.markdown(f"⚪ {step}")

    st.markdown("---")
    # 在侧边栏中
    st.markdown("### 🤖 AI模型建议")

    if not st.session_state.get('ai_enabled', False):
        st.caption("⚠️ AI功能未启用")
    elif st.session_state.ai_advice is not None:
        advice = st.session_state.ai_advice

        # 显示当前建议（最新的一条）
        st.info(f"📊 **当前模型评估**: {advice['评价']}")

        st.markdown("**💡 推荐配置:**")
        test_size = advice['优化建议']['测试集比例']
        if isinstance(test_size, float) and test_size < 1:
            test_size_display = int(test_size * 100)
        else:
            test_size_display = test_size

        st.markdown(f"""
            - 隐藏层: {advice['优化建议']['隐藏层']}层
            - 神经元: {advice['优化建议']['神经元']}个
            - 激活函数: {advice['优化建议']['激活函数']}
            - 学习率: {advice['优化建议']['学习率']}
            - 批次大小: {advice['优化建议']['批次大小']}
            - 训练轮次: {advice['优化建议']['训练轮次']}
            - 测试集比例: {test_size_display}%
            """)

        st.success(f"🎯 **预期效果**: {advice['预期效果']}")

        # 历史记录折叠起来
        if st.session_state.ai_history and len(st.session_state.ai_history) > 0:
            with st.expander("📜 查看历史建议记录"):
                for i, h in enumerate(st.session_state.ai_history):
                    st.write(h)
    else:
        st.caption("点击主界面的【AI评价及建议】按钮获取优化建议")
    # 帮助信息（同步更新）
    with st.expander("❓ 使用帮助", expanded=False):
        st.markdown("""
        **数据格式：**
        - 最后一列为标签
        - 文本列需要编码
        - 数值列保持原样
        **模型上传：**
        - 仅上传训练好的模型文件（.h5）即可直接进入预测步骤
        - 预测时可选择手动输入特征或上传数据文件
        """)

# ========== 主界面（完全保留你的布局） ==========
st.title("🧠 神经网络训练器")
# 步骤导航显示
st.subheader("当前进度")
steps_names = ["上传数据", "清洗数据", "训练模型", "预测结果"]
cols = st.columns(4)
for i, (col, step_name) in enumerate(zip(cols, steps_names)):
    with col:
        if i + 1 == st.session_state.step:
            col.success(f"**{step_name}**✅")
        elif i + 1 < st.session_state.step:
            col.info(f"{step_name} ✓")
        else:
            col.write(f"{step_name}")
st.markdown("---")
# ========== 步骤1：上传数据（完全保留你的功能） ==========
if st.session_state.step == 1:
    if 'raw_df' in st.session_state and st.session_state.raw_df is not None:
        df = st.session_state.raw_df.copy()  # 使用原始数据的副本
        st.success(f"✅ 数据已加载")
        # 基本信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("数据行数", df.shape[0])
        with col2:
            st.metric("数据列数", df.shape[1])
        with col3:
            st.metric("缺失值", df.isnull().sum().sum())
        # 数据预览
        st.write("**原始数据预览（前10行）:**")
        st.dataframe(df.head(10))
        st.markdown("---")
        # ===== 数据编码区域 =====
        st.subheader("🔢 数据编码")
        st.info("系统将自动检测并编码文本列（object类型）为数字")
        # 检测文本列
        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        if text_cols:
            st.write(f"**检测到 {len(text_cols)} 个文本列，需要编码：**")
            # 显示各列的样本值
            sample_data = []
            for col in text_cols:
                unique_vals = df[col].dropna().unique()[:3]  # 最多显示3个唯一值
                sample_vals = ', '.join([str(v) for v in unique_vals])
                if len(df[col].dropna().unique()) > 3:
                    sample_vals += '...'
                sample_data.append({
                    '列名': col,
                    '非空值数量': df[col].count(),
                    '唯一值数量': df[col].nunique(),
                    '样本值': sample_vals
                })
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True, hide_index=True)
            # 编码按钮
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🔢 开始编码", type="primary", use_container_width=True):
                    label_encoders = {}
                    encoding_log = []
                    encoding_error = False
                    # 执行编码
                    for col in text_cols:
                        try:
                            le = LabelEncoder()
                            non_null_mask = df[col].notna()
                            if non_null_mask.any():
                                encoded_vals = le.fit_transform(
                                    df.loc[non_null_mask, col].astype(str)
                                )
                                encoded_vals = encoded_vals.astype('int64')
                                df.loc[non_null_mask, col] = encoded_vals
                                label_encoders[col] = le
                                if col == df.columns[-1]:
                                    df[col] = df[col].astype('int64')
                                encoding_log.append(f"✅ {col}: 已编码 ({len(le.classes_)}个类别)")
                            else:
                                encoding_log.append(f"⚠️ {col}: 全为缺失值，跳过编码")
                        except Exception as e:
                            encoding_log.append(f"❌ {col}: 编码失败 - {str(e)}")
                            encoding_error = True
                    if encoding_error:
                        st.error("❌ 编码失败，请检查数据格式")
                        with st.expander("📝 错误详情"):
                            for log in encoding_log:
                                st.write(log)
                    else:
                        # 保存编码后的数据和编码器
                        st.session_state.df = df
                        st.session_state.label_encoders = label_encoders
                        st.session_state.encoding_log = encoding_log
                        st.success("✅ 编码完成！")
                        with st.container():
                            st.markdown("### 📋 编码规则")
                            # 创建包含所有列编码规则的DataFrame
                            all_rules = []
                            for col, le in st.session_state.label_encoders.items():
                                for idx, cls in enumerate(le.classes_):
                                    all_rules.append({
                                        '列名': col,
                                        '原始值': cls,
                                        '编码值': idx
                                    })
                            rules_df = pd.DataFrame(all_rules)
                            st.dataframe(rules_df, hide_index=True, use_container_width=True)
                            csv_data = rules_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 下载编码规则表",
                                data=csv_data,
                                file_name="encoding_rules.csv",
                                mime="text/csv"
                            )
                        # 显示编码记录
                        with st.expander("📝 查看编码记录"):
                            for log in encoding_log:
                                st.write(log)
            with col2:
                if st.button("⏭️ 跳过编码（所有列为连续型数值）", use_container_width=True):
                    st.session_state.df = df  # 直接保存未编码的数据
                    st.session_state.label_encoders = {}
                    st.warning("⚠️ 跳过编码步骤，只适用于所有列为连续型数值的数据！")
        else:
            st.success("✅ 没有检测到文本列，无需编码")
            st.session_state.df = df  # 直接保存数据
        # 显示编码后的数据预览（如果有）
        if 'df' in st.session_state and st.session_state.df is not None:
            st.markdown("---")
            st.subheader("📋 处理后数据预览")
            st.dataframe(st.session_state.df.head(10))
            # 显示编码信息（如果有）
            if st.session_state.label_encoders:
                st.write("**编码信息：**")
                for col, le in st.session_state.label_encoders.items():
                    st.write(f"- {col}: 编码为 {len(le.classes_)} 个类别")
        # 下一步按钮
        if 'df' in st.session_state and st.session_state.df is not None:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("➡️ 进入数据清洗", type="primary", use_container_width=True):
                    st.session_state.step = 2
                    st.rerun()
    else:
        st.info("请在左侧边栏上传CSV或Excel文件")

# ========== 步骤2：数据清洗（保留所有功能+恢复原结构） ==========
elif st.session_state.step == 2:
    st.subheader("🧹 数据清洗")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        label_col = df.columns[-1]
        # ===== 首次检测 =====
        if 'detection_results' not in st.session_state or st.session_state.detection_results == {}:
            with st.spinner("正在评估数据质量..."):
                scores, total_score, max_total, warnings, detection_results = data_quality_score(
                    df, label_col, st.session_state.label_encoders
                )
                st.session_state.scores = scores
                st.session_state.total_score = total_score
                st.session_state.max_total = max_total
                st.session_state.warnings = warnings
                st.session_state.detection_results = detection_results
                st.session_state.original_df = df.copy()
                st.session_state.processed_df = df.copy()
                # 初始化临时数据
                if st.session_state.temp_df is None:
                    st.session_state.temp_df = df.copy()
        # ===== 显示评分卡（完全保留你的布局） =====
        st.markdown("### 📊 数据质量评分")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总分", f"{st.session_state.total_score}/{st.session_state.max_total}")
        with col2:
            if st.session_state.total_score >= 80:
                st.metric("评级", "🌟 优秀")
            elif st.session_state.total_score >= 60:
                st.metric("评级", "⚠️ 良好")
            else:
                st.metric("评级", "❌ 较差")
        with col3:
            st.metric("待处理项", len(st.session_state.warnings))
        st.markdown("#### 各项评分详情")
        score_cols = st.columns(4)
        for i, (item, score_info) in enumerate(st.session_state.scores.items()):
            with score_cols[i]:
                st.markdown(f"**{item}**")
                st.markdown(f"##### {score_info['score']}/{score_info['max_score']}")
                st.caption(score_info['status'])
                st.caption(score_info['detail'])
        if st.session_state.warnings:
            st.warning("⚠️ **检测到的问题：**")
            for w in st.session_state.warnings:
                st.write(f"- {w}")
        else:
            st.success("✅ 数据质量良好！")
        st.markdown("---")
        st.markdown("### 🛠️ 交互式数据处理")

        # ===== 第一步：缺失值处理（独立显示，不嵌套） =====
        st.markdown("#### 步骤1：选择缺失值处理方式")
        if st.session_state.detection_results['missing']['missing_cols']:
            missing_data = st.session_state.detection_results['missing']
            st.write("缺失值统计：")
            st.dataframe(pd.DataFrame({
                '列名': missing_data['missing_cols'],
                '缺失数量': [missing_data['missing_counts'][col] for col in missing_data['missing_cols']],
                '缺失比例': [f"{missing_data['missing_counts'][col] / len(df) * 100:.1f}%" for col in
                             missing_data['missing_cols']]
            }))
            st.info(f"📌 包含缺失值的行号：{[idx + 2 for idx in missing_data['rows_with_na']]}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**路径A：删除缺失行**")
                df_del = st.session_state.temp_df.dropna().reset_index(drop=True)
                st.metric("剩余行数", len(df_del))
                if st.button("选择删除缺失行", key="choose_del", use_container_width=True) and not \
                st.session_state.user_choices['missing_confirmed']:
                    st.session_state.user_choices['missing'] = 'delete'
                    st.session_state.temp_df = df_del
                    st.session_state.user_choices['missing_confirmed'] = True
                    st.rerun()
            with col2:
                st.markdown("**路径B：智能填充**")
                df_fill = st.session_state.temp_df.copy()
                for col in df_fill.columns[:-1]:
                    if df_fill[col].isnull().any():
                        if col in st.session_state.label_encoders:
                            df_fill[col] = df_fill[col].fillna(df_fill[col].mode()[0])
                        else:
                            df_fill[col] = df_fill[col].fillna(df_fill[col].median())
                st.metric("行数不变", len(df_fill))
                if st.button("选择智能填充", key="choose_fill", use_container_width=True) and not \
                st.session_state.user_choices['missing_confirmed']:
                    st.session_state.user_choices['missing'] = 'fill'
                    st.session_state.temp_df = df_fill
                    st.session_state.user_choices['missing_confirmed'] = True
                    st.rerun()
        else:
            st.success("✅ 当前数据无缺失值")
            st.session_state.user_choices['missing'] = 'none'
            st.session_state.user_choices['missing_confirmed'] = True
            if 'temp_df' not in st.session_state or st.session_state.temp_df is None:
                st.session_state.temp_df = df

        # ===== 第二步：异常值处理（优化版） =====
        st.markdown("---")
        st.markdown("#### 步骤2：选择异常值处理方式")

        # 安全检查
        if 'temp_df' not in st.session_state or st.session_state.temp_df is None:
            st.session_state.temp_df = df.copy()
        current_df = st.session_state.temp_df.copy()

        if len(current_df) == 0:
            st.error("❌ 当前数据为空，请先处理缺失值")
            st.stop()

        # ===== 只做一次检测，结果存入session_state =====
        if 'outlier_detection_results' not in st.session_state:
            outlier_detection_results = {
                'outlier_info': [],
                'outlier_mask': pd.Series(False, index=current_df.index),
                'col_methods': {}  # 记录每列用的方法
            }

            for col in current_df.columns[:-1]:
                data = current_df[col].dropna()
                if len(data) <= 3:
                    continue

                # 判断是否是分类变量
                if col in st.session_state.label_encoders:
                    # 分类变量：直接用IQR
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    col_outlier_mask = (current_df[col] < lower) | (current_df[col] > upper)
                    outliers = col_outlier_mask.sum()
                    method = "IQR (分类变量)"
                    distribution = "分类变量"
                else:
                    # 连续变量：判断正态性（考虑样本量）
                    n = len(data)
                    if n > 1000:
                        alpha = 0.01
                    elif n > 500:
                        alpha = 0.05
                    else:
                        alpha = 0.1

                    shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, n)))
                    is_normal = shapiro_p > alpha

                    if is_normal:
                        # 正态用Z-score
                        mean_val = data.mean()
                        std_val = data.std()
                        z_scores = np.abs((data - mean_val) / std_val)
                        col_outliers = (z_scores > 2.5)
                        col_outlier_mask = pd.Series(False, index=current_df.index)
                        col_outlier_mask[data.index] = col_outliers
                        outliers = col_outliers.sum()
                        method = "Z-score"
                        distribution = "正态分布"
                    else:
                        # 非正态用IQR
                        Q1 = data.quantile(0.25)
                        Q3 = data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        col_outlier_mask = (current_df[col] < lower) | (current_df[col] > upper)
                        outliers = col_outlier_mask.sum()
                        method = "IQR"
                        distribution = "非正态分布"

                # 记录检测结果
                outlier_detection_results['outlier_mask'] |= col_outlier_mask

                if outliers > 0:
                    outlier_detection_results['outlier_info'].append({
                        '列名': col,
                        '异常值数量': outliers,
                        '比例': f"{outliers / len(data) * 100:.1f}%",
                        '检测方法': method,
                        '分布形态': distribution
                    })

                outlier_detection_results['col_methods'][col] = {
                    'method': method,
                    'distribution': distribution
                }

            st.session_state.outlier_detection_results = outlier_detection_results

        # 获取检测结果
        outlier_info = st.session_state.outlier_detection_results['outlier_info']
        outlier_mask = st.session_state.outlier_detection_results['outlier_mask']
        col_methods = st.session_state.outlier_detection_results['col_methods']

        # 显示检测结果
        if outlier_info:
            st.write("检测到的异常值：")
            st.dataframe(pd.DataFrame(outlier_info))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**路径A：删除异常行**")
                df_del_outlier = current_df[~outlier_mask].reset_index(drop=True)
                st.metric("剩余行数", len(df_del_outlier))
                st.caption(f"删除 {outlier_mask.sum()} 行")
                if st.button("选择删除异常行", key="choose_del_outlier", use_container_width=True) and not \
                        st.session_state.user_choices['outlier_confirmed']:
                    st.session_state.user_choices['outlier'] = 'delete'
                    st.session_state.temp_df = df_del_outlier
                    st.session_state.user_choices['outlier_confirmed'] = True
                    st.rerun()

            with col2:
                st.markdown("**路径B：缩尾处理**")
                df_winsor = current_df.copy()
                for col in df_winsor.columns[:-1]:
                    lower = df_winsor[col].quantile(0.05)
                    upper = df_winsor[col].quantile(0.95)
                    df_winsor[col] = df_winsor[col].clip(lower, upper)
                st.metric("行数不变", len(df_winsor))
                if st.button("选择缩尾处理", key="choose_winsor", use_container_width=True) and not \
                        st.session_state.user_choices['outlier_confirmed']:
                    st.session_state.user_choices['outlier'] = 'winsor'
                    st.session_state.temp_df = df_winsor
                    st.session_state.user_choices['outlier_confirmed'] = True
                    st.rerun()

            with col3:
                st.markdown("**路径C：中位数替换**")
                df_median = current_df.copy()
                for col in df_median.columns[:-1]:
                    method_info = col_methods.get(col, {})
                    if method_info.get('method', '').startswith('IQR'):
                        # 如果是IQR检测的，用IQR的边界
                        data = df_median[col]
                        Q1 = data.quantile(0.25)
                        Q3 = data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        median = data.median()
                        df_median.loc[(data < lower) | (data > upper), col] = median
                    else:
                        # 如果是Z-score检测的，用Z-score的边界
                        data = df_median[col]
                        mean_val = data.mean()
                        std_val = data.std()
                        z_scores = np.abs((data - mean_val) / std_val)
                        median = data.median()
                        df_median.loc[z_scores > 2.5, col] = median
                st.metric("行数不变", len(df_median))
                if st.button("选择中位数替换", key="choose_median", use_container_width=True) and not \
                        st.session_state.user_choices['outlier_confirmed']:
                    st.session_state.user_choices['outlier'] = 'median'
                    st.session_state.temp_df = df_median
                    st.session_state.user_choices['outlier_confirmed'] = True
                    st.rerun()
        else:
            st.success("✅ 当前数据无异常值")
            st.session_state.user_choices['outlier'] = 'none'
            st.session_state.user_choices['outlier_confirmed'] = True

        # ===== 第三步：样本平衡处理（根据任务类型动态显示） =====
        st.markdown("---")
        st.markdown("#### 步骤3：选择样本平衡处理方式")

        # 安全检查
        if 'temp_df' not in st.session_state or st.session_state.temp_df is None:
            st.session_state.temp_df = df.copy()
        current_df = st.session_state.temp_df.copy()
        if len(current_df) == 0:
            st.error("❌ 当前数据为空，请先处理缺失值和异常值")
            st.stop()

        # 判断是分类还是回归问题
        is_classification = label_col in st.session_state.label_encoders

        if is_classification:
            # ===== 分类问题的平衡处理 =====
            value_counts = current_df[label_col].value_counts()
            if len(value_counts) > 1:
                st.write("当前类别分布：")
                dist_df = pd.DataFrame({
                    '类别': value_counts.index,
                    '样本数': value_counts.values,
                    '占比': [f"{v / len(current_df) * 100:.1f}%" for v in value_counts.values]
                })
                st.dataframe(dist_df)
                imbalance_ratio = value_counts.max() / value_counts.min()
                st.info(f"不平衡比例: {imbalance_ratio:.2f}:1")

                # 分类问题的4个选项
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown("**路径A：不处理**")
                    st.caption("保持数据不变")
                    st.metric("样本数", len(current_df))
                    if st.button("选择不处理", key="choose_no_balance_cls", use_container_width=True) and not \
                            st.session_state.user_choices['balance_confirmed']:
                        st.session_state.user_choices['balance'] = 'none'
                        st.session_state.temp_df = current_df
                        st.session_state.user_choices['balance_confirmed'] = True
                        st.rerun()

                with col2:
                    st.markdown("**路径B：Class Weight**")
                    st.caption("训练时加权")
                    st.metric("样本数", len(current_df))
                    if st.button("选择Class Weight", key="choose_class_weight", use_container_width=True) and not \
                            st.session_state.user_choices['balance_confirmed']:
                        st.session_state.user_choices['balance'] = 'class_weight'
                        # 不改变数据，只记录选择
                        st.session_state.temp_df = current_df
                        st.session_state.user_choices['balance_confirmed'] = True
                        st.success("✅ 已选择Class Weight，将在训练时自动应用")
                        st.rerun()

                with col3:
                    st.markdown("**路径C：SMOTENC**")
                    st.caption("智能处理混合数据")
                    try:
                        X = current_df.iloc[:, :-1]
                        y = current_df.iloc[:, -1]

                        # 识别分类变量的索引
                        categorical_features = []
                        for i, col in enumerate(X.columns):
                            if col in st.session_state.label_encoders and col != label_col:
                                categorical_features.append(i)

                        if categorical_features:
                            from imblearn.over_sampling import SMOTENC

                            smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)
                            X_res, y_res = smote_nc.fit_resample(X, y)
                        else:
                            # 如果没有分类变量，用普通SMOTE
                            from imblearn.over_sampling import SMOTE

                            smote = SMOTE(random_state=42)
                            X_res, y_res = smote.fit_resample(X, y)

                        df_smote = pd.concat([
                            pd.DataFrame(X_res, columns=X.columns),
                            pd.Series(y_res, name=label_col)
                        ], axis=1)

                        st.metric("处理后样本数", len(df_smote))
                        if st.button("选择SMOTENC", key="choose_smote_nc", use_container_width=True) and not \
                                st.session_state.user_choices['balance_confirmed']:
                            st.session_state.user_choices['balance'] = 'smote_nc'
                            st.session_state.temp_df = df_smote
                            st.session_state.user_choices['balance_confirmed'] = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"SMOTENC失败: {str(e)}")

                with col4:
                    st.markdown("**路径D：欠采样**")
                    st.caption("减少多数类")
                    try:
                        X = current_df.iloc[:, :-1]
                        y = current_df.iloc[:, -1]
                        from imblearn.under_sampling import RandomUnderSampler

                        rus = RandomUnderSampler(random_state=42)
                        X_res, y_res = rus.fit_resample(X, y)
                        df_under = pd.concat([
                            pd.DataFrame(X_res, columns=X.columns),
                            pd.Series(y_res, name=label_col)
                        ], axis=1)
                        st.metric("处理后样本数", len(df_under))
                        if st.button("选择欠采样", key="choose_under_cls", use_container_width=True) and not \
                                st.session_state.user_choices['balance_confirmed']:
                            st.session_state.user_choices['balance'] = 'undersample'
                            st.session_state.temp_df = df_under
                            st.session_state.user_choices['balance_confirmed'] = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"欠采样失败: {str(e)}")
            else:
                st.success("✅ 只有一个类别，无需平衡")
                st.session_state.user_choices['balance'] = 'single'
                st.session_state.user_choices['balance_confirmed'] = True

        else:
            # ===== 回归问题的平衡处理 =====
            st.write("回归问题：使用GMM聚类进行样本平衡性分析")
            y = current_df[label_col].values
            labels, gmm = gmm_clustering_for_regression(y, max_components=8)

            if gmm is not None and len(np.unique(labels)) > 1:
                cluster_counts = pd.Series(labels).value_counts().sort_index()
                st.write("GMM聚类结果：")
                cluster_df = pd.DataFrame({
                    '聚类': cluster_counts.index,
                    '样本数': cluster_counts.values,
                    '占比': [f"{v / len(y) * 100:.1f}%" for v in cluster_counts.values]
                })
                st.dataframe(cluster_df)
                imbalance_ratio = cluster_counts.max() / cluster_counts.min()
                st.info(f"聚类不平衡比例: {imbalance_ratio:.2f}:1")

                # 回归问题的3个选项
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**路径A：不处理**")
                    st.metric("样本数", len(current_df))
                    if st.button("选择不处理", key="choose_no_balance_reg", use_container_width=True) and not \
                            st.session_state.user_choices['balance_confirmed']:
                        st.session_state.user_choices['balance'] = 'none'
                        st.session_state.temp_df = current_df
                        st.session_state.user_choices['balance_confirmed'] = True
                        st.rerun()

                with col2:
                    st.markdown("**路径B：GMM过采样**")
                    st.caption("基于聚类生成合成样本")
                    try:
                        df_balanced, _ = balance_regression_by_gmm(current_df, label_col, method='smote')
                        st.metric("处理后样本数", len(df_balanced))
                        if st.button("选择GMM过采样", key="choose_smote_reg", use_container_width=True) and not \
                                st.session_state.user_choices['balance_confirmed']:
                            st.session_state.user_choices['balance'] = 'gmm_oversample'
                            st.session_state.temp_df = df_balanced
                            st.session_state.user_choices['balance_confirmed'] = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"GMM过采样失败: {str(e)}")

                with col3:
                    st.markdown("**路径C：GMM欠采样**")
                    st.caption("减少多数聚类样本")
                    try:
                        df_balanced, _ = balance_regression_by_gmm(current_df, label_col, method='undersample')
                        st.metric("处理后样本数", len(df_balanced))
                        if st.button("选择GMM欠采样", key="choose_under_reg", use_container_width=True) and not \
                                st.session_state.user_choices['balance_confirmed']:
                            st.session_state.user_choices['balance'] = 'gmm_undersample'
                            st.session_state.temp_df = df_balanced
                            st.session_state.user_choices['balance_confirmed'] = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"GMM欠采样失败: {str(e)}")
            else:
                st.success("✅ 目标值呈现单峰分布，无需平衡处理")
                st.session_state.user_choices['balance'] = 'single'
                st.session_state.user_choices['balance_confirmed'] = True

        # ===== 第四步：特征筛选（仅当前三步都确认后才显示） =====
        if st.session_state.user_choices['missing_confirmed'] and st.session_state.user_choices['outlier_confirmed'] and \
                st.session_state.user_choices['balance_confirmed']:
            st.markdown("---")
            st.markdown("#### 步骤4：特征重要性评估与特征筛选")
            # 安全检查
            if 'temp_df' not in st.session_state or st.session_state.temp_df is None:
                st.session_state.temp_df = df.copy()
            final_df = st.session_state.temp_df.copy()
            if len(final_df) == 0:
                st.error("❌ 当前数据为空，请返回重新选择")
                st.stop()
            # 计算特征重要性
            try:
                X = final_df.iloc[:, :-1]
                y = final_df.iloc[:, -1]
                if len(final_df) >= 30 and not X.isnull().any().any():
                    if label_col in st.session_state.label_encoders:
                        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                        y_encoded = y.values
                    else:
                        rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                        y_encoded = y.values
                    rf_model.fit(X, y_encoded)
                    importance = rf_model.feature_importances_
                    importance_normalized = importance / importance.sum() * 100
                    importance_df = pd.DataFrame({
                        '特征': X.columns,
                        '重要性(%)': importance_normalized
                    }).sort_values('重要性(%)', ascending=False)
                    st.write("特征重要性排名：")
                    st.dataframe(importance_df.style.format({'重要性(%)': '{:.2f}%'}))
                    # 特征筛选
                    keep_features = st.multiselect(
                        "选择要保留的特征（默认全选）",
                        options=X.columns.tolist(),
                        default=X.columns.tolist()
                    )
                    if st.button("应用特征筛选", key="apply_features", use_container_width=True) and not \
                    st.session_state.user_choices['features_confirmed']:
                        if not keep_features:
                            st.warning(
                                "⚠️ 未选择任何特征列！模型训练需要至少一个特征作为输入，请勾选至少一个特征列后重试～")
                        else:
                            st.session_state.user_choices['features'] = keep_features
                            # 应用特征筛选
                            final_selected_df = final_df[keep_features + [label_col]]
                            st.session_state.processed_df = final_selected_df
                            # 重新评分
                            with st.spinner("正在重新评估最终数据质量..."):
                                scores, total_score, max_total, warnings, detection_results = data_quality_score(
                                    final_selected_df, label_col, st.session_state.label_encoders
                                )
                                st.session_state.scores = scores
                                st.session_state.total_score = total_score
                                st.session_state.max_total = max_total
                                st.session_state.warnings = warnings
                                st.session_state.detection_results = detection_results
                            st.success("✅ 数据清洗完成！")
                            st.session_state.user_choices['features_confirmed'] = True
                            st.rerun()
                else:
                    st.warning("⚠️ 当前数据无法计算特征重要性，请返回上一步调整")
                    if st.button("直接跳过特征筛选", key="skip_features") and not st.session_state.user_choices[
                        'features_confirmed']:
                        st.session_state.user_choices['features'] = X.columns.tolist()
                        st.session_state.processed_df = final_df
                        st.session_state.user_choices['features_confirmed'] = True
                        st.rerun()
            except Exception as e:
                st.error(f"特征重要性计算失败: {str(e)}")
        else:
            st.info("ℹ️ 请先完成前3步处理（缺失值→异常值→样本平衡），再进行特征筛选")

        # ===== 重新清洗按钮 =====
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 重新清洗（重置所有步骤）", type="secondary", use_container_width=True):
                # 重置所有状态
                st.session_state.user_choices = {
                    'missing': None,
                    'outlier': None,
                    'balance': None,
                    'features': None,
                    'missing_confirmed': False,
                    'outlier_confirmed': False,
                    'balance_confirmed': False,
                    'features_confirmed': False
                }
                # 重置检测结果
                if 'outlier_detection_results' in st.session_state:
                    del st.session_state.outlier_detection_results

                # 关键修复：恢复到原始数据，而不是保持当前数据
                st.session_state.temp_df = st.session_state.df.copy()  # 这里要用 df，不是 current_df
                st.session_state.processed_df = st.session_state.df.copy()

                # 重新计算初始评分
                with st.spinner("正在恢复初始数据质量评分..."):
                    scores, total_score, max_total, warnings, detection_results = data_quality_score(
                        st.session_state.df, label_col, st.session_state.label_encoders
                    )
                    st.session_state.scores = scores
                    st.session_state.total_score = total_score
                    st.session_state.max_total = max_total
                    st.session_state.warnings = warnings
                    st.session_state.detection_results = detection_results
                st.success("✅ 已恢复到初始清洗状态！")
                st.rerun()

        # ===== 数据分布可视化+下载功能（原有逻辑不变） =====
        st.markdown("### 📈 数据分布可视化")
        viz_df = st.session_state.temp_df.copy() if st.session_state.temp_df is not None else st.session_state.df.copy()
        label_col = viz_df.columns[-1]
        if len(viz_df.columns) > 0:
            cate_cols = list(st.session_state.label_encoders.keys())
            num_cols = [col for col in viz_df.columns if
                        col not in cate_cols and viz_df[col].dtype in ['int64', 'float64']]
            all_cols = viz_df.columns.tolist()
            selected_col = st.selectbox("选择要可视化的列", all_cols, key="viz_all_col")
            is_label_col = (selected_col == label_col)
            is_cate_col = (selected_col in cate_cols)
            is_num_col = (selected_col in num_cols)
            label_is_cate = (label_col in cate_cols)
            label_is_num = (label_col in num_cols)
            col_data = viz_df[selected_col].dropna()

            # 图表生成逻辑（完全保留）
            if is_cate_col or (is_label_col and label_is_cate):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
                le = st.session_state.label_encoders[selected_col]
                col_data_text = col_data.map(
                    lambda x: le.classes_[int(x)] if int(x) < len(le.classes_) else f'未知({x})')
                value_counts = col_data_text.value_counts().sort_index()
                ax1.bar(range(len(value_counts)), value_counts.values, color='#1f77b4')
                ax1.set_xticks(range(len(value_counts)))
                ax1.set_xticklabels(value_counts.index, rotation=30, ha='right', fontsize=9)
                ax1.set_title(f'{selected_col} - 柱状图', fontsize=12)
                ax1.set_xlabel('类别', fontsize=10)
                ax1.set_ylabel('频数', fontsize=10)
                for i, count in enumerate(value_counts.values):
                    ax1.text(i, count + 0.1, f'{int(count)}', ha='center', va='bottom', fontsize=8)
                wedges, texts, autotexts = ax2.pie(
                    value_counts.values,
                    labels=value_counts.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                            '#17becf', '#1f77b4']
                )
                for text in texts:
                    text.set_rotation(30)
                    text.set_fontsize(9)
                ax2.set_title(f'{selected_col} - 扇形图', fontsize=12)
            elif is_num_col and not is_label_col:
                if label_is_num:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
                    unique_vals = col_data.nunique()
                    bins = min(20, unique_vals)
                    ax1.hist(col_data, bins=bins, edgecolor='black', alpha=0.7, color='#1f77b4')
                    ax1.set_title(f'{selected_col} - 直方图', fontsize=12)
                    ax1.set_xlabel('值', fontsize=10)
                    ax1.set_ylabel('频数', fontsize=10)
                    label_data = viz_df[label_col].dropna()
                    common_idx = col_data.index.intersection(label_data.index)
                    ax2.scatter(col_data.loc[common_idx], label_data.loc[common_idx], alpha=0.6, color='#ff7f0e')
                    ax2.set_title(f'{selected_col} vs {label_col} - 散点图', fontsize=12)
                    ax2.set_xlabel(selected_col, fontsize=10)
                    ax2.set_ylabel(label_col, fontsize=10)
                    ax2.grid(alpha=0.3)
                else:
                    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
                    unique_vals = col_data.nunique()
                    bins = min(20, unique_vals)
                    ax.hist(col_data, bins=bins, edgecolor='black', alpha=0.7, color='#1f77b4')
                    ax.set_title(f'{selected_col} - 直方图', fontsize=12)
                    ax.set_xlabel('值', fontsize=10)
                    ax.set_ylabel('频数', fontsize=10)
            elif is_label_col and label_is_num:
                fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
                unique_vals = col_data.nunique()
                bins = min(20, unique_vals)
                ax.hist(col_data, bins=bins, edgecolor='black', alpha=0.7, color='#1f77b4')
                ax.set_title(f'{label_col} - 直方图（标签列）', fontsize=12)
                ax.set_xlabel('值', fontsize=10)
                ax.set_ylabel('频数', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)
            # 图表下载
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="📥 下载图表（PNG）",
                data=buf,
                file_name=f"{selected_col}_可视化图表.png",
                mime="image/png",
                use_container_width=True
            )
            # 列统计信息
            st.markdown("#### 📊 列统计信息")
            if is_num_col or (is_label_col and label_is_num):
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("均值", f"{col_data.mean():.2f}")
                with stats_col2:
                    st.metric("标准差", f"{col_data.std():.2f}")
                with stats_col3:
                    st.metric("最小值", f"{col_data.min():.2f}")
                with stats_col4:
                    st.metric("最大值", f"{col_data.max():.2f}")
            else:
                total = len(col_data)
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.metric("类别数", f"{col_data.nunique()}")
                with stats_col2:
                    st.metric("非空数量", f"{total}")
        else:
            st.info("⚠️ 暂无数据可可视化")

        # 清洗后数据预览+下载
        st.markdown("### 📋 清洗后数据预览")
        st.dataframe(viz_df.head(10), use_container_width=True)
        st.write(f"当前数据形状: **{viz_df.shape[0]}行 × {viz_df.shape[1]}列**")
        st.markdown("#### 📥 清洗后数据下载")
        col_download_csv, col_download_excel = st.columns(2)
        with col_download_csv:
            csv_data = viz_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="下载CSV格式",
                data=csv_data,
                file_name=f"清洗后数据_{viz_df.shape[0]}行{viz_df.shape[1]}列.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_download_excel:
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                viz_df.to_excel(writer, sheet_name='清洗后数据', index=False)
            excel_buf.seek(0)
            st.download_button(
                label="下载Excel格式",
                data=excel_buf,
                file_name=f"清洗后数据_{viz_df.shape[0]}行{viz_df.shape[1]}列.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # 进入下一步按钮
        if st.session_state.user_choices['features_confirmed']:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col2:
                if st.button("➡️ 进入模型训练", type="primary", use_container_width=True):
                    st.session_state.step = 3
                    st.rerun()
    else:
        st.warning("请先上传数据")
    

# ========== 步骤 3：模型训练 ==========
elif st.session_state.step == 3:
    st.subheader("🤖 模型训练")
    if st.session_state.df is not None:
        # 使用清洗后的最终数据（适配清洗流程）
        df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
        st.write("📌 **标签列规定**：提交表格的最后一列作为标签列")
        all_columns = df.columns.tolist()
        target = all_columns[-1] if len(all_columns) > 0 else None
        feature = all_columns[:-1] if len(all_columns) > 1 else []

        if target:
            st.success(f"✅ 系统已识别标签列：**{target}**")
        else:
            st.error("❌ 无法识别标签列，请检查数据格式")
            st.stop()

        st.markdown("### 1. 数据识别")
        st.info(f"系统识别到：{len(feature)}个特征列，1个标签列")

        # 显示数据结构
        col1, col2 = st.columns(2)
        with col1:
            st.write("**特征列（前5列）:**")
            if len(feature) > 0:
                st.write(", ".join(feature[:5]))
                if len(feature) > 5:
                    st.write(f"... 等{len(feature)}列")
            else:
                st.warning("⚠️ 未识别到特征列")

        with col2:
            st.write("**标签列:**")
            if target:
                st.success(f"{target}")
                st.write(f"数据类型: {df[target].dtype}")

        # 数据预览（直接显示，不需选择）
        if target and feature:
            st.success(f"✅ 确定使用：{len(feature)}个特征列，1个标签列")
            # 数据预览
            with st.expander("📊 查看训练数据"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**特征数据预览（前8行）:**")
                    st.dataframe(df[feature].head(8), use_container_width=True)
                with col2:
                    st.write("**标签数据预览（前8行）:**")
                    st.dataframe(df[[target]].head(8), use_container_width=True)

        st.markdown("---")

        # 3. ANN模型参数配置
        st.markdown("### 2. ANN模型配置")
        col1, col2 = st.columns(2)
        with col1:
            # 网络结构
            st.write("**网络结构**")
            hidden_layers = st.slider("隐藏层层数", 1, 5, 2, key="slider_hidden_layers")
            neurons_per_layer = st.slider("每层神经元数", 8, 256, 64, key="slider_neurons")
            activation = st.selectbox("激活函数", ["relu", "sigmoid", "tanh"], key="select_activation")

        with col2:
            # 训练参数
            st.write("**训练参数**")
            epochs = st.slider("训练轮次", 10, 500, 50, key="slider_epochs")
            batch_size = st.slider("批次大小", 8, 128, 32, key="slider_batch")
            learning_rate = st.selectbox("学习率", [0.1, 0.01, 0.001, 0.0005, 0.0001, 0.00001], index=2, key="select_lr")

        # 4. 数据划分
        st.markdown("### 3. 数据划分")
        test_size = st.slider("测试集比例 (%)", 10, 40, 20, key="slider_test_size")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总样本数", len(df))
        with col2:
            train_samples = int(len(df) * (1 - test_size / 100))
            st.metric("训练样本", train_samples)
        with col3:
            test_samples = int(len(df) * (test_size / 100))
            st.metric("测试样本", test_samples)

        st.markdown("---")

        # 5. 开始训练按钮
        st.markdown("### 4. 开始训练")

        if st.button("🚀 开始训练模型", type="primary", use_container_width=True, key="btn_start_train"):
            # 保存配置
            st.session_state.model_config = {
                'target': target,
                'feature': feature,
                'hidden_layers': hidden_layers,
                'neurons': neurons_per_layer,
                'activation': activation,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'test_size': test_size / 100
            }

            # 数据预处理
            X = df[feature].values
            y = df[target].values

            # 判断任务类型（分类/回归）
            is_classification = target in st.session_state.label_encoders
            st.session_state.is_classification = is_classification

            # 分类任务处理
            if is_classification:
                le = st.session_state.label_encoders[target]
                st.session_state.label_encoder = le
                st.session_state.num_classes = len(le.classes_)
                # 标签编码（确保为整数类型）
                y = y.astype(int)
                # 多分类任务转换为one-hot编码
                if len(le.classes_) > 2:
                    y = tf.keras.utils.to_categorical(y, num_classes=len(le.classes_))
            else:
                st.session_state.num_classes = None

            # 数据划分
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42, stratify=y if is_classification else None
            )

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.session_state.scaler = scaler
            st.session_state.X_test = X_test_scaled
            st.session_state.y_test = y_test

            # 构建模型
            model = tf.keras.Sequential()
            # 输入层
            model.add(tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)))
            # 隐藏层
            for _ in range(hidden_layers):
                model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation))
            # 输出层
            if is_classification:
                if len(le.classes_) == 2:
                    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                    loss = 'binary_crossentropy'
                    metric = 'accuracy'
                else:
                    model.add(tf.keras.layers.Dense(len(le.classes_), activation='softmax'))
                    loss = 'categorical_crossentropy'
                    metric = 'accuracy'
            else:
                model.add(tf.keras.layers.Dense(1, activation='linear'))
                loss = 'mse'
                metric = 'mae'

            # 编译模型
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

            # 训练模型
            with st.spinner("模型训练中..."):
                # 在数据划分和标准化之后，构建模型之前，添加这段代码
                # ===== 新增：Class Weight处理（仅分类问题有效） =====
                class_weight_dict = None
                if st.session_state.user_choices.get('balance') == 'class_weight' and is_classification:
                    try:
                        from sklearn.utils.class_weight import compute_class_weight

                        # 获取原始标签（非one-hot）
                        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                            # 如果是one-hot编码，转换回标签
                            y_train_labels = np.argmax(y_train, axis=1)
                        else:
                            y_train_labels = y_train

                        # 计算权重
                        classes = np.unique(y_train_labels)
                        weights = compute_class_weight('balanced', classes=classes, y=y_train_labels)
                        class_weight_dict = dict(zip(classes, weights))

                        st.info(f"📊 应用Class Weight：{class_weight_dict}")
                    except Exception as e:
                        st.warning(f"Class Weight计算失败，将不使用：{str(e)}")

                # ===== 根据样本量动态设置验证集策略 =====
                n_samples = len(X_train)  # 训练集样本数

                if n_samples < 500:
                    # 小样本：不用验证集，或者留极少
                    validation_split = 0.0
                    st.info(f"📌 样本量较少({n_samples}行)，不使用验证集，充分利用所有数据训练")
                elif n_samples < 2000:
                    # 中等样本：用10%验证集
                    validation_split = 0.1
                    st.info(f"📌 样本量适中({n_samples}行)，使用10%数据作为验证集")
                else:
                    # 大样本：用15%验证集
                    validation_split = 0.15
                    st.info(f"📌 样本量充足({n_samples}行)，使用15%数据作为验证集")

                # 训练模型
                history = model.fit(
                    X_train_scaled, y_train,
                    class_weight=class_weight_dict,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,  # 动态设置
                    verbose=1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)]
                    if validation_split > 0 else None
                )

            # 保存模型和训练记录
            st.session_state.model = model
            st.session_state.history = history.history
            st.success("✅ 模型训练完成！")
            st.session_state.step = 4
            st.rerun()

        st.markdown("---")

        # 操作按钮（加唯一 key，避免重复）
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← 返回数据清洗", type="secondary", key="btn_back_clean"):
                st.session_state.ai_messages = [
                    {"role": "system",
                     "content": "你是一个机器学习调参专家，请基于对话历史给出连贯的建议。记住之前给用户的建议，让每次建议都有连续性。"}
                ]
                st.session_state.ai_history = []
                st.session_state.ai_advice = None
                st.session_state.step = 2
                st.rerun()

        with col2:
            if st.button("跳过配置 →", type="secondary", key="btn_skip_config"):
                # 使用默认配置
                if target and len(feature) > 0:
                    st.session_state.model_config = {
                        'target': target,
                        'feature': feature,
                        'hidden_layers': 2,
                        'neurons': 64,
                        'activation': 'relu',
                        'epochs': 50,
                        'batch_size': 32,
                        'learning_rate': 0.001,
                        'test_size': 0.2
                    }
                    # 自动执行训练流程
                    X = df[feature].values
                    y = df[target].values
                    is_classification = target in st.session_state.label_encoders
                    st.session_state.is_classification = is_classification

                    if is_classification:
                        le = st.session_state.label_encoders[target]
                        st.session_state.label_encoder = le
                        st.session_state.num_classes = len(le.classes_)
                        y = y.astype(int)
                        if len(le.classes_) > 2:
                            y = tf.keras.utils.to_categorical(y, num_classes=len(le.classes_))
                    else:
                        st.session_state.num_classes = None

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y if is_classification else None
                    )

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    st.session_state.scaler = scaler
                    st.session_state.X_test = X_test_scaled
                    st.session_state.y_test = y_test

                    model = tf.keras.Sequential()
                    model.add(tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)))
                    for _ in range(2):
                        model.add(tf.keras.layers.Dense(64, activation='relu'))
                    if is_classification:
                        if len(le.classes_) == 2:
                            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                            loss = 'binary_crossentropy'
                            metric = 'accuracy'
                        else:
                            model.add(tf.keras.layers.Dense(len(le.classes_), activation='softmax'))
                            loss = 'categorical_crossentropy'
                            metric = 'accuracy'
                    else:
                        model.add(tf.keras.layers.Dense(1, activation='linear'))
                        loss = 'mse'
                        metric = 'mae'

                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

                    with st.spinner("模型训练中..."):
                        history = model.fit(
                            X_train_scaled, y_train,
                            epochs=50,
                            batch_size=32,
                            validation_split=0.1,
                            verbose=1,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
                        )

                    st.session_state.model = model
                    st.session_state.history = history.history
                    st.success("✅ 模型训练完成！")
                    st.session_state.step = 4
                    st.rerun()
                else:
                    st.error("无法自动识别数据格式")
    else:
        st.error("请先上传文件并完成数据清洗")
        if st.button("返回上传页面", key="btn_back_upload"):
            st.session_state.step = 1
            st.rerun()

# ========== 步骤4：模型训练与预测（修复无限循环版+适配无配置文件上传+修复Plotly报错） ==========
elif st.session_state.step == 4:
    try:
        st.subheader("🔮 模型训练与预测")
        # 检查是否有上传的模型
        if st.session_state.uploaded_model is not None:
            # 使用上传的模型
            model = st.session_state.uploaded_model
            df = st.session_state.df
            config = st.session_state.model_config
            st.success("✅ 使用上传的模型进行预测")
            # 自动识别任务类型（补充逻辑）
            label_col = config['target']
            if label_col in st.session_state.label_encoders:
                st.session_state.is_classification = True
                st.session_state.label_encoder = st.session_state.label_encoders[label_col]
                st.session_state.num_classes = len(st.session_state.label_encoder.classes_)
                st.info(f"✅ 自动识别为分类问题：标签已编码为 {st.session_state.num_classes} 个类别")
            else:
                st.session_state.is_classification = False
                st.info(f"✅ 自动识别为回归问题：标签列类型为 {df[label_col].dtype}")
            # 初始化标准化器（默认）
            if 'scaler' not in st.session_state or st.session_state.scaler is None:
                scaler = StandardScaler()
                X = df[config['feature']].values
                scaler.fit(X)
                st.session_state.scaler = scaler
        elif st.session_state.df is not None and 'model_config' in st.session_state:
            # 使用训练好的模型
            df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
            config = st.session_state.model_config
            # 检查是否已经训练过模型
            if 'model' not in st.session_state or st.session_state.model is None:
                st.info("请先训练模型")
                st.stop()
            else:
                model = st.session_state.model
                scaler = st.session_state.scaler
        else:
            st.warning("请先完成模型配置或上传训练好的模型")
            if st.button("返回模型配置", type="primary"):
                st.session_state.step = 3
                st.rerun()
            st.stop()
        # 显示配置信息
        st.markdown("### 📋 训练配置")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**数据配置:**")
            st.write(f"- 特征列: {len(config['feature'])}个")
            st.write(f"- 标签列: {config['target']}")
            if df is not None:
                st.write(f"- 总样本: {len(df)}个")
            st.write(f"- 测试集: {config.get('test_size', 0.2) * 100:.0f}%")
        with col2:
            st.write("**模型配置:**")
            st.write(f"- 隐藏层: {config.get('hidden_layers', 2)}层")
            st.write(f"- 神经元: {config.get('neurons', 64)}个/层")
            st.write(f"- 激活函数: {config.get('activation', 'relu')}")
            st.write(f"- 训练轮次: {config.get('epochs', 50)}")
            st.write(f"- 批次大小: {config.get('batch_size', 32)}")
            st.write(f"- 学习率: {config.get('learning_rate', 0.001)}")
        # 如果模型已训练，显示训练结果
        if 'model' in st.session_state and st.session_state.model is not None and 'history' in st.session_state:
            st.markdown("---")
            st.markdown("### 📈 训练结果")
            # 创建选项卡
            tab1, tab2, tab3 = st.tabs(["训练曲线", "评估指标", "模型结构"])

            with tab1:
                fig = go.Figure()
                if st.session_state.history:
                    # 训练损失
                    loss_history = st.session_state.history.get('loss', [])
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(loss_history) + 1)),
                        y=loss_history,
                        mode='lines',
                        name='训练损失',
                        line=dict(color='blue')
                    ))

                    # 如果有验证损失才显示
                    val_loss = st.session_state.history.get('val_loss')
                    if val_loss and len(val_loss) > 0:
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(val_loss) + 1)),
                            y=val_loss,
                            mode='lines',
                            name='验证损失',
                            line=dict(color='red', dash='dash')
                        ))

                fig.update_layout(
                    title='训练损失曲线' + (' (含验证集)' if val_loss else ' (无验证集)'),
                    xaxis_title='训练轮次',
                    yaxis_title='损失值'
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                if st.session_state.model is not None and hasattr(st.session_state, 'X_test') and hasattr(
                        st.session_state, 'y_test'):
                    test_loss, test_metric = st.session_state.model.evaluate(
                        st.session_state.X_test, st.session_state.y_test, verbose=0
                    )
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("测试集损失", f"{test_loss:.4f}")
                    with col2:
                        metric_name = "准确率" if st.session_state.is_classification else "MAE"
                        st.metric(f"测试集{metric_name}", f"{test_metric:.4f}")
                    with col3:
                        st.metric("训练轮次", config.get('epochs', 50))

            with tab3:
                if st.session_state.model is not None:
                    st.write("**模型结构:**")
                    model_summary = []
                    st.session_state.model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text("\n".join(model_summary))
                    st.metric("总参数量", f"{st.session_state.model.count_params():,}")
        st.markdown("---")
        # ========== 模型保存功能 ==========
        col1, col2, col3 = st.columns(3)
        with col1:
            # 保存模型（修复临时文件泄漏）
            tmp_file_path = None
            model_bytes = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                    st.session_state.model.save(tmp_file.name)
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name
                    with open(tmp_file.name, 'rb') as f:
                        model_bytes = f.read()
            except Exception as e:
                st.error(f"❌ 模型保存失败：{str(e)}")
            finally:
                # 强制删除临时文件
                if tmp_file_path and os.path.exists(tmp_file_path):
                    try:
                        os.unlink(tmp_file_path)
                    except PermissionError:
                        time.sleep(1)
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                    except Exception as e:
                        st.warning(f"⚠️ 临时文件清理警告：{str(e)}")
            # 下载按钮（仅当模型保存成功时显示）
            if model_bytes is not None:
                st.download_button(
                    label="📥 下载模型文件 (.h5)",
                    data=model_bytes,
                    file_name=f"ann_model_{time.strftime('%Y%m%d_%H%M%S')}.h5",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        with col2:
            # 保存配置（原有逻辑不变）
            scaler_info = {
                'mean': st.session_state.scaler.mean_.tolist(),
                'scale': st.session_state.scaler.scale_.tolist(),
                'feature_names': config['feature'],
                'target_name': config['target'],
                'is_classification': st.session_state.is_classification,
                'num_classes': st.session_state.num_classes if st.session_state.is_classification else None,
                'label_encoder_classes': st.session_state.label_encoder.classes_.tolist()
                if hasattr(st.session_state, 'label_encoder') and st.session_state.label_encoder is not None else None
            }
            scaler_json = json.dumps(scaler_info, indent=2, ensure_ascii=False)
            st.download_button(
                label="📄 下载预处理配置 (.json)",
                data=scaler_json,
                file_name=f"model_config_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        with col3:
            if st.button("🤖 AI评价及建议", use_container_width=True, key="ai_advice_btn"):
                if not st.session_state.get('ai_enabled', False):
                    st.error("❌ AI功能未启用，请检查API Key配置")
                    st.stop()

                # ===== 获取测试集比例，确保是数值 =====
                test_size_value = config.get('test_size', 0.2)
                if isinstance(test_size_value, str):
                    test_size_value = float(test_size_value)
                test_size_percent = int(test_size_value * 100)

                # ===== 获取准确率 =====
                if st.session_state.is_classification:
                    # 从评估指标中获取准确率
                    if hasattr(st.session_state, 'X_test') and hasattr(st.session_state, 'y_test'):
                        test_loss, test_metric = st.session_state.model.evaluate(
                            st.session_state.X_test, st.session_state.y_test, verbose=0
                        )
                        accuracy = f"{test_metric:.2%}"
                    else:
                        accuracy = "N/A"
                else:
                    accuracy = "N/A"

                # 收集必要信息
                ai_input = {
                    "数据概况": {
                        "样本量": len(df),
                        "特征数": len(config['feature']),
                        "问题类型": "分类" if st.session_state.is_classification else "回归",
                        "类别数": st.session_state.num_classes if st.session_state.is_classification else "N/A"
                    },
                    "当前配置": {
                        "隐藏层": config.get('hidden_layers'),
                        "神经元": config.get('neurons'),
                        "激活函数": config.get('activation'),
                        "学习率": config.get('learning_rate'),
                        "批次大小": config.get('batch_size'),
                        "训练轮次": config.get('epochs'),
                        "测试集比例": test_size_percent
                    },
                    "当前效果": {
                        "准确率": accuracy,
                        "损失": test_loss if 'test_loss' in locals() else "N/A"
                    }
                }

                # 调用AI
                with st.spinner("AI正在分析模型表现..."):
                    advice = get_ai_advice(ai_input)

                if advice:
                    st.session_state.ai_advice = advice
                    st.session_state.ai_advice_generated = True
                    st.rerun()

            # ===== 提示信息放在按钮下方 =====
            if 'ai_advice_generated' in st.session_state and st.session_state.ai_advice_generated:
                st.success("✅ AI建议已生成！请查看左侧边栏最下方")
                st.info("💡 提示：请根据建议在步骤3中手动调整参数，多次尝试以达到最佳效果")

        st.markdown("---")
        # ========== 预测功能 ==========
        st.markdown("### 🔮 使用模型预测")
        # 新增：批量预测数据格式要求提示（醒目且不干扰操作）
        with st.container(border=True):
            st.markdown("#### 📌 批量预测文件格式要求")
            st.markdown("""
            为避免预测报错，请确保上传的CSV/Excel文件满足以下条件：
            1. **特征列完整**：必须包含模型训练时的所有特征列（列名需与训练数据完全一致，大小写/空格敏感）；
            2. **无缺失值**：特征列不能有空白单元格（NaN），需提前删除含缺失值的行或填充；
            3. **数据类型正确**：
               (1) 数值型特征（如面积、年龄）：仅保留纯数字（int/float），不含文本、特殊符号（如「120㎡」「二十岁」）；
               (2)类别型特征（如性别、学历）：填写原始文本值（如「男/女」「本科/硕士」），**不要填写编码后的数字**；
            4. **无需标签列**：预测文件仅需特征列，无需包含训练时的标签列（模型会自动生成预测结果）；
            5. **编码一致**：类别特征的取值需与训练数据一致（如训练时「性别」只有「男/女」，预测时不能出现「未知」）。
            
            ❗ 若预测报错，请按上述要求检查文件后重试（常见问题：列名不一致、含非数值字符、存在缺失值）。
            """)
        # 初始化预测状态
        if 'show_prediction_results' not in st.session_state:
            st.session_state.show_prediction_results = False
        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None
        if 'manual_pred_result' not in st.session_state:
            st.session_state.manual_pred_result = None
        if 'manual_input_counter' not in st.session_state:
            st.session_state.manual_input_counter = '0'
        if 'file_uploader_key' not in st.session_state:
            st.session_state.file_uploader_key = 'pred_upload_file'
        # 选择输入方式
        input_method = st.radio("选择输入方式", ["手动输入", "上传文件"], horizontal=True, key="input_method_radio")
        if input_method == "手动输入":
            st.write("请输入特征值：")
            # 只有当不显示预测结果时才显示输入表单
            if not st.session_state.show_prediction_results:
                input_values = []
                input_display = []
                # 获取计数器，用于刷新输入组件
                counter = st.session_state.get('manual_input_counter', '0')
                # 创建输入控件
                for i, feat in enumerate(config['feature']):
                    if feat in st.session_state.label_encoders:
                        le_feat = st.session_state.label_encoders[feat]
                        opts = le_feat.classes_.tolist()
                        default_index = 0
                        sel_text = st.selectbox(f"选择 {feat}", opts, key=f"pred_feat_{i}_{counter}",
                                                index=default_index)
                        input_values.append(float(le_feat.transform([sel_text])[0]))
                        input_display.append(sel_text)
                    else:
                        val = st.number_input(f"输入 {feat}", value=0.0, step=0.1, key=f"pred_num_{i}_{counter}",
                                              format="%f")
                        input_values.append(val)
                        input_display.append(f"{val}")
                # 预测按钮
                if st.button("🚀 开始预测", type="primary", use_container_width=True, key="manual_predict_btn"):
                    try:
                        # 转换输入
                        input_arr = np.array(input_values).reshape(1, -1).astype(float)
                        # 标准化
                        input_scaled = scaler.transform(input_arr)
                        # 预测
                        pred = model.predict(input_scaled, verbose=0)
                        # 保存预测结果到session state
                        st.session_state.manual_pred_result = {
                            'input_display': input_display,
                            'features': config['feature'],
                            'prediction': pred,
                            'timestamp': time.time()
                        }
                        st.session_state.show_prediction_results = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 预测出错：{str(e)}")
                        st.exception(e)
            # 显示预测结果
            if st.session_state.get('show_prediction_results', False) and 'manual_pred_result' in st.session_state:
                result = st.session_state.manual_pred_result
                st.markdown("---")
                st.markdown("### 📝 输入特征：")
                for i, feat in enumerate(result['features']):
                    st.write(f"- {feat}：{result['input_display'][i]}")
                st.markdown("### 🎯 预测结果")
                pred = result['prediction']
                if st.session_state.is_classification:
                    if st.session_state.num_classes == 2:
                        prob = pred[0][0]
                        label = 1 if prob > 0.5 else 0
                        res_text = st.session_state.label_encoder.inverse_transform([label])[0]
                        st.success(f"**预测类别：{res_text}**")
                        st.progress(float(prob))
                        st.write(f"置信度：{prob:.2%}")
                    else:
                        cls_idx = np.argmax(pred[0])
                        res_text = st.session_state.label_encoder.inverse_transform([cls_idx])[0]
                        st.success(f"**预测类别：{res_text}**")
                        st.write(f"各类别概率：")
                        for i, prob in enumerate(pred[0]):
                            class_name = st.session_state.label_encoder.inverse_transform([i])[0]
                            st.write(f"- {class_name}: {prob:.2%}")
                else:
                    st.success(f"**预测值：{pred[0][0]:.4f}**")
        else:  # 上传文件模式
            st.write("上传文件进行批量预测：")
            uploader_key = st.session_state.get('file_uploader_key', 'pred_upload_file')
            pred_file = st.file_uploader("选择预测文件", ['csv', 'xlsx'], key=uploader_key)
            # 如果已经上传了文件且不显示结果
            if pred_file and not st.session_state.show_prediction_results:
                try:
                    if pred_file.name.endswith('.csv'):
                        df_pred = pd.read_csv(pred_file, encoding='gbk')
                    else:
                        df_pred = pd.read_excel(pred_file)
                    # 显示数据预览
                    with st.expander("📊 查看上传数据预览"):
                        st.dataframe(df_pred.head(5), use_container_width=True)
                        st.write(f"数据形状：{df_pred.shape[0]}行 × {df_pred.shape[1]}列")
                    missing_feats = [f for f in config['feature'] if f not in df_pred.columns]
                    if missing_feats:
                        st.error(f"缺失特征列：{missing_feats}")
                    else:
                        if st.button("开始批量预测", type="primary", key="batch_predict_btn"):
                            with st.spinner("正在预测中..."):
                                X_pred = df_pred[config['feature']].values
                                X_pred_scaled = scaler.transform(X_pred)
                                preds = model.predict(X_pred_scaled, verbose=0)
                                # 处理预测结果
                                result_df = df_pred.copy()
                                if st.session_state.is_classification:
                                    if st.session_state.num_classes == 2:
                                        result_df['预测概率'] = preds.flatten()
                                        result_df['预测结果'] = result_df['预测概率'].apply(
                                            lambda x: st.session_state.label_encoder.inverse_transform([1])[0]
                                            if x > 0.5 else st.session_state.label_encoder.inverse_transform([0])[0]
                                        )
                                    else:
                                        result_df['预测结果'] = [
                                            st.session_state.label_encoder.inverse_transform([np.argmax(p)])[0]
                                            for p in preds
                                        ]
                                else:
                                    result_df['预测值'] = preds.flatten()
                                # 保存结果到session state
                                st.session_state.prediction_result = result_df
                                st.session_state.show_prediction_results = True
                                st.rerun()
                except Exception as e:
                    st.error(f"❌ 预测出错：{str(e)}")
                    st.exception(e)
            # 显示预测结果
            if st.session_state.get('show_prediction_results', False) and 'prediction_result' in st.session_state:
                result_df = st.session_state.prediction_result
                st.success(f"✅ 完成 {len(result_df)} 条数据预测")
                st.dataframe(result_df, use_container_width=True)
                # 下载预测结果
                csv_data = result_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 下载预测结果",
                    data=csv_data,
                    file_name="预测结果.csv",
                    mime="text/csv",
                    key="download_pred_results"
                )
        st.markdown("---")
        # ========== 操作按钮（保留4个主要功能） ==========
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("← 返回配置", use_container_width=True, key="back_to_config"):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.button("🔄 重新训练", use_container_width=True, key="retrain_model"):
                keys_to_remove = ['model', 'scaler', 'history', 'X_test', 'y_test',
                                  'label_encoder', 'is_classification', 'num_classes',
                                  'show_prediction_results', 'prediction_result',
                                  'manual_pred_result', 'uploaded_model', 'uploaded_scaler', 'uploaded_config']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.step = 3
                st.rerun()
        with col3:
            if st.button("🔮 新预测（点击两下）", use_container_width=True, key="new_prediction"):
                st.session_state.show_prediction_results = False
                st.session_state.prediction_result = None
                st.session_state.manual_pred_result = None
                st.session_state.manual_input_counter = str(np.random.rand())
                st.session_state.file_uploader_key = str(np.random.rand())
        with col4:
            if st.button("🏁 全新开始", use_container_width=True, key="fresh_start"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    # ========== 步骤4专属异常处理 ==========
    except Exception as e:
        st.error(f"数据处理出错：{str(e)}")
        st.exception(e)
        if st.button("返回数据清洗", use_container_width=True):
            st.session_state.step = 2
            st.rerun()



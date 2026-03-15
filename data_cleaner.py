import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import io
import warnings

warnings.filterwarnings('ignore')


# ========== GMM聚类函数（用于回归问题的样本平衡性处理） ==========
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


# ========== 数据质量评分函数 ==========
def data_quality_score(df, label_col, label_encoders):
    """评估数据质量并评分"""
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
            # 先判断是否是分类变量
            if col in label_encoders:
                # 分类变量：直接用IQR
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((data < lower_bound) | (data > upper_bound)).sum()
                method = "IQR (分类变量)"
                is_normal = False
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
                    mean_val = data.mean()
                    std_val = data.std()
                    z_scores = np.abs((data - mean_val) / std_val)
                    outliers = (z_scores > 2.5).sum()
                    method = "Z-score"
                else:
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

    # 3. 样本平衡性检测（25分）
    is_classification = label_col in label_encoders
    if is_classification:
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
        y = df[label_col].values
        labels, gmm = gmm_clustering_for_regression(y, max_components=8)
        cluster_counts = pd.Series(labels).value_counts()
        n_clusters = len(cluster_counts)
        detection_results['balance'] = {
            'is_classification': False,
            'gmm_clusters': n_clusters,
            'cluster_counts': cluster_counts.to_dict(),
            'gmm_model': gmm
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


# ========== 处理回归样本不平衡的函数 ==========
def balance_regression_by_gmm(df, label_col, method='smote'):
    """
    使用GMM聚类结果为回归问题生成合成样本
    method: 'smote', 'adasyn', 'undersample', 'none'
    返回：平衡后的DataFrame, 聚类标签, GMM模型
    """
    y = df[label_col].values
    X = df.iloc[:, :-1].copy()
    # 使用GMM聚类
    labels, gmm = gmm_clustering_for_regression(y, max_components=8)

    if gmm is None or len(np.unique(labels)) <= 1:
        return df, labels, gmm  # ✅ 返回3个值

    # 基于聚类标签进行平衡处理
    cluster_counts = pd.Series(labels).value_counts()
    if method == 'none' or len(cluster_counts) <= 1:
        return df, labels, gmm  #
    # 准备数据
    X_balanced = X.copy()
    y_balanced = pd.Series(y.copy(), name=label_col)
    labels_balanced = labels.copy()
    if method == 'smote' or method == 'adasyn':
        # 对每个聚类内部进行过采样
        max_cluster_size = cluster_counts.max()
        for cluster_id in cluster_counts.index:
            current_size = cluster_counts[cluster_id]
            if current_size < max_cluster_size:
                cluster_mask = labels == cluster_id
                X_cluster = X[cluster_mask]
                y_cluster = y[cluster_mask]
                n_synthesize = max_cluster_size - current_size
                for _ in range(n_synthesize):
                    idx = np.random.randint(0, len(X_cluster))
                    X_sample = X_cluster.iloc[idx].copy()
                    y_sample = y_cluster[idx]
                    noise_scale = 0.01 * X_sample.std() if X_sample.std() > 0 else 0.01
                    X_noise = np.random.normal(0, noise_scale, size=len(X_sample))
                    X_sample = X_sample + X_noise
                    X_balanced = pd.concat([X_balanced, pd.DataFrame([X_sample])], ignore_index=True)
                    y_balanced = pd.concat([y_balanced, pd.Series([y_sample])], ignore_index=True)
                    labels_balanced = np.append(labels_balanced, cluster_id)
        balanced_df = X_balanced.copy()
        balanced_df[label_col] = y_balanced.values
    elif method == 'undersample':
        min_cluster_size = cluster_counts.min()
        indices_to_keep = []
        for cluster_id in cluster_counts.index:
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) > min_cluster_size:
                selected = np.random.choice(cluster_indices, min_cluster_size, replace=False)
                indices_to_keep.extend(selected)
            else:
                indices_to_keep.extend(cluster_indices)
        indices_to_keep = sorted(indices_to_keep)
        balanced_df = df.iloc[indices_to_keep].reset_index(drop=True)
        labels_balanced = labels[indices_to_keep]
    else:
        balanced_df = df.copy()
    return balanced_df, labels_balanced, gmm


# ========== 数据清洗主函数 ==========
def run_data_cleaning():
    """数据清洗主函数"""
    st.subheader("🧹 数据清洗")

    df = st.session_state.df.copy()
    label_col = df.columns[-1]
    label_encoders = st.session_state.label_encoders

    # ===== 首次检测 =====
    if 'detection_results' not in st.session_state or st.session_state.detection_results == {}:
        with st.spinner("正在评估数据质量..."):
            scores, total_score, max_total, warnings, detection_results = data_quality_score(
                df, label_col, label_encoders
            )
            st.session_state.scores = scores
            st.session_state.total_score = total_score
            st.session_state.max_total = max_total
            st.session_state.warnings = warnings
            st.session_state.detection_results = detection_results
            st.session_state.original_df = df.copy()
            st.session_state.processed_df = df.copy()
            if st.session_state.temp_df is None:
                st.session_state.temp_df = df.copy()

    # ===== 显示评分卡 =====
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

    # ===== 第一步：缺失值处理 =====
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
                    if col in label_encoders:
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
        if st.session_state.temp_df is None:
            st.session_state.temp_df = df

    # ===== 第二步：异常值处理 =====
    st.markdown("---")
    st.markdown("#### 步骤2：选择异常值处理方式")

    current_df = st.session_state.temp_df.copy()
    if len(current_df) == 0:
        st.error("❌ 当前数据为空，请先处理缺失值")
        st.stop()

    # 异常值检测
    if 'outlier_detection_results' not in st.session_state:
        outlier_detection_results = {
            'outlier_info': [],
            'outlier_mask': pd.Series(False, index=current_df.index),
            'col_methods': {}
        }

        for col in current_df.columns[:-1]:
            data = current_df[col].dropna()
            if len(data) <= 3:
                continue

            if col in label_encoders:
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
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    col_outlier_mask = (current_df[col] < lower) | (current_df[col] > upper)
                    outliers = col_outlier_mask.sum()
                    method = "IQR"
                    distribution = "非正态分布"

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

    outlier_info = st.session_state.outlier_detection_results['outlier_info']
    outlier_mask = st.session_state.outlier_detection_results['outlier_mask']
    col_methods = st.session_state.outlier_detection_results['col_methods']

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
                    data = df_median[col]
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    median = data.median()
                    df_median.loc[(data < lower) | (data > upper), col] = median
                else:
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

    # ===== 第三步：样本平衡处理 =====
    st.markdown("---")
    st.markdown("#### 步骤3：选择样本平衡处理方式")

    current_df = st.session_state.temp_df.copy()
    is_classification = label_col in label_encoders

    if is_classification:
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
                    st.session_state.temp_df = current_df
                    st.session_state.user_choices['balance_confirmed'] = True
                    st.success("✅ 已选择Class Weight，将在训练时自动应用")
                    st.rerun()

            with col3:
                st.markdown("**路径C：SMOTENC**")
                st.caption("智能处理混合数据")
                try:
                    # 创建副本，避免修改原始数据
                    X = current_df.iloc[:, :-1].copy()
                    y = current_df.iloc[:, -1].copy()

                    # 确保分类特征是整数类型
                    for col in label_encoders:
                        if col in X.columns:
                            X[col] = X[col].astype('int64')

                    categorical_features = []
                    for i, col in enumerate(X.columns):
                        if col in label_encoders and col != label_col:
                            categorical_features.append(i)

                    if categorical_features:
                        from imblearn.over_sampling import SMOTENC
                        smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)
                        X_res, y_res = smote_nc.fit_resample(X, y)
                    else:
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

    else:  # 回归问题
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
                    df_balanced, labels, gmm = balance_regression_by_gmm(current_df, label_col, method='smote')
                    st.session_state.gmm_model = gmm  # 👈 在这里保存GMM模型
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
                    df_balanced, labels, gmm = balance_regression_by_gmm(current_df, label_col, method='undersample')
                    st.session_state.gmm_model = gmm  # 👈 在这里保存GMM模型
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

    # ===== 第四步：特征筛选 =====
    if st.session_state.user_choices['missing_confirmed'] and st.session_state.user_choices['outlier_confirmed'] and \
            st.session_state.user_choices['balance_confirmed']:
        st.markdown("---")
        st.markdown("#### 步骤4：特征重要性评估与特征筛选")

        final_df = st.session_state.temp_df.copy()
        if len(final_df) == 0:
            st.error("❌ 当前数据为空，请返回重新选择")
            st.stop()

        try:
            X = final_df.iloc[:, :-1]
            y = final_df.iloc[:, -1]
            if len(final_df) >= 30 and not X.isnull().any().any():
                if label_col in label_encoders:
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

                keep_features = st.multiselect(
                    "选择要保留的特征（默认全选）",
                    options=X.columns.tolist(),
                    default=X.columns.tolist()
                )

                if st.button("应用特征筛选", key="apply_features", use_container_width=True) and not \
                st.session_state.user_choices['features_confirmed']:
                    if not keep_features:
                        st.warning("⚠️ 未选择任何特征列！模型训练需要至少一个特征作为输入，请勾选至少一个特征列后重试～")
                    else:
                        st.session_state.user_choices['features'] = keep_features
                        final_selected_df = final_df[keep_features + [label_col]]
                        st.session_state.processed_df = final_selected_df

                        with st.spinner("正在重新评估最终数据质量..."):
                            # 重新计算特征重要性，传入筛选后的特征
                            scores, total_score, max_total, warnings, detection_results = data_quality_score(
                                final_selected_df, label_col, label_encoders
                            )

                            # 更新detection_results中的特征列信息
                            detection_results['feature']['feature_cols'] = keep_features

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

            # 删除所有检测相关的session_state
            detection_keys = ['outlier_detection_results', 'gmm_model', 'cluster_labels']
            for key in detection_keys:
                if key in st.session_state:
                    del st.session_state[key]

            # 重置临时数据
            st.session_state.temp_df = st.session_state.df.copy()
            st.session_state.processed_df = st.session_state.df.copy()

            # 重新计算初始评分
            with st.spinner("正在恢复初始数据质量评分..."):
                scores, total_score, max_total, warnings, detection_results = data_quality_score(
                    st.session_state.df, label_col, label_encoders
                )
                st.session_state.scores = scores
                st.session_state.total_score = total_score
                st.session_state.max_total = max_total
                st.session_state.warnings = warnings
                st.session_state.detection_results = detection_results
            st.success("✅ 已恢复到初始清洗状态！")
            st.rerun()

    # ===== 数据分布可视化 =====
    st.markdown("### 📈 数据分布可视化")
    viz_df = st.session_state.temp_df.copy() if st.session_state.temp_df is not None else st.session_state.df.copy()
    label_col = viz_df.columns[-1]

    if len(viz_df.columns) > 0:
        cate_cols = list(label_encoders.keys())
        num_cols = [col for col in viz_df.columns if col not in cate_cols and viz_df[col].dtype in ['int64', 'float64']]
        all_cols = viz_df.columns.tolist()
        selected_col = st.selectbox("选择要可视化的列", all_cols, key="viz_all_col")

        is_label_col = (selected_col == label_col)
        is_cate_col = (selected_col in cate_cols)
        is_num_col = (selected_col in num_cols)
        label_is_cate = (label_col in cate_cols)
        label_is_num = (label_col in num_cols)
        col_data = viz_df[selected_col].dropna()

        # 图表生成逻辑（与原代码相同）
        if is_cate_col or (is_label_col and label_is_cate):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
            le = label_encoders[selected_col]
            col_data_text = col_data.map(lambda x: le.classes_[int(x)] if int(x) < len(le.classes_) else f'未知({x})')
            value_counts = col_data_text.value_counts().sort_index()
            ax1.bar(range(len(value_counts)), value_counts.values, color='#1f77b4')
            ax1.set_xticks(range(len(value_counts)))
            ax1.set_xticklabels(value_counts.index, rotation=30, ha='right', fontsize=9)
            ax1.set_title(f'{selected_col} - 柱状图', fontsize=12)
            ax1.set_xlabel('类别', fontsize=10)
            ax1.set_ylabel('频数', fontsize=10)
            for i, count in enumerate(value_counts.values):
                ax1.text(i, count + 0.1, f'{int(count)}', ha='center', va='bottom', fontsize=8)

            ax2.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'{selected_col} - 扇形图', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)

            # 图表下载
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button(label="📥 下载图表（PNG）", data=buf, file_name=f"{selected_col}_可视化图表.png",
                               mime="image/png")

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
                plt.tight_layout()
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
                unique_vals = col_data.nunique()
                bins = min(20, unique_vals)
                ax.hist(col_data, bins=bins, edgecolor='black', alpha=0.7, color='#1f77b4')
                ax.set_title(f'{selected_col} - 直方图', fontsize=12)
                ax.set_xlabel('值', fontsize=10)
                ax.set_ylabel('频数', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)

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

    # ===== 清洗后数据预览+下载 =====
    st.markdown("### 📋 清洗后数据预览")
    st.dataframe(viz_df.head(10), use_container_width=True)
    st.write(f"当前数据形状: **{viz_df.shape[0]}行 × {viz_df.shape[1]}列**")

    st.markdown("#### 📥 清洗后数据下载")
    col_download_csv, col_download_excel = st.columns(2)
    with col_download_csv:
        csv_data = viz_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(label="下载CSV格式", data=csv_data,
                           file_name=f"清洗后数据_{viz_df.shape[0]}行{viz_df.shape[1]}列.csv", mime="text/csv",
                           use_container_width=True)

    with col_download_excel:
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
            viz_df.to_excel(writer, sheet_name='清洗后数据', index=False)
        excel_buf.seek(0)
        st.download_button(label="下载Excel格式", data=excel_buf,
                           file_name=f"清洗后数据_{viz_df.shape[0]}行{viz_df.shape[1]}列.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

    # ===== 进入下一步按钮 =====
    if st.session_state.user_choices['features_confirmed']:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("➡️ 进入模型训练", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import tempfile
import os
import joblib
import warnings
from openai import OpenAI
import re
warnings.filterwarnings('ignore')


# ========== AI调用函数（为LightGBM定制） ==========
def get_lgb_ai_advice(data):
    """调用DeepSeek API获取LightGBM模型评价和优化建议"""
    try:
        from openai import OpenAI
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
- 提升迭代次数: {data['当前配置']['n_estimators']}
- 学习率: {data['当前配置']['learning_rate']}
- 最大深度: {data['当前配置']['max_depth']}
- 叶子节点数: {data['当前配置']['num_leaves']}
- 最小叶子样本数: {data['当前配置']['min_child_samples']}
- 测试集比例: {data['当前配置']['test_size']}%

当前效果：
- 准确率: {data['当前效果']['准确率']}

【历史建议回顾】
{st.session_state.ai_history[-1] if st.session_state.ai_history else '这是第一次咨询，还没有历史建议。'}

【重要要求】
1. LightGBM参数较多，根据样本量给出建议：
   - 小样本 (<1000)：减少迭代次数，降低复杂度
   - 中等样本 (1000-5000)：默认参数即可
   - 大样本 (>5000)：可以增加迭代次数和叶子节点数
2. 学习率通常与迭代次数配合：learning_rate 越小，n_estimators 需要越大
3. 如果过拟合（训练集准确率高，测试集低）：
   - 减小 max_depth 或 num_leaves
   - 增大 min_child_samples
   - 增大 learning_rate 同时减小 n_estimators

如果是第二次或更多次咨询，请参考之前给的建议，告诉用户这次调整后的效果如何。

请以JSON格式返回，只返回JSON：
{{
    "评价": "结合基准和样本量评价当前模型",
    "优化建议": {{
        "n_estimators": 建议值,
        "learning_rate": 建议值,
        "max_depth": 建议值,
        "num_leaves": 建议值,
        "min_child_samples": 建议值,
        "subsample": 建议值（0.5-1.0之间）,
        "colsample_bytree": 建议值（0.5-1.0之间）,
        "reg_alpha": 建议值,
        "reg_lambda": 建议值,
        "min_split_gain": 建议值,
        "test_size": 建议值（整数百分比）
        }},
    "注意：必须返回所有参数，即使保持默认也要写出来"
    "预期效果": "说明优化后可能达到的效果"
}}
"""

        # 添加用户消息到历史
        st.session_state.ai_messages.append({"role": "user", "content": current_prompt})

        # 调用API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=st.session_state.ai_messages,
            temperature=0.3,
            max_tokens=1000
        )

        advice_text = response.choices[0].message.content
        st.session_state.ai_messages.append({"role": "assistant", "content": advice_text})

        # 解析JSON
        import json
        import re

        json_match = re.search(r'(\{.*\})', advice_text, re.DOTALL)
        if json_match:
            advice_json = json.loads(json_match.group(1))
            history_entry = f"第{len(st.session_state.ai_history) + 1}次：准确率{data['当前效果']['准确率']} → {advice_json['评价']}"
            st.session_state.ai_history.append(history_entry)
            return advice_json
        else:
            return {
                "评价": "模型表现不理想，建议调整参数",
                "优化建议": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": -1,
                    "num_leaves": 31,
                    "min_child_samples": 20,
                    "test_size": 20
                },
                "预期效果": "优化后准确率预计可提升10-20%"
            }

    except Exception as e:
        print(f"AI调用失败：{str(e)}")
        return {
            "评价": f"AI服务暂时不可用，使用默认建议",
            "优化建议": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": -1,
                "num_leaves": 31,
                "min_child_samples": 20,
                "test_size": 20
            },
            "预期效果": "优化后准确率预计可提升10-20%"
        }


# ========== LightGBM训练函数 ==========
def train_lgb():
    """LightGBM训练主函数"""
    st.subheader("⚡ LightGBM训练")

    df = st.session_state.processed_df
    target = df.columns[-1]
    feature = df.columns[:-1].tolist()
    label_encoders = st.session_state.label_encoders
    is_classification = target in label_encoders

    if is_classification:
        le = label_encoders[target]
        num_classes = len(le.classes_)
    else:
        num_classes = None

    # ===== 检查是否有AI建议的参数 =====
    if 'ai_suggested_params' in st.session_state:
        suggested = st.session_state.ai_suggested_params
        st.success("✨ 已应用AI建议的参数，你可以直接点击训练")

        # 读取建议值，如果没有则用默认值
        default_n_estimators = suggested.get('n_estimators', 100)
        default_learning_rate = suggested.get('learning_rate', 0.1)
        default_max_depth = suggested.get('max_depth', -1)
        default_num_leaves = suggested.get('num_leaves', 31)
        default_min_child_samples = suggested.get('min_child_samples', 20)
        default_subsample = suggested.get('subsample', 1.0)
        default_colsample = suggested.get('colsample_bytree', 1.0)
        default_reg_alpha = suggested.get('reg_alpha', 0.0)
        default_reg_lambda = suggested.get('reg_lambda', 0.0)
        default_min_split_gain = suggested.get('min_split_gain', 0.0)
        default_test_size = suggested.get('test_size', 20)

    else:
        default_n_estimators = 100
        default_learning_rate = 0.1
        default_max_depth = -1
        default_num_leaves = 31
        default_min_child_samples = 20
        default_subsample = 1.0
        default_colsample = 1.0
        default_reg_alpha = 0.0
        default_reg_lambda = 0.0
        default_min_split_gain = 0.0
        default_test_size = 20
    # 显示数据信息
    st.markdown("### 1. 数据识别")
    st.info(f"系统识别到：{len(feature)}个特征列，1个标签列")
    st.write(f"**标签列:** {target}")
    st.write(f"**问题类型:** {'分类' if is_classification else '回归'}")
    if is_classification:
        st.write(f"**类别数:** {num_classes}")

    n_samples = len(df)
    if n_samples > 2000:
        st.success(f"✅ 当前样本量 {n_samples}，LightGBM能发挥优势")
    else:
        st.info(f"ℹ️ 当前样本量 {n_samples}，LightGBM可能不如随机森林稳定")

    # 参数配置
    st.markdown("### 2. LightGBM参数配置")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**核心参数**")
        n_estimators = st.number_input("提升迭代次数",
                                       min_value=10, max_value=1000,
                                       value=default_n_estimators,  # 使用AI建议的值
                                       step=10, key="lgb_n_estimators")

        learning_rate = st.number_input("学习率",
                                        min_value=0.01, max_value=1.0,
                                        value=default_learning_rate,  # 使用AI建议的值
                                        step=0.05, key="lgb_learning_rate")

        max_depth = st.number_input("最大深度",
                                    min_value=-1, max_value=50,
                                    value=default_max_depth,  # 使用AI建议的值
                                    step=1, key="lgb_max_depth")

    with col2:
        st.write("**叶子参数**")
        num_leaves = st.number_input("叶子节点数",
                                     min_value=2, max_value=255,
                                     value=default_num_leaves,  # 使用AI建议的值
                                     step=5, key="lgb_num_leaves")

        min_child_samples = st.number_input("最小叶子样本数",
                                            min_value=1, max_value=100,
                                            value=default_min_child_samples,  # 使用AI建议的值
                                            step=5, key="lgb_min_child_samples")

        subsample = st.number_input("样本采样比例",
                                    min_value=0.1, max_value=1.0,
                                    value=default_subsample,  # 使用AI建议的值
                                    step=0.1, key="lgb_subsample")

    # 高级参数（折叠）
    with st.expander("高级参数"):
        col1, col2 = st.columns(2)
        with col1:
            colsample_bytree = st.number_input("特征采样比例",
                                               min_value=0.1, max_value=1.0,
                                               value=default_colsample,  # 使用AI建议的值
                                               step=0.1, key="lgb_colsample")
            reg_alpha = st.number_input("L1正则化",
                                        min_value=0.0, max_value=10.0,
                                        value=default_reg_alpha,  # 使用AI建议的值
                                        step=0.1, key="lgb_reg_alpha")
        with col2:
            reg_lambda = st.number_input("L2正则化",
                                         min_value=0.0, max_value=10.0,
                                         value=default_reg_lambda,  # 使用AI建议的值
                                         step=0.1, key="lgb_reg_lambda")
            min_split_gain = st.number_input("最小分裂增益",
                                             min_value=0.0, max_value=1.0,
                                             value=default_min_split_gain,  # 使用AI建议的值
                                             step=0.1, key="lgb_min_split_gain")

    # 数据划分
    st.markdown("### 3. 数据划分")
    test_size = st.slider("测试集比例 (%)", 10, 40,
                          value=int(default_test_size),
                          key="lgb_test_size")

    X = df[feature].values
    y = df[target].values

    # 分类问题需要处理标签
    if is_classification and num_classes > 2:
        # LightGBM可以直接处理多分类，不需要one-hot
        pass

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42,
        stratify=y if is_classification else None
    )

    # 可选：标准化
    use_scaler = st.checkbox("使用标准化", value=False,
                             help="树模型对尺度不敏感，通常不需要标准化")

    # 显示样本数
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总样本数", len(df))
    with col2:
        st.metric("训练样本", len(X_train))
    with col3:
        st.metric("测试样本", len(X_test))

    st.markdown("---")

    # 训练按钮
    if st.button("🚀 开始训练LightGBM", type="primary", use_container_width=True, key="lgb_train_btn"):

        # 保存配置
        st.session_state.lgb_config = {
            'target': target,
            'feature': feature,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth if max_depth != -1 else -1,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'min_split_gain': min_split_gain,
            'test_size': test_size / 100,
            'use_scaler': use_scaler
        }
        st.session_state.is_classification = is_classification
        st.session_state.num_classes = num_classes
        st.session_state.label_encoder = le if is_classification else None

        # 数据预处理
        if use_scaler:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.session_state.lgb_scaler = scaler
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # ✅ 添加 Class Weight 处理
        class_weight = None
        if is_classification and st.session_state.user_choices.get('balance') == 'class_weight':
            try:
                # 计算样本权重
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.unique(y_train)
                weights = compute_class_weight('balanced', classes=classes, y=y_train)
                class_weight = dict(zip(classes, weights))
                st.info(f"📊 应用Class Weight：{class_weight}")
            except Exception as e:
                st.warning(f"Class Weight计算失败：{str(e)}")

        # 创建模型
        with st.spinner("模型训练中..."):
            start_time = time.time()

            # 设置目标函数
            if is_classification:
                if num_classes == 2:
                    objective = 'binary'
                    metric = 'binary_logloss'
                else:
                    objective = 'multiclass'
                    metric = 'multi_logloss'
            else:
                objective = 'regression'
                metric = 'l2'

            model = lgb.LGBMClassifier(
                n_estimators=int(n_estimators),
                learning_rate=float(learning_rate),
                max_depth=int(max_depth) if max_depth != -1 else -1,
                num_leaves=int(num_leaves),
                min_child_samples=int(min_child_samples),
                subsample=float(subsample),
                colsample_bytree=float(colsample_bytree),
                reg_alpha=float(reg_alpha),
                reg_lambda=float(reg_lambda),
                min_split_gain=float(min_split_gain),
                objective=objective,
                metric=metric,
                class_weight=class_weight,  # ✅ 添加 class_weight 参数
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ) if is_classification else lgb.LGBMRegressor(
                n_estimators=int(n_estimators),
                learning_rate=float(learning_rate),
                max_depth=int(max_depth) if max_depth != -1 else -1,
                num_leaves=int(num_leaves),
                min_child_samples=int(min_child_samples),
                subsample=float(subsample),
                colsample_bytree=float(colsample_bytree),
                reg_alpha=float(reg_alpha),
                reg_lambda=float(reg_lambda),
                min_split_gain=float(min_split_gain),
                objective=objective,
                metric=metric,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

            # 训练
            model.fit(X_train_scaled, y_train,
                      eval_set=[(X_test_scaled, y_test)],
                      eval_metric=metric if is_classification else 'l2',
                      callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])

            train_time = time.time() - start_time

            # 预测
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # 评估
            if is_classification:
                train_score = accuracy_score(y_train, y_pred_train)
                test_score = accuracy_score(y_test, y_pred_test)
                metric_name = "准确率"
            else:
                train_score = mean_absolute_error(y_train, y_pred_train)
                test_score = mean_absolute_error(y_test, y_pred_test)
                metric_name = "MAE"

            # 特征重要性
            feature_importance = dict(zip(feature, model.feature_importances_))

            # 修复2: 保存最佳迭代次数
            best_iteration = None
            if hasattr(model, 'booster_') and hasattr(model.booster_, 'best_iteration'):
                best_iteration = model.booster_.best_iteration
            elif hasattr(model, 'best_iteration_'):
                best_iteration = model.best_iteration_
            else:
                best_iteration = n_estimators

            # 保存结果
            st.session_state.lgb_model = model
            st.session_state.lgb_train_score = train_score
            st.session_state.lgb_test_score = test_score
            st.session_state.lgb_train_time = train_time
            st.session_state.lgb_feature_importance = feature_importance
            st.session_state.lgb_X_test = X_test_scaled
            st.session_state.lgb_y_test = y_test
            st.session_state.lgb_y_pred_test = y_pred_test
            st.session_state.lgb_best_iteration = best_iteration

            st.success(f"✅ 模型训练完成！ 训练时间: {train_time:.2f}秒 最佳迭代次数: {best_iteration}")
            st.session_state.step = 4
            st.rerun()

    # 返回按钮
    if st.button("← 返回数据清洗", key="lgb_back_clean"):
        st.session_state.step = 2
        st.rerun()


# ========== LightGBM预测函数 ==========
def predict_lgb():
    """LightGBM预测函数"""
    st.subheader("🔮 LightGBM预测")
    if 'lgb_predict_counter' not in st.session_state:
        st.session_state.lgb_predict_counter = 0
    current_counter = st.session_state.lgb_predict_counter

    # 判断模型来源：训练得到的还是上传的
    model = None
    config = None
    feature = None
    scaler = None

    # 情况1：通过训练流程得到的模型（有完整配置）
    if 'lgb_model' in st.session_state and st.session_state.lgb_model is not None:
        model = st.session_state.lgb_model
        if 'lgb_config' in st.session_state and st.session_state.lgb_config is not None:
            config = st.session_state.lgb_config
            feature = config['feature']
            # ✅ 修改：优先使用 lgb_scaler
            scaler = st.session_state.get('lgb_scaler', None)
            st.success("✅ 使用训练好的模型进行预测")
        else:
            st.warning("⚠️ 模型缺少训练配置信息")
            # 尝试从原始数据获取特征名
            if st.session_state.df is not None:
                feature = st.session_state.df.columns[:-1].tolist()
                st.info(f"📌 从数据中获取特征名: {feature}")
            else:
                # 从模型推断特征数
                if hasattr(model, 'n_features_in_'):
                    n_features = model.n_features_in_
                else:
                    n_features = 6
                feature = [f"特征_{i + 1}" for i in range(n_features)]
            scaler = st.session_state.get('lgb_scaler', None)

    # 情况2：通过上传得到的模型
    elif 'uploaded_model' in st.session_state and st.session_state.uploaded_model is not None:
        if st.session_state.get('uploaded_model_type') == 'lgb':
            model = st.session_state.uploaded_model
            st.info("📌 使用上传的LightGBM模型进行预测")

            # ✅ 修改：从上传的模型获取scaler
            scaler = st.session_state.get('lgb_scaler', st.session_state.get('uploaded_scaler', None))

            # 尝试从原始数据获取特征名
            if st.session_state.df is not None:
                feature = st.session_state.df.columns[:-1].tolist()
                st.info(f"📌 从数据中获取特征名: {feature}")
            else:
                # 从模型推断特征数
                if hasattr(model, 'n_features_in_'):
                    n_features = model.n_features_in_
                elif hasattr(model, 'coef_'):
                    n_features = len(model.coef_)
                else:
                    n_features = 6
                feature = [f"特征_{i + 1}" for i in range(n_features)]
        else:
            st.warning("请先训练模型或上传LightGBM模型文件")
            if st.button("返回训练"):
                st.session_state.step = 3
                st.rerun()
            return
    else:
        st.warning("请先训练模型或上传模型文件")
        if st.button("返回训练"):
            st.session_state.step = 3
            st.rerun()
        return


    # 获取必要的 session state 变量
    label_encoders = st.session_state.label_encoders if 'label_encoders' in st.session_state else {}
    label_encoder = st.session_state.get('label_encoder', None)
    is_classification = st.session_state.get('is_classification', False)
    num_classes = st.session_state.get('num_classes', None)

    # 显示配置信息
    with st.expander("📋 当前模型配置", expanded=False):
        st.write(f"- 提升迭代次数: {config.get('n_estimators')}")
        st.write(f"- 学习率: {config.get('learning_rate')}")
        st.write(f"- 最大深度: {config.get('max_depth')}")
        st.write(f"- 叶子节点数: {config.get('num_leaves')}")
        st.write(f"- 最小叶子样本数: {config.get('min_child_samples')}")
        st.write(f"- 测试集比例: {config.get('test_size', 0.2) * 100:.0f}%")
        if 'lgb_best_iteration' in st.session_state:
            st.write(f"- 最佳迭代次数: {st.session_state.lgb_best_iteration}")


    # 训练结果
    if 'lgb_test_score' in st.session_state:
        st.markdown("### 📊 训练结果")

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_name = "准确率" if is_classification else "MAE"
            st.metric(f"训练集{metric_name}", f"{st.session_state.lgb_train_score:.4f}")
        with col2:
            st.metric(f"测试集{metric_name}", f"{st.session_state.lgb_test_score:.4f}")
        with col3:
            st.metric("训练时间", f"{st.session_state.lgb_train_time:.2f}秒")

        # 特征重要性可视化
        if 'lgb_feature_importance' in st.session_state:
            st.markdown("### 📈 特征重要性")

            importance_dict = st.session_state.lgb_feature_importance
            importance_df = pd.DataFrame({
                '特征': list(importance_dict.keys()),
                '重要性': list(importance_dict.values())
            }).sort_values('重要性', ascending=True)

            fig = px.bar(importance_df, x='重要性', y='特征', orientation='h',
                         title='特征重要性排序',
                         color='重要性', color_continuous_scale='Greens')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # 分类报告
        if is_classification and 'lgb_y_test' in st.session_state and 'lgb_y_pred_test' in st.session_state:
            with st.expander("📋 分类详细报告"):
                report = classification_report(
                    st.session_state.lgb_y_test,
                    st.session_state.lgb_y_pred_test,
                    target_names=label_encoder.classes_ if label_encoder else None,
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.2f}"))

    st.markdown("---")

    # AI建议按钮
    col1, col2, col3 = st.columns(3)
    with col3:
        if st.button("🤖 AI评价及建议", use_container_width=True, key="lgb_ai_btn"):
            if not st.session_state.get('ai_enabled', False):
                st.error("❌ AI功能未启用，请检查API Key配置")
                st.stop()

            test_size_percent = int(config.get('test_size', 0.2) * 100)
            accuracy = f"{st.session_state.lgb_test_score:.2%}" if is_classification else "N/A"

            ai_input = {
                "数据概况": {
                    "样本量": len(st.session_state.processed_df),
                    "特征数": len(feature),
                    "问题类型": "分类" if is_classification else "回归",
                    "类别数": num_classes if is_classification else "N/A"
                },
                "当前配置": {
                    "n_estimators": config.get('n_estimators'),
                    "learning_rate": config.get('learning_rate'),
                    "max_depth": config.get('max_depth'),
                    "num_leaves": config.get('num_leaves'),
                    "min_child_samples": config.get('min_child_samples'),
                    "subsample": config.get('subsample', 1.0),
                    "colsample_bytree": config.get('colsample_bytree', 1.0),
                    "reg_alpha": config.get('reg_alpha', 0.0),
                    "reg_lambda": config.get('reg_lambda', 0.0),
                    "min_split_gain": config.get('min_split_gain', 0.0),
                    "test_size": test_size_percent
                },
                "当前效果": {
                    "准确率": accuracy
                }
            }

            with st.spinner("AI正在分析模型表现..."):
                advice = get_lgb_ai_advice(ai_input)

            if advice:
                st.session_state.ai_advice = advice
                st.session_state.ai_advice_generated = True
                st.rerun()

    # AI提示信息
    if st.session_state.get('ai_advice_generated', False):
        st.success("✅ AI建议已生成！请查看左侧边栏下方🤖 AI模型建议，并点击✨ 应用AI建议参数。")
        st.info("💡 提示：请根据建议在步骤3中手动调整参数")

    # 模型保存
    col1, col2 = st.columns(2)
    with col1:
        try:
            # 创建打包数据
            model_package = {
                'model': model,
                'scaler': st.session_state.get('lgb_scaler', None),
                'config': config,
                'model_type': 'lgb',
                'feature_names': feature,
                'num_classes': num_classes if is_classification else None,
                'label_encoder_classes': label_encoder.classes_.tolist() if label_encoder else None,
                'best_iteration': st.session_state.get('lgb_best_iteration')
            }

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                joblib.dump(model_package, tmp_file.name)
                tmp_file_path = tmp_file.name
                with open(tmp_file_path, 'rb') as f:
                    model_bytes = f.read()

            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

            st.download_button(
                label="📥 下载完整模型包 (.pkl)",
                data=model_bytes,
                file_name=f"lgb_model_package_{time.strftime('%Y%m%d_%H%M%S')}.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"❌ 模型保存失败：{str(e)}")

    with col2:
        try:
            config_json = json.dumps(config, indent=2, ensure_ascii=False)
            st.download_button(
                label="📄 下载配置 (.json)",
                data=config_json,
                file_name=f"lgb_config_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"❌ 配置保存失败：{str(e)}")

    st.markdown("---")

    # ===== 预测功能 =====
    st.markdown("### 🔮 使用模型预测")

    # 初始化session state
    if 'lgb_show_prediction' not in st.session_state:
        st.session_state.lgb_show_prediction = False
    if 'lgb_pred_result' not in st.session_state:
        st.session_state.lgb_pred_result = None
    if 'lgb_batch_result' not in st.session_state:
        st.session_state.lgb_batch_result = None
    if 'lgb_manual_counter' not in st.session_state:
        st.session_state.lgb_manual_counter = 0

    # 预测方式选择
    input_method = st.radio(
        "选择输入方式",
        ["手动输入", "上传文件"],
        horizontal=True,
        key=f"lgb_input_method_{current_counter}"
    )

    if input_method == "手动输入":
        st.write("请输入特征值：")

        input_values = []
        input_display = []

        for i, feat in enumerate(feature):
            if feat in label_encoders:
                le = label_encoders[feat]
                original_categories = le.classes_.tolist()

                selected_text = st.selectbox(
                    f"选择 {feat}",
                    options=original_categories,
                    key=f"lgb_pred_feat_{i}_{current_counter}"
                )

                encoded_value = le.transform([selected_text])[0]
                input_values.append(float(encoded_value))
                input_display.append(selected_text)
            else:
                val = st.number_input(
                    f"输入 {feat}",
                    value=0.0,
                    step=0.1,
                    key=f"lgb_pred_num_{i}_{current_counter}",
                    format="%f"
                )
                input_values.append(val)
                input_display.append(f"{val}")

        # 预测按钮用当前计数器
        if st.button("🚀 开始预测", type="primary", use_container_width=True,
                     key=f"lgb_predict_btn_{current_counter}"):
            try:
                input_arr = np.array(input_values).reshape(1, -1)
                # 检查 scaler 是否存在
                if scaler is not None:
                    input_scaled = scaler.transform(input_arr)
                    pred = model.predict(input_scaled)[0]
                    proba = model.predict_proba(input_scaled)[0] if is_classification else None
                else:
                    pred = model.predict(input_arr)[0]
                    proba = model.predict_proba(input_arr)[0] if is_classification else None
                    st.info("ℹ️ 使用原始数据进行预测（无标准化）")


                st.session_state.lgb_pred_result = {
                    'input_display': input_display,
                    'features': feature,
                    'prediction': pred
                }
                st.session_state.lgb_show_prediction = True
                # 增加计数器，下次进入时key会变化
                st.session_state.lgb_predict_counter += 1
                st.rerun()
            except Exception as e:
                st.error(f"❌ 预测出错：{str(e)}")

        # 显示预测结果
        if st.session_state.lgb_show_prediction and st.session_state.lgb_pred_result is not None:
            result = st.session_state.lgb_pred_result
            st.markdown("---")
            st.markdown("### 📝 输入特征：")
            for i, feat in enumerate(result['features']):
                st.write(f"- {feat}：{result['input_display'][i]}")

            st.markdown("### 🎯 预测结果")
            pred = result['prediction']
            proba = result.get('probabilities')

            if is_classification:
                # 确保pred是整数且在范围内
                pred_int = int(pred)
                if 0 <= pred_int < len(label_encoder.classes_):
                    res_text = label_encoder.inverse_transform([pred_int])[0]
                    st.success(f"**预测类别：{res_text}**")
                else:
                    st.success(f"**预测类别索引：{pred_int}**")

                if proba is not None:
                    st.write("各类别概率：")
                    for i, p in enumerate(proba):
                        if i < len(label_encoder.classes_):
                            class_name = label_encoder.inverse_transform([i])[0]
                            st.write(f"- {class_name}: {p:.2%}")
                        else:
                            st.write(f"- 类别{i}: {p:.2%}")
            else:
                st.success(f"**预测值：{pred:.4f}**")



    else:  # 上传文件模式

        st.write("上传文件进行批量预测：")

        uploader_key = f"lgb_pred_file_{st.session_state.get('lgb_batch_counter', 0)}"

        pred_file = st.file_uploader("选择预测文件", ['csv', 'xlsx'], key=uploader_key)

        if pred_file and st.session_state.lgb_batch_result is None:

            try:

                if pred_file.name.endswith('.csv'):

                    try:

                        df_pred = pd.read_csv(pred_file, encoding='utf-8')

                    except:

                        pred_file.seek(0)

                        df_pred = pd.read_csv(pred_file, encoding='gbk')

                else:

                    df_pred = pd.read_excel(pred_file)

                with st.expander("📊 查看上传数据预览"):

                    st.dataframe(df_pred.head(5))

                    st.write(f"数据形状：{df_pred.shape[0]}行 × {df_pred.shape[1]}列")

                missing_feats = [f for f in feature if f not in df_pred.columns]

                if missing_feats:

                    st.error(f"缺失特征列：{missing_feats}")

                else:

                    if st.button("开始批量预测", type="primary", key="lgb_batch_predict"):

                        with st.spinner("正在预测中..."):

                            X_pred = df_pred[feature].values

                            if scaler:
                                X_pred = scaler.transform(X_pred)

                            preds = model.predict(X_pred)

                            result_df = df_pred.copy()

                            # ✅ 通用维度处理

                            if is_classification:

                                if num_classes == 2:  # 二分类

                                    probas = model.predict_proba(X_pred)

                                    result_df['预测概率'] = probas[:, 1]  # 正类的概率

                                    result_df['预测结果'] = preds

                                    if label_encoder:
                                        result_df['预测结果'] = result_df['预测结果'].apply(

                                            lambda x: label_encoder.inverse_transform([int(x)])[0]

                                        )

                                else:  # 多分类

                                    probas = model.predict_proba(X_pred)

                                    result_df['预测结果'] = preds

                                    if label_encoder:
                                        result_df['预测结果'] = result_df['预测结果'].apply(

                                            lambda x: label_encoder.inverse_transform([int(x)])[0]

                                        )

                                    for i in range(probas.shape[1]):

                                        if label_encoder and i < len(label_encoder.classes_):

                                            class_name = label_encoder.inverse_transform([i])[0]

                                        else:

                                            class_name = f'类别{i}'

                                        result_df[f'{class_name}_概率'] = probas[:, i]

                            else:  # 回归

                                result_df['预测值'] = preds

                            st.session_state.lgb_batch_result = result_df

                            st.rerun()

            except Exception as e:

                st.error(f"❌ 预测出错：{str(e)}")

        if st.session_state.lgb_batch_result is not None:
            result_df = st.session_state.lgb_batch_result
            st.success(f"✅ 完成 {len(result_df)} 条数据预测")
            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载预测结果",
                data=csv_data,
                file_name="预测结果.csv",
                mime="text/csv",
                key="lgb_download_results"
            )


    st.markdown("---")

    # 操作按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 重新训练", use_container_width=True, key="lgb_retrain"):
            keys_to_remove = ['lgb_model', 'lgb_config', 'lgb_train_score', 'lgb_test_score',
                              'lgb_feature_importance', 'lgb_X_test', 'lgb_y_test', 'lgb_y_pred_test',
                              'lgb_scaler', 'lgb_best_iteration', 'lgb_batch_result', 'lgb_pred_result']
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("🔮 新预测", use_container_width=True, key="lgb_new_prediction"):
            st.session_state.lgb_show_prediction = False
            st.session_state.lgb_pred_result = None
            st.session_state.lgb_batch_result = None
            st.session_state.lgb_manual_counter += 1
            if 'lgb_batch_counter' in st.session_state:
                st.session_state.lgb_batch_counter += 1
            st.rerun()
    with col3:
        if st.button("🏁 全新开始", use_container_width=True, key="lgb_fresh"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

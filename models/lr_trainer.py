import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
import re
warnings.filterwarnings('ignore')


# ========== AI调用函数（为逻辑回归定制） ==========
def get_lr_ai_advice(data):
    """调用DeepSeek API获取逻辑回归模型评价和优化建议"""
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
            baseline_text = "这是回归问题，逻辑回归主要用于分类"

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
- C值（正则化强度）: {data['当前配置']['C']}
- 正则化类型: {data['当前配置']['penalty']}
- 求解器: {data['当前配置']['solver']}
- 最大迭代次数: {data['当前配置']['max_iter']}
- 测试集比例: {data['当前配置']['test_size']}%

当前效果：
- 准确率: {data['当前效果']['准确率']}

【历史建议回顾】
{st.session_state.ai_history[-1] if st.session_state.ai_history else '这是第一次咨询，还没有历史建议。'}

【重要要求】
1. 逻辑回归参数简单，建议根据样本量调整C值：
   - 小样本 (<500)：C=1.0 或更大（少正则化）
   - 中等样本 (500-2000)：C=0.5-1.0
   - 大样本 (>2000)：C=0.1-0.5（强正则化防止过拟合）
2. 如果准确率低，可以尝试减小C值（加强正则化）
3. 如果训练时间长，可以减小max_iter或换求解器

如果是第二次或更多次咨询，请参考之前给的建议，告诉用户这次调整后的效果如何。

请以JSON格式返回，只返回JSON：
{{
    "评价": "结合基准和样本量评价当前模型",
    "优化建议": {{
        "C": 建议值,
        "penalty": "建议值",
        "solver": "建议值",
        "max_iter": 建议值,
        "test_size": 建议值（整数百分比）
    }},
    "预期效果": "说明优化后可能达到的效果"
}}
【约束条件】
- 当前可用求解器: {data['约束条件']['可用求解器']}
- 是否多分类: {'是' if data['约束条件']['是否多分类'] else '否'}
- 请只从可用求解器中选择，不要建议不可用的求解器
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
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "max_iter": 100,
                    "test_size": 20
                },
                "预期效果": "优化后准确率预计可提升5-15%"
            }

    except Exception as e:
        print(f"AI调用失败：{str(e)}")
        return {
            "评价": f"AI服务暂时不可用，使用默认建议",
            "优化建议": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 100,
                "test_size": 20
            },
            "预期效果": "优化后准确率预计可提升5-15%"
        }


# ========== 逻辑回归训练函数 ==========
def train_lr():
    """逻辑回归训练主函数"""
    st.subheader("📈 逻辑回归训练")

    df = st.session_state.processed_df
    target = df.columns[-1]
    feature = df.columns[:-1].tolist()
    label_encoders = st.session_state.label_encoders
    is_classification = target in label_encoders

    # ===== 检查是否有AI建议的参数 =====
    if 'ai_suggested_params' in st.session_state:
        suggested = st.session_state.ai_suggested_params
        st.success("✨ 已应用AI建议的参数，你可以直接点击训练")

        # 处理C值
        try:
            default_C = float(suggested.get('C', 1.0))
            if default_C <= 0:
                default_C = 1.0
        except:
            default_C = 1.0

        # 处理penalty - 确保是合法的值
        valid_penalties = ['l1', 'l2', 'elasticnet', 'none']
        default_penalty = suggested.get('penalty', 'l2')
        if default_penalty not in valid_penalties:
            default_penalty = 'l2'
            st.warning(f"penalty值 '{default_penalty}' 无效，已自动调整为 'l2'")

        # 处理solver - 确保是合法的值
        valid_solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
        default_solver = suggested.get('solver', 'lbfgs')
        if default_solver not in valid_solvers:
            default_solver = 'lbfgs'
            st.warning(f"solver值 '{default_solver}' 无效，已自动调整为 'lbfgs'")

        # 检查penalty和solver的组合是否合法
        if default_penalty == 'l1' and default_solver not in ['liblinear', 'saga']:
            default_solver = 'liblinear'
            st.warning(f"l1 penalty需要liblinear或saga求解器，已自动调整为liblinear")
        elif default_penalty == 'elasticnet' and default_solver != 'saga':
            default_solver = 'saga'
            st.warning(f"elasticnet penalty需要saga求解器，已自动调整为saga")

        # 处理max_iter
        try:
            default_max_iter = int(suggested.get('max_iter', 100))
            if default_max_iter <= 0:
                default_max_iter = 100
        except:
            default_max_iter = 100

        default_test_size = int(suggested.get('test_size', 20))
    else:
        default_C = 1.0
        default_penalty = 'l2'
        default_solver = 'lbfgs'
        default_max_iter = 100
        default_test_size = 20

    if not is_classification:
        st.warning("⚠️ 逻辑回归主要用于分类问题，回归问题请使用其他模型")
        if st.button("返回选择"):
            st.session_state.step = 1
            st.rerun()
        return

    le = label_encoders[target]
    num_classes = len(le.classes_)

    # 显示数据信息
    st.markdown("### 1. 数据识别")
    st.info(f"系统识别到：{len(feature)}个特征列，1个标签列")
    st.write(f"**标签列:** {target}")
    st.write(f"**问题类型:** 分类")
    st.write(f"**类别数:** {num_classes}")

    n_samples = len(df)
    if n_samples < 500:
        st.success(f"✅ 当前样本量 {n_samples}，逻辑回归是理想选择")

    # 参数配置
    st.markdown("### 2. 逻辑回归参数配置")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**正则化参数**")
        C = st.number_input("C值（正则化强度）", min_value=0.01, max_value=10.0,
                            value=default_C, step=0.1, key="lr_C")

        penalty = st.selectbox("正则化类型", ["l2", "l1", "elasticnet", "none"],
                               index=["l2", "l1", "elasticnet", "none"].index(default_penalty), key="lr_penalty")

    with col2:
        st.write("**求解器参数**")
        # 根据类别数动态调整求解器选项
        if num_classes > 2:
            solver_options = ["lbfgs", "newton-cg", "sag", "saga"]
            default_solver = default_solver if default_solver in solver_options else "lbfgs"
        else:
            solver_options = ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
            default_solver = default_solver if default_solver in solver_options else "lbfgs"

        solver = st.selectbox("求解器", solver_options,
                              index=solver_options.index(default_solver) if default_solver in solver_options else 0,
                              key="lr_solver")

        max_iter = st.number_input("最大迭代次数", min_value=100, max_value=1000,
                                   value=default_max_iter, step=50, key="lr_max_iter")

    # 数据划分
    st.markdown("### 3. 数据划分")
    test_size = st.slider("测试集比例 (%)", 10, 40, value=int(default_test_size), key="lr_test_size")

    X = df[feature].values
    y = df[target].values

    try:
        test_size_float = float(test_size) / 100.0
        if test_size_float <= 0 or test_size_float >= 1:
            test_size_float = 0.2
            st.warning(f"测试集比例 {test_size}% 无效，已自动调整为 20%")
    except:
        test_size_float = 0.2
        st.warning(f"测试集比例格式错误，已自动调整为 20%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_float, random_state=42,
        stratify=y
    )

    # 可选：标准化
    use_scaler = st.checkbox("使用标准化", value=True,
                             help="逻辑回归对特征尺度敏感，建议标准化")

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
    if st.button("🚀 开始训练逻辑回归", type="primary", use_container_width=True, key="lr_train_btn"):

        # 保存配置
        st.session_state.lr_config = {
            'target': target,
            'feature': feature,
            'C': C,
            'penalty': penalty,
            'solver': solver,
            'max_iter': max_iter,
            'test_size': test_size / 100,
            'use_scaler': use_scaler
        }
        st.session_state.is_classification = True
        st.session_state.num_classes = num_classes
        st.session_state.label_encoder = le

        # 数据预处理
        if use_scaler:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.session_state.lr_scaler = scaler
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # ✅ 添加 Class Weight 处理
        class_weight = None
        if st.session_state.user_choices.get('balance') == 'class_weight':
            class_weight = 'balanced'
            st.info(f"📊 应用Class Weight：balanced")

        # 在创建模型之前，确保所有参数都合法
        with st.spinner("模型训练中..."):
            start_time = time.time()

            # 多分类自动处理
            if num_classes > 2:
                # 多分类问题
                multi_class = 'multinomial'
                # 多分类不能使用liblinear
                if solver == 'liblinear':
                    solver = 'lbfgs'
                    st.warning("多分类问题不支持liblinear求解器，已自动调整为lbfgs")
            else:
                multi_class = 'ovr'

            # 确保solver和penalty的组合合法
            if penalty == 'l1':
                if solver not in ['liblinear', 'saga']:
                    solver = 'liblinear'
                    st.warning("l1 penalty需要liblinear或saga求解器，已自动调整为liblinear")
            elif penalty == 'elasticnet':
                if solver != 'saga':
                    solver = 'saga'
                    st.warning("elasticnet penalty需要saga求解器，已自动调整为saga")
            elif penalty == 'none':
                if solver not in ['lbfgs', 'newton-cg', 'sag', 'saga']:
                    solver = 'lbfgs'
                    st.warning("none penalty需要lbfgs、newton-cg、sag或saga求解器，已自动调整为lbfgs")
            elif penalty == 'l2':
                # l2是默认值，大多数求解器都支持
                if num_classes > 2 and solver not in ['lbfgs', 'newton-cg', 'sag', 'saga']:
                    solver = 'lbfgs'
                    st.warning("多分类问题需要lbfgs、newton-cg、sag或saga求解器，已自动调整为lbfgs")

            # 确保solver和multi_class的组合合法
            if solver == 'liblinear' and multi_class == 'multinomial':
                solver = 'lbfgs'
                st.warning("liblinear求解器不支持multinomial，已自动调整为lbfgs")

            # 在创建模型之前添加参数检查
            try:
                # 确保C是正数
                C_value = float(C)
                if C_value <= 0:
                    C_value = 1.0
                    st.warning(f"C值 {C} 无效，已自动调整为 1.0")
            except:
                C_value = 1.0
                st.warning(f"C值格式错误，已自动调整为 1.0")

            # 确保max_iter是正整数
            try:
                max_iter_value = int(max_iter)
                if max_iter_value <= 0:
                    max_iter_value = 100
                    st.warning(f"最大迭代次数 {max_iter} 无效，已自动调整为 100")
            except:
                max_iter_value = 100
                st.warning(f"最大迭代次数格式错误，已自动调整为 100")

            # 确保class_weight合法
            if class_weight is not None and class_weight not in ['balanced', None]:
                class_weight = 'balanced'
                st.warning(f"class_weight值无效，已自动调整为 'balanced'")

            # 显示最终使用的参数
            st.info(
                f"使用参数 - C: {C_value}, penalty: {penalty}, solver: {solver}, max_iter: {max_iter_value}, multi_class: {multi_class}")

            model = LogisticRegression(
                C=C_value,
                penalty=penalty,
                solver=solver,
                max_iter=max_iter_value,
                multi_class=multi_class,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )

            # 训练
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time

            # 预测
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # 评估
            train_score = accuracy_score(y_train, y_pred_train)
            test_score = accuracy_score(y_test, y_pred_test)

            # 特征重要性（系数）
            if num_classes == 2:
                coefficients = dict(zip(feature, model.coef_[0]))
            else:
                coefficients = {f'类别{i}': dict(zip(feature, model.coef_[i]))
                                for i in range(num_classes)}

            # 保存结果
            st.session_state.lr_model = model
            st.session_state.lr_train_score = train_score
            st.session_state.lr_test_score = test_score
            st.session_state.lr_train_time = train_time
            st.session_state.lr_coefficients = coefficients
            st.session_state.lr_X_test = X_test_scaled
            st.session_state.lr_y_test = y_test
            st.session_state.lr_y_pred_test = y_pred_test

            st.success(f"✅ 模型训练完成！ 训练时间: {train_time:.2f}秒")
            st.session_state.step = 4
            st.rerun()

    # 返回按钮
    if st.button("← 返回数据清洗", key="lr_back_clean"):
        st.session_state.step = 2
        st.rerun()


# ========== 逻辑回归预测函数 ==========
def predict_lr():
    """逻辑回归预测函数"""
    st.subheader("🔮 逻辑回归预测")
    if 'lr_predict_counter' not in st.session_state:
        st.session_state.lr_predict_counter = 0
    current_counter = st.session_state.lr_predict_counter

    # 判断模型来源：训练得到的还是上传的
    model = None
    config = None
    feature = None
    scaler = None

    # 情况1：通过训练流程得到的模型（有完整配置）
    if 'lr_model' in st.session_state and st.session_state.lr_model is not None:
        model = st.session_state.lr_model
        if 'lr_config' in st.session_state and st.session_state.lr_config is not None:
            config = st.session_state.lr_config
            feature = config['feature']
            # ✅ 修改：优先使用 lr_scaler
            scaler = st.session_state.get('lr_scaler', None)
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
            scaler = st.session_state.get('lr_scaler', None)

    # 情况2：通过上传得到的模型
    elif 'uploaded_model' in st.session_state and st.session_state.uploaded_model is not None:
        if st.session_state.get('uploaded_model_type') == 'lr':
            model = st.session_state.uploaded_model
            st.info("📌 使用上传的逻辑回归模型进行预测")

            # ✅ 修改：从上传的模型获取scaler
            scaler = st.session_state.get('lr_scaler', st.session_state.get('uploaded_scaler', None))

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
            st.warning("请先训练模型或上传逻辑回归模型文件")
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
    is_classification = st.session_state.get('is_classification', True)
    num_classes = st.session_state.get('num_classes', None)

    if num_classes is None and label_encoder is not None:
        num_classes = len(label_encoder.classes_)

    # 显示配置信息（如果有）
    if config is not None:
        with st.expander("📋 当前模型配置", expanded=False):
            st.write(f"- C值: {config.get('C')}")
            st.write(f"- 正则化类型: {config.get('penalty')}")
            st.write(f"- 求解器: {config.get('solver')}")
            st.write(f"- 最大迭代次数: {config.get('max_iter')}")
            st.write(f"- 测试集比例: {config.get('test_size', 0.2) * 100:.0f}%")
            st.write(f"- 使用标准化: {'是' if config.get('use_scaler') else '否'}")
    else:
        st.info("ℹ️ 当前为上传模型，仅支持预测功能")


    # 训练结果（仅当有训练结果时显示）
    if 'lr_test_score' in st.session_state:
        st.markdown("### 📊 训练结果")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("训练集准确率", f"{st.session_state.lr_train_score:.4f}")
        with col2:
            st.metric("测试集准确率", f"{st.session_state.lr_test_score:.4f}")
        with col3:
            st.metric("训练时间", f"{st.session_state.lr_train_time:.2f}秒")

        # 系数可视化（如果有）
        if 'lr_coefficients' in st.session_state:
            st.markdown("### 📈 特征系数（影响程度）")
            coef_dict = st.session_state.lr_coefficients
            if num_classes == 2:
                coef_df = pd.DataFrame({
                    '特征': list(coef_dict.keys()),
                    '系数': list(coef_dict.values())
                }).sort_values('系数', ascending=True)
                fig = px.bar(coef_df, x='系数', y='特征', orientation='h',
                             title='特征系数（正数表示正向影响）',
                             color='系数', color_continuous_scale='RdBu_r')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 多分类的系数显示
                if label_encoder:
                    tabs = st.tabs([f'类别 {label_encoder.inverse_transform([i])[0]}' for i in range(num_classes)])
                    for i, tab in enumerate(tabs):
                        with tab:
                            coef_df = pd.DataFrame({
                                '特征': list(coef_dict[f'类别{i}'].keys()),
                                '系数': list(coef_dict[f'类别{i}'].values())
                            }).sort_values('系数', ascending=True)
                            fig = px.bar(coef_df, x='系数', y='特征', orientation='h',
                                         title=f'类别 {label_encoder.inverse_transform([i])[0]} 的系数',
                                         color='系数', color_continuous_scale='RdBu_r')
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)

        # 分类报告（如果有）
        if 'lr_y_test' in st.session_state and 'lr_y_pred_test' in st.session_state:
            with st.expander("📋 分类详细报告"):
                report = classification_report(
                    st.session_state.lr_y_test,
                    st.session_state.lr_y_pred_test,
                    target_names=label_encoder.classes_ if label_encoder else None,
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.2f}"))

    st.markdown("---")

    # AI建议按钮（仅当有配置时显示）
    if config is not None:
        col1, col2, col3 = st.columns(3)
        with col3:
            if st.button("🤖 AI评价及建议", use_container_width=True, key="lr_ai_btn"):
                if not st.session_state.get('ai_enabled', False):
                    st.error("❌ AI功能未启用，请检查API Key配置")
                    st.stop()

                # 重新定义 solver_options
                if num_classes > 2:
                    solver_options = ["lbfgs", "newton-cg", "sag", "saga"]
                    default_solver = "lbfgs"
                else:
                    solver_options = ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
                    penalty = config.get('penalty', 'l2')
                    if penalty == "l1":
                        default_solver = "liblinear"
                    elif penalty == "elasticnet":
                        default_solver = "saga"
                    else:
                        default_solver = "lbfgs"

                test_size_percent = int(config.get('test_size', 0.2) * 100)
                accuracy = f"{st.session_state.lr_test_score:.2%}" if 'lr_test_score' in st.session_state else "N/A"

                ai_input = {
                    "数据概况": {
                        "样本量": len(st.session_state.get('processed_df', pd.DataFrame())),
                        "特征数": len(feature),
                        "问题类型": "分类",
                        "类别数": num_classes
                    },
                    "当前配置": {
                        "C": config.get('C'),
                        "penalty": config.get('penalty'),
                        "solver": config.get('solver'),
                        "max_iter": config.get('max_iter'),
                        "test_size": test_size_percent
                    },
                    "当前效果": {
                        "准确率": accuracy
                    },
                    "约束条件": {
                        "可用求解器": solver_options,
                        "是否多分类": num_classes > 2
                    }
                }

                with st.spinner("AI正在分析模型表现..."):
                    advice = get_lr_ai_advice(ai_input)

                if advice:
                    suggested_solver = advice['优化建议'].get('solver')
                    if suggested_solver and suggested_solver not in solver_options:
                        advice['优化建议']['solver'] = default_solver
                        st.warning(f"⚠️ AI建议的求解器 '{suggested_solver}' 当前不可用，已自动替换为 '{default_solver}'")

                    st.session_state.ai_advice = advice
                    st.session_state.ai_advice_generated = True
                    st.rerun()

    # AI提示信息
    if st.session_state.get('ai_advice_generated', False):
        st.success("✅ AI建议已生成！请查看左侧边栏下方🤖 AI模型建议，并点击✨ 应用AI建议参数。")
        st.info("💡 提示：请根据建议在步骤3中手动调整参数")

    # 模型保存（仅当有配置时显示）
    if config is not None:
        col1, col2 = st.columns(2)
        with col1:
            try:
                # 创建打包数据
                model_package = {
                    'model': st.session_state.lr_model,
                    'scaler': st.session_state.get('lr_scaler', None),
                    'config': config,
                    'model_type': 'lr',
                    'feature_names': feature,
                    'num_classes': num_classes,
                    'label_encoder_classes': label_encoder.classes_.tolist() if label_encoder else None
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
                    file_name=f"lr_model_package_{time.strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ 模型保存失败：{str(e)}")

        with col2:
            config_json = json.dumps(config, indent=2, ensure_ascii=False)
            st.download_button(
                label="📄 下载配置 (.json)",
                data=config_json,
                file_name=f"lr_config_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    st.markdown("---")

    # ===== 预测功能 =====
    st.markdown("### 🔮 使用模型预测")
    st.markdown("""
                    上传文件批量预测需注意：
                    1.确保上传文件为：CSV或Excel
                    2. **特征列完整**：必须包含模型训练时的所有特征列（列名需与训练数据完全一致，大小写/空格敏感）；
                    3. **无缺失值**：特征列不能有空白单元格（NaN），需提前删除含缺失值的行或填充；
                    4. **数据类型正确**：
                       (1) 数值型特征（如面积、年龄）：仅保留纯数字（int/float），不含文本、特殊符号（如「120㎡」「二十岁」）；
                       (2)类别型特征（如性别、学历）：填写原始文本值（如「男/女」「本科/硕士」），**不要填写编码后的数字**；
                    5. **无需标签列**：预测文件仅需特征列，无需包含训练时的标签列（模型会自动生成预测结果）；
                    6. **编码一致**：类别特征的取值需与训练数据一致（如训练时「性别」只有「男/女」，预测时不能出现「未知」）。

                    ❗ 若预测报错，请按上述要求检查文件后重试（常见问题：列名不一致、含非数值字符、存在缺失值）。
                    """)

    # 初始化session state
    if 'lr_show_prediction' not in st.session_state:
        st.session_state.lr_show_prediction = False
    if 'lr_pred_result' not in st.session_state:
        st.session_state.lr_pred_result = None
    if 'lr_batch_result' not in st.session_state:
        st.session_state.lr_batch_result = None
    if 'lr_manual_counter' not in st.session_state:
        st.session_state.lr_manual_counter = 0

    # 预测方式选择
    input_method = st.radio(
        "选择输入方式",
        ["手动输入", "上传文件"],
        horizontal=True,
        key=f"lr_input_method_{current_counter}"
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
                    key=f"lr_pred_feat_{i}_{current_counter}"
                )

                encoded_value = le.transform([selected_text])[0]
                input_values.append(float(encoded_value))
                input_display.append(selected_text)
            else:
                val = st.number_input(
                    f"输入 {feat}",
                    value=0.0,
                    step=0.1,
                    key=f"lr_pred_num_{i}_{current_counter}",
                    format="%f"
                )
                input_values.append(val)
                input_display.append(f"{val}")

        # 预测按钮用当前计数器
        if st.button("🚀 开始预测", type="primary", use_container_width=True,
                     key=f"lr_predict_btn_{current_counter}"):
            try:
                input_arr = np.array(input_values).reshape(1, -1)

                # 关键修复：先判断 scaler 是不是 None
                if scaler is not None:
                    input_scaled = scaler.transform(input_arr)
                    pred = model.predict(input_scaled)
                    proba = model.predict_proba(input_scaled)
                else:
                    # scaler 是 None 时，直接用原始数据
                    pred = model.predict(input_arr)
                    proba = model.predict_proba(input_arr)
                    st.info("ℹ️ 使用原始数据进行预测（无标准化）")

                st.session_state.lr_pred_result = {
                    'input_display': input_display,
                    'features': feature,
                    'prediction': pred[0],
                    'probabilities': proba[0]
                }
                st.session_state.lr_show_prediction = True
                st.session_state.lr_predict_counter += 1
                st.rerun()
            except Exception as e:
                st.error(f"❌ 预测出错：{str(e)}")

        if st.session_state.lr_show_prediction and st.session_state.lr_pred_result is not None:
            result = st.session_state.lr_pred_result
            st.markdown("---")
            st.markdown("### 📝 输入特征：")
            for i, feat in enumerate(result['features']):
                st.write(f"- {feat}：{result['input_display'][i]}")

            st.markdown("### 🎯 预测结果")
            pred = result['prediction']
            proba = result['probabilities']

            if label_encoder:
                res_text = label_encoder.inverse_transform([int(pred)])[0]
                st.success(f"**预测类别：{res_text}**")
                st.write("各类别概率：")
                for i, p in enumerate(proba):
                    if i < len(label_encoder.classes_):
                        class_name = label_encoder.inverse_transform([i])[0]
                        st.write(f"- {class_name}: {p:.2%}")
            else:
                st.success(f"**预测类别索引：{int(pred)}**")
                st.write("各类别概率：")
                for i, p in enumerate(proba):
                    st.write(f"- 类别{i}: {p:.2%}")


    else:  # 上传文件模式

        st.write("上传文件进行批量预测：")

        uploader_key = f"lr_pred_file_{st.session_state.get('lr_batch_counter', 0)}"

        pred_file = st.file_uploader("选择预测文件", ['csv', 'xlsx'], key=uploader_key)

        if pred_file and st.session_state.lr_batch_result is None:

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

                    if st.button("开始批量预测", type="primary", key="lr_batch_predict"):

                        with st.spinner("正在预测中..."):

                            X_pred = df_pred[feature].values

                            if scaler:
                                X_pred = scaler.transform(X_pred)

                            preds = model.predict(X_pred)

                            probas = model.predict_proba(X_pred)

                            result_df = df_pred.copy()

                            # ✅ 逻辑回归总是分类问题

                            if num_classes == 2:  # 二分类

                                if label_encoder:

                                    result_df['预测结果'] = [label_encoder.inverse_transform([int(p)])[0] for p in
                                                             preds]

                                else:

                                    result_df['预测结果'] = preds

                                result_df['预测概率'] = probas[:, 1]  # 正类的概率

                            else:  # 多分类

                                if label_encoder:

                                    result_df['预测结果'] = [label_encoder.inverse_transform([int(p)])[0] for p in
                                                             preds]

                                else:

                                    result_df['预测结果'] = preds

                                for i in range(probas.shape[1]):

                                    if label_encoder and i < len(label_encoder.classes_):

                                        class_name = label_encoder.inverse_transform([i])[0]

                                    else:

                                        class_name = f"类别{i}"

                                    result_df[f'{class_name}_概率'] = probas[:, i]

                            st.session_state.lr_batch_result = result_df

                            st.rerun()

            except Exception as e:

                st.error(f"❌ 预测出错：{str(e)}")

        if st.session_state.lr_batch_result is not None:
            result_df = st.session_state.lr_batch_result
            st.success(f"✅ 完成 {len(result_df)} 条数据预测")
            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载预测结果",
                data=csv_data,
                file_name="预测结果.csv",
                mime="text/csv",
                key="lr_download_results"
            )

    st.markdown("---")

    # 操作按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 重新训练", use_container_width=True, key="lr_retrain"):
            keys_to_remove = ['lr_model', 'lr_config', 'lr_train_score', 'lr_test_score',
                              'lr_coefficients', 'lr_X_test', 'lr_y_test', 'lr_y_pred_test',
                              'lr_scaler', 'lr_batch_result', 'lr_pred_result']
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("🔮 新预测", use_container_width=True, key="lr_new_prediction"):
            st.session_state.lr_show_prediction = False
            st.session_state.lr_pred_result = None
            st.session_state.lr_batch_result = None
            st.session_state.lr_manual_counter += 1
            if 'lr_batch_counter' in st.session_state:
                st.session_state.lr_batch_counter += 1
            st.rerun()
    with col3:
        if st.button("🏁 全新开始", use_container_width=True, key="lr_fresh"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import tempfile
import os
import joblib
import warnings

warnings.filterwarnings('ignore')


# ========== AI调用函数（为Ridge回归定制） ==========
def get_ridge_ai_advice(data):
    """调用DeepSeek API获取Ridge回归模型评价和优化建议"""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )

        # 构建当前对话的prompt
        current_prompt = f"""
【当前模型表现】
数据概况：
- 样本量：{data['数据概况']['样本量']}行
- 特征数：{data['数据概况']['特征数']}个
- 问题类型：回归

当前配置：
- alpha（正则化强度）: {data['当前配置']['alpha']}
- 是否标准化: {'是' if data['当前配置']['normalize'] else '否'}
- 求解器: {data['当前配置']['solver']}
- 测试集比例: {data['当前配置']['test_size']}%

当前效果：
- MAE: {data['当前效果']['MAE']}
- R²: {data['当前效果']['R2']}

【历史建议回顾】
{st.session_state.ai_history[-1] if st.session_state.ai_history else '这是第一次咨询，还没有历史建议。'}

【重要要求】
1. Ridge回归适合小样本，主要参数就是alpha：
   - alpha越小（0.01-0.1）：模型更灵活，可能过拟合
   - alpha适中（0.1-1.0）：平衡拟合度和稳定性
   - alpha越大（1.0-10.0）：正则化更强，防止过拟合
2. 根据样本量建议：
   - 样本<100：alpha=1.0-5.0（强正则化）
   - 样本100-500：alpha=0.5-2.0（你的范围）
   - 样本>500：alpha=0.1-1.0（弱正则化）
3. 如果MAE很大，可以尝试减小alpha
4. 如果R²为负，说明模型完全不能用，建议检查数据或换模型

如果是第二次或更多次咨询，请参考之前给的建议，告诉用户这次调整后的效果如何。

请以JSON格式返回，只返回JSON：
{{
    "评价": "结合样本量和当前效果评价模型",
    "优化建议": {{
        "alpha": 建议值,
        "normalize": 建议值,
        "solver": "建议值",
        "test_size": 建议值（整数百分比）
    }},
    "预期效果": "说明优化后MAE或R²可能的变化"
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
            history_entry = f"第{len(st.session_state.ai_history) + 1}次：MAE={data['当前效果']['MAE']:.4f} → {advice_json['评价']}"
            st.session_state.ai_history.append(history_entry)
            return advice_json
        else:
            return {
                "评价": "模型表现不理想，建议调整alpha",
                "优化建议": {
                    "alpha": 1.0,
                    "normalize": True,
                    "solver": "auto",
                    "test_size": 20
                },
                "预期效果": "优化后MAE预计可降低10-20%"
            }

    except Exception as e:
        print(f"AI调用失败：{str(e)}")
        return {
            "评价": f"AI服务暂时不可用，使用默认建议",
            "优化建议": {
                "alpha": 1.0,
                "normalize": True,
                "solver": "auto",
                "test_size": 20
            },
            "预期效果": "优化后MAE预计可降低10-20%"
        }


# ========== Ridge回归训练函数 ==========
def train_ridge():
    """Ridge回归训练主函数"""
    st.subheader("📈 Ridge回归训练（小样本专用）")

    df = st.session_state.processed_df
    target = df.columns[-1]
    feature = df.columns[:-1].tolist()
    label_encoders = st.session_state.label_encoders
    is_classification = target in label_encoders

    if 'ai_suggested_params' in st.session_state:
        suggested = st.session_state.ai_suggested_params
        st.success("✨ 已应用AI建议的参数，你可以直接点击训练")

        default_alpha = float(suggested.get('alpha', 1.0))
        default_normalize = bool(suggested.get('normalize', True))
        default_solver = suggested.get('solver', 'auto')
        default_test_size = int(suggested.get('test_size', 20))
    else:
        default_alpha = 1.0
        default_normalize = True
        default_solver = 'auto'
        default_test_size = 20

    if is_classification:
        st.error("❌ Ridge回归只能用于回归问题！当前数据是分类问题，请选择其他模型。")
        if st.button("返回选择"):
            st.session_state.step = 1
            st.rerun()
        return

    # 显示数据信息
    st.markdown("### 1. 数据识别")
    st.info(f"系统识别到：{len(feature)}个特征列，1个标签列")
    st.write(f"**标签列:** {target}")
    st.write(f"**问题类型:** 回归")

    n_samples = len(df)
    if n_samples < 500:
        st.success(f"✅ 当前样本量 {n_samples}，Ridge回归是最佳选择！")
    elif n_samples < 1000:
        st.info(f"ℹ️ 当前样本量 {n_samples}，Ridge回归可用，随机森林可能更好")
    else:
        st.warning(f"⚠️ 当前样本量 {n_samples}较大，可以考虑使用随机森林或LightGBM")

    # 参数配置
    st.markdown("### 2. 参数配置")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**核心参数**")
        alpha = st.number_input("alpha（正则化强度）", min_value=0.01, max_value=10.0,
                                value=default_alpha, step=0.1, key="ridge_alpha")

        normalize = st.checkbox("是否标准化", value=default_normalize, key="ridge_normalize")

    with col2:
        st.write("**求解器**")
        solver = st.selectbox("求解器", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                              index=["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"].index(
                                  default_solver),
                              key="ridge_solver")

        # 交叉验证选项
        use_cv = st.checkbox("使用交叉验证自动选择alpha", value=False,
                             help="会用多个alpha值交叉验证，选出最优alpha，但较慢")

    if use_cv:
        cv_alphas = st.text_input("alpha候选值（用逗号分隔）", "0.1,0.5,1.0,2.0,5.0",
                                  help="例如：0.1,0.5,1.0,2.0,5.0")

    # 数据划分
    st.markdown("### 3. 数据划分")
    test_size = st.slider("测试集比例 (%)", 10, 40, value=int(default_test_size), key="ridge_test_size")

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
        X, y, test_size=test_size_float, random_state=42
    )

    # 标准化
    if normalize:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = None
        X_train_scaled = X_train
        X_test_scaled = X_test

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
    if st.button("🚀 开始训练Ridge回归", type="primary", use_container_width=True, key="ridge_train_btn"):

        # 保存配置
        st.session_state.ridge_config = {
            'target': target,
            'feature': feature,
            'alpha': alpha,
            'normalize': normalize,
            'solver': solver,
            'test_size': test_size / 100,
            'use_cv': use_cv
        }
        st.session_state.is_classification = False
        st.session_state.num_classes = None
        st.session_state.label_encoder = None
        st.session_state.ridge_scaler = scaler  # 修复1: 保存scaler供预测使用

        # 创建模型
        with st.spinner("模型训练中..."):
            start_time = time.time()

            if use_cv:
                # 使用交叉验证
                alphas = [float(a.strip()) for a in cv_alphas.split(',')]
                model = RidgeCV(alphas=alphas, store_cv_results=True)
            else:
                model = Ridge(alpha=float(alpha), solver=solver)

            # 训练
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time

            # 预测
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # 评估
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            # 系数
            coefficients = dict(zip(feature, model.coef_))

            # 保存结果
            st.session_state.ridge_model = model
            st.session_state.ridge_train_mae = train_mae
            st.session_state.ridge_test_mae = test_mae
            st.session_state.ridge_train_r2 = train_r2
            st.session_state.ridge_test_r2 = test_r2
            st.session_state.ridge_train_time = train_time
            st.session_state.ridge_coefficients = coefficients
            st.session_state.ridge_X_test = X_test_scaled
            st.session_state.ridge_y_test = y_test
            st.session_state.ridge_y_pred_test = y_pred_test
            if use_cv:
                st.session_state.ridge_best_alpha = model.alpha_

            st.success(f"✅ 模型训练完成！ 训练时间: {train_time:.2f}秒")
            st.session_state.step = 4
            st.rerun()

    # 返回按钮
    if st.button("← 返回数据清洗", key="ridge_back_clean"):
        st.session_state.step = 2
        st.rerun()


# ========== Ridge回归预测函数 ==========
def predict_ridge():
    """Ridge回归预测函数"""
    st.subheader("🔮 Ridge回归预测")
    if 'ridge_predict_counter' not in st.session_state:
        st.session_state.ridge_predict_counter = 0
    current_counter = st.session_state.ridge_predict_counter

    # 判断模型来源：训练得到的还是上传的
    model = None
    config = None
    feature = None
    scaler = None

    # 情况1：通过训练流程得到的模型（有完整配置）
    if 'ridge_model' in st.session_state and st.session_state.ridge_model is not None:
        model = st.session_state.ridge_model
        if 'ridge_config' in st.session_state and st.session_state.ridge_config is not None:
            config = st.session_state.ridge_config
            feature = config['feature']
            # ✅ 修改：优先使用 ridge_scaler
            scaler = st.session_state.get('ridge_scaler', None)
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
            scaler = st.session_state.get('ridge_scaler', None)

    # 情况2：通过上传得到的模型
    elif 'uploaded_model' in st.session_state and st.session_state.uploaded_model is not None:
        if st.session_state.get('uploaded_model_type') == 'ridge':
            model = st.session_state.uploaded_model
            st.info("📌 使用上传的Ridge回归模型进行预测")

            # ✅ 修改：从上传的模型获取scaler
            scaler = st.session_state.get('ridge_scaler', st.session_state.get('uploaded_scaler', None))

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
            st.warning("请先训练模型或上传Ridge回归模型文件")
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

    if config is not None:
        with st.expander("📋 当前模型配置", expanded=False):
            st.write(f"- alpha: {config.get('alpha')}")
            st.write(f"- 标准化: {'是' if config.get('normalize') else '否'}")
            st.write(f"- 求解器: {config.get('solver')}")
            st.write(f"- 测试集比例: {config.get('test_size', 0.2) * 100:.0f}%")
            if 'ridge_best_alpha' in st.session_state:
                st.write(f"- 最佳alpha: {st.session_state.ridge_best_alpha:.4f}")
    else:
        st.info("ℹ️ 当前为上传模型，仅支持预测功能")

    # 训练结果（仅当有训练结果时显示）
    if 'ridge_test_mae' in st.session_state:
        st.markdown("### 📊 训练结果")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("训练集MAE", f"{st.session_state.ridge_train_mae:.4f}")
            st.metric("训练集R²", f"{st.session_state.ridge_train_r2:.4f}")
        with col2:
            st.metric("测试集MAE", f"{st.session_state.ridge_test_mae:.4f}")
            st.metric("测试集R²", f"{st.session_state.ridge_test_r2:.4f}")
        with col3:
            st.metric("训练时间", f"{st.session_state.ridge_train_time:.2f}秒")

        # 系数可视化
        if 'ridge_coefficients' in st.session_state:
            st.markdown("### 📈 特征系数（影响程度）")
            coef_dict = st.session_state.ridge_coefficients
            coef_df = pd.DataFrame({
                '特征': list(coef_dict.keys()),
                '系数': list(coef_dict.values())
            }).sort_values('系数', ascending=True)
            fig = px.bar(coef_df, x='系数', y='特征', orientation='h',
                         title='特征系数（正数表示正向影响）',
                         color='系数', color_continuous_scale='RdBu_r')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # 预测值 vs 真实值
        if 'ridge_y_test' in st.session_state and 'ridge_y_pred_test' in st.session_state:
            st.markdown("### 📉 预测值 vs 真实值")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.ridge_y_test,
                y=st.session_state.ridge_y_pred_test,
                mode='markers',
                name='预测值',
                marker=dict(color='blue', size=8)
            ))
            min_val = min(st.session_state.ridge_y_test.min(), st.session_state.ridge_y_pred_test.min())
            max_val = max(st.session_state.ridge_y_test.max(), st.session_state.ridge_y_pred_test.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='理想线',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(title='预测值 vs 真实值', xaxis_title='真实值', yaxis_title='预测值')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # AI建议按钮（仅当有配置时显示）
    if config is not None:
        col1, col2, col3 = st.columns(3)
        with col3:
            if st.button("🤖 AI评价及建议", use_container_width=True, key="ridge_ai_btn"):
                if not st.session_state.get('ai_enabled', False):
                    st.error("❌ AI功能未启用，请检查API Key配置")
                    st.stop()

                test_size_percent = int(config.get('test_size', 0.2) * 100)

                ai_input = {
                    "数据概况": {
                        "样本量": len(st.session_state.processed_df),
                        "特征数": len(feature),
                        "问题类型": "回归"
                    },
                    "当前配置": {
                        "alpha": config.get('alpha'),
                        "normalize": config.get('normalize'),
                        "solver": config.get('solver'),
                        "test_size": test_size_percent
                    },
                    "当前效果": {
                        "MAE": f"{st.session_state.ridge_test_mae:.4f}",
                        "R2": f"{st.session_state.ridge_test_r2:.4f}"
                    }
                }

                with st.spinner("AI正在分析模型表现..."):
                    advice = get_ridge_ai_advice(ai_input)

                if advice:
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
                    'model': st.session_state.ridge_model,
                    'scaler': st.session_state.get('ridge_scaler', None),
                    'config': config,
                    'model_type': 'ridge',
                    'feature_names': feature
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
                    file_name=f"ridge_model_package_{time.strftime('%Y%m%d_%H%M%S')}.pkl",
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
                file_name=f"ridge_config_{time.strftime('%Y%m%d_%H%M%S')}.json",
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
    if 'ridge_show_prediction' not in st.session_state:
        st.session_state.ridge_show_prediction = False
    if 'ridge_pred_result' not in st.session_state:
        st.session_state.ridge_pred_result = None
    if 'ridge_batch_result' not in st.session_state:
        st.session_state.ridge_batch_result = None
    if 'ridge_manual_counter' not in st.session_state:
        st.session_state.ridge_manual_counter = 0

    # 预测方式选择 - 修复变量名
    input_method = st.radio(
        "选择输入方式",
        ["手动输入", "上传文件"],
        horizontal=True,
        key=f"ridge_input_method_{current_counter}"
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
                    key=f"ridge_pred_feat_{i}_{current_counter}"
                )

                encoded_value = le.transform([selected_text])[0]
                input_values.append(float(encoded_value))
                input_display.append(selected_text)
            else:
                val = st.number_input(
                    f"输入 {feat}",
                    value=0.0,
                    step=0.1,
                    key=f"ridge_pred_num_{i}_{current_counter}",
                    format="%f"
                )
                input_values.append(val)
                input_display.append(f"{val}")

        # 预测按钮用当前计数器
        if st.button("🚀 开始预测", type="primary", use_container_width=True,
                     key=f"ridge_predict_btn_{current_counter}"):
            try:
                input_arr = np.array(input_values).reshape(1, -1)
                # 检查 scaler 是否存在
                if scaler is not None:
                    input_scaled = scaler.transform(input_arr)
                    pred = model.predict(input_scaled)[0]
                else:
                    pred = model.predict(input_arr)[0]
                    st.info("ℹ️ 使用原始数据进行预测（无标准化）")


                st.session_state.ridge_pred_result = {
                    'input_display': input_display,
                    'features': feature,
                    'prediction': pred
                }
                st.session_state.ridge_show_prediction = True
                # 增加计数器，下次进入时key会变化
                st.session_state.ridge_predict_counter += 1
                st.rerun()
            except Exception as e:
                st.error(f"❌ 预测出错：{str(e)}")

        if st.session_state.ridge_show_prediction and st.session_state.ridge_pred_result is not None:
            result = st.session_state.ridge_pred_result
            st.markdown("---")
            st.markdown("### 📝 输入特征：")
            for i, feat in enumerate(result['features']):
                st.write(f"- {feat}：{result['input_display'][i]}")
            st.markdown("### 🎯 预测结果")
            pred = result['prediction']
            st.success(f"**预测值：{pred:.4f}**")


    else:  # 上传文件模式

        st.write("上传文件进行批量预测：")

        uploader_key = f"ridge_pred_file_{st.session_state.get('ridge_batch_counter', 0)}"

        pred_file = st.file_uploader("选择预测文件", ['csv', 'xlsx'], key=uploader_key)

        if pred_file and st.session_state.ridge_batch_result is None:

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

                    if st.button("开始批量预测", type="primary", key="ridge_batch_predict"):

                        with st.spinner("正在预测中..."):

                            X_pred = df_pred[feature].values

                            if scaler:
                                X_pred = scaler.transform(X_pred)

                            preds = model.predict(X_pred)

                            result_df = df_pred.copy()

                            # ✅ Ridge回归总是回归问题，输出是一维或(n_samples,1)

                            if len(preds.shape) == 1:

                                result_df['预测值'] = preds

                            else:

                                result_df['预测值'] = preds.flatten()

                            st.session_state.ridge_batch_result = result_df

                            st.rerun()

            except Exception as e:

                st.error(f"❌ 预测出错：{str(e)}")

        if st.session_state.ridge_batch_result is not None:
            result_df = st.session_state.ridge_batch_result
            st.success(f"✅ 完成 {len(result_df)} 条数据预测")
            st.dataframe(result_df, use_container_width=True)
            csv_data = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载预测结果",
                data=csv_data,
                file_name="预测结果.csv",
                mime="text/csv",
                key="ridge_download_results"
            )

    st.markdown("---")

    # 操作按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 重新训练", use_container_width=True, key="ridge_retrain"):
            keys_to_remove = ['ridge_model', 'ridge_config', 'ridge_train_mae', 'ridge_test_mae',
                              'ridge_train_r2', 'ridge_test_r2', 'ridge_coefficients',
                              'ridge_X_test', 'ridge_y_test', 'ridge_y_pred_test',
                              'ridge_scaler', 'ridge_best_alpha', 'ridge_batch_result', 'ridge_pred_result']
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("🔮 新预测", use_container_width=True, key="ridge_new_prediction"):
            st.session_state.ridge_show_prediction = False
            st.session_state.ridge_pred_result = None
            st.session_state.ridge_batch_result = None
            st.session_state.ridge_manual_counter += 1
            if 'ridge_batch_counter' in st.session_state:
                st.session_state.ridge_batch_counter += 1
            st.rerun()
    with col3:
        if st.button("🏁 全新开始", use_container_width=True, key="ridge_fresh"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

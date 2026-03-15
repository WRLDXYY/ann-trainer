import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import tempfile
import os
import joblib  # 修改导入方式
import warnings

warnings.filterwarnings('ignore')


# ========== AI调用函数（为随机森林定制） ==========
def get_rf_ai_advice(data):
    """调用DeepSeek API获取随机森林模型评价和优化建议"""
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
        - 树的数量：{data['当前配置']['n_estimators']}
        - 最大深度：{data['当前配置']['max_depth']}
        - 最小分裂样本数：{data['当前配置']['min_samples_split']}
        - 最小叶子样本数：{data['当前配置']['min_samples_leaf']}
        - 最大特征数：{data['当前配置']['max_features']}
        - 测试集比例：{data['当前配置']['test_size']}%

        当前效果：
        - 准确率：{data['当前效果']['准确率']}

        【历史建议回顾】
        {st.session_state.ai_history[-1] if st.session_state.ai_history else '这是第一次咨询，还没有历史建议。'}

        【重要要求】
        1. 请基于以上**当前数据**（样本量{data['数据概况']['样本量']}行，特征数{data['数据概况']['特征数']}个）给出优化建议
        2. 不要直接复制历史建议，要根据当前配置和效果重新分析
        3. 如果准确率已经很高（>80%），可以建议微调；如果准确率低，建议大幅调整
        4. 考虑样本量大小：小样本（<500）可以减少树的数量防止过拟合

        如果是第二次或更多次咨询，请参考之前给的建议，告诉用户这次调整后的效果如何，下一步该怎么优化。

        请以JSON格式返回，只返回JSON：
        {{
            "评价": "结合基准和历史表现评价当前模型",
            "优化建议": {{
                "n_estimators": 建议值,
                "max_depth": 建议值,
                "min_samples_split": 建议值,
                "min_samples_leaf": 建议值,
                "max_features": "建议值",
                "test_size": 建议值（整数百分比）
            }},
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
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
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
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "test_size": 20
            },
            "预期效果": "优化后准确率预计可提升10-20%"
        }


# ========== 随机森林训练函数 ==========
def train_rf():
    """随机森林训练主函数"""
    st.subheader("🌲 随机森林训练")

    df = st.session_state.processed_df
    target = df.columns[-1]
    feature = df.columns[:-1].tolist()
    label_encoders = st.session_state.label_encoders
    is_classification = target in label_encoders

    # 定义随机种子
    random_state = 42

    # ===== 检查是否有AI建议的参数 =====
    if 'ai_suggested_params' in st.session_state:
        suggested = st.session_state.ai_suggested_params
        st.success("✨ 已应用AI建议的参数，你可以直接点击训练")

        default_n_estimators = int(suggested.get('n_estimators', 100))
        default_max_depth = int(suggested.get('max_depth', 10))
        default_min_samples_split = int(suggested.get('min_samples_split', 2))
        default_min_samples_leaf = int(suggested.get('min_samples_leaf', 1))
        default_max_features = suggested.get('max_features', 'sqrt')
        default_test_size = int(suggested.get('test_size', 20))
    else:
        default_n_estimators = 100
        default_max_depth = 10
        default_min_samples_split = 2
        default_min_samples_leaf = 1
        default_max_features = 'sqrt'
        default_test_size = 20

    if is_classification:
        le = label_encoders[target]
        num_classes = len(le.classes_)
    else:
        num_classes = None

    # 显示数据信息
    st.markdown("### 1. 数据识别")
    st.info(f"系统识别到：{len(feature)}个特征列，1个标签列")
    st.write(f"**标签列:** {target}")
    st.write(f"**问题类型:** {'分类' if is_classification else '回归'}")
    if is_classification:
        st.write(f"**类别数:** {num_classes}")

    # 参数配置
    st.markdown("### 2. 随机森林参数配置")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**基础参数**")
        n_estimators = st.number_input("树的数量", min_value=10, max_value=500,
                                       value=default_n_estimators, step=10, key="rf_n_estimators")

        max_depth = st.number_input("最大深度", min_value=1, max_value=50,
                                    value=default_max_depth, step=1, key="rf_max_depth")
        if max_depth == 50:
            max_depth = None

    with col2:
        st.write("**高级参数**")
        min_samples_split = st.number_input("最小分裂样本数", min_value=2, max_value=20,
                                            value=default_min_samples_split, key="rf_min_samples_split")
        min_samples_leaf = st.number_input("最小叶子样本数", min_value=1, max_value=20,
                                           value=default_min_samples_leaf, key="rf_min_samples_leaf")

        max_features = st.selectbox("最大特征数", ["sqrt", "log2", "None"],
                                    index=["sqrt", "log2", "None"].index(default_max_features),
                                    key="rf_max_features")
        if max_features == "None":
            max_features = None

    # 数据划分
    st.markdown("### 3. 数据划分")
    test_size = st.slider("测试集比例 (%)", 10, 40, value=int(default_test_size), key="rf_test_size")

    X = df[feature].values
    y = df[target].values

    # 确保 test_size 是 float 且在 0-1 之间
    try:
        test_size_float = float(test_size) / 100.0
        # 确保 test_size_float 在有效范围内
        if test_size_float <= 0 or test_size_float >= 1:
            test_size_float = 0.2
            st.warning(f"测试集比例 {test_size}% 无效，已自动调整为 20%")
    except:
        test_size_float = 0.2
        st.warning(f"测试集比例格式错误，已自动调整为 20%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_float, random_state=random_state,
        stratify=y if is_classification else None
    )

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
    if st.button("🚀 开始训练随机森林", type="primary", use_container_width=True, key="rf_train_btn"):

        # 保存配置
        st.session_state.rf_config = {
            'target': target,
            'feature': feature,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': random_state,
            'test_size': test_size / 100
        }
        st.session_state.is_classification = is_classification
        st.session_state.num_classes = num_classes
        st.session_state.label_encoder = le if is_classification else None
        st.session_state.rf_feature_names = feature

        # ✅ 添加 Class Weight 处理（仅分类问题）
        class_weight = None
        if is_classification and st.session_state.user_choices.get('balance') == 'class_weight':
            class_weight = 'balanced'
            st.info(f"📊 应用Class Weight：balanced")

        # 创建模型
        with st.spinner("模型训练中..."):
            start_time = time.time()

            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=max_depth,
                    min_samples_split=int(min_samples_split),
                    min_samples_leaf=int(min_samples_leaf),
                    max_features=max_features,
                    class_weight=class_weight,  # ✅ 添加 class_weight 参数
                    random_state=random_state,
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=max_depth,
                    min_samples_split=int(min_samples_split),
                    min_samples_leaf=int(min_samples_leaf),
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=-1
                )

            # 训练
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # 预测
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

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

            # 保存结果
            st.session_state.rf_model = model
            st.session_state.rf_train_score = train_score
            st.session_state.rf_test_score = test_score
            st.session_state.rf_train_time = train_time
            st.session_state.rf_feature_importance = feature_importance
            st.session_state.rf_X_test = X_test
            st.session_state.rf_y_test = y_test
            st.session_state.rf_y_pred_test = y_pred_test

            st.success(f"✅ 模型训练完成！ 训练时间: {train_time:.2f}秒")
            st.session_state.step = 4
            st.rerun()

    # 返回按钮
    if st.button("← 返回数据清洗", key="rf_back_clean"):
        st.session_state.step = 2
        st.rerun()


# ========== 随机森林预测函数 ==========
def predict_rf():
    """随机森林预测函数"""
    st.subheader("🔮 随机森林预测")
    if 'rf_predict_counter' not in st.session_state:
        st.session_state.rf_predict_counter = 0
    current_counter = st.session_state.rf_predict_counter

    # 判断模型来源：训练得到的还是上传的
    model = None
    config = None
    feature = None
    scaler = None  # 随机森林不需要scaler，但为了代码统一，仍然保留

    # 情况1：通过训练流程得到的模型（有完整配置）
    if 'rf_model' in st.session_state and st.session_state.rf_model is not None:
        model = st.session_state.rf_model
        if 'rf_config' in st.session_state and st.session_state.rf_config is not None:
            config = st.session_state.rf_config
            feature = config['feature']
            # ✅ 修改：随机森林不需要scaler，始终为None
            scaler = None
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
            scaler = None

    # 情况2：通过上传得到的模型
    elif 'uploaded_model' in st.session_state and st.session_state.uploaded_model is not None:
        if st.session_state.get('uploaded_model_type') == 'rf':
            model = st.session_state.uploaded_model
            st.info("📌 使用上传的随机森林模型进行预测")

            # ✅ 修改：随机森林不需要scaler
            scaler = None

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
            st.warning("请先训练模型或上传随机森林模型文件")
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

    # 显示配置信息（如果有）
    if config is not None:
        with st.expander("📋 当前模型配置", expanded=False):
            st.write(f"- 树的数量: {config.get('n_estimators')}")
            st.write(f"- 最大深度: {config.get('max_depth')}")
            st.write(f"- 最小分裂样本数: {config.get('min_samples_split')}")
            st.write(f"- 最小叶子样本数: {config.get('min_samples_leaf')}")
            st.write(f"- 最大特征数: {config.get('max_features')}")
            st.write(f"- 测试集比例: {config.get('test_size', 0.2) * 100:.0f}%")
    else:
        st.info("ℹ️ 当前为上传模型，仅支持预测功能")

    # 训练结果（仅当有训练结果时显示）
    if 'rf_test_score' in st.session_state:
        st.markdown("### 📊 训练结果")
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_name = "准确率" if is_classification else "MAE"
            st.metric(f"训练集{metric_name}", f"{st.session_state.rf_train_score:.4f}")
        with col2:
            st.metric(f"测试集{metric_name}", f"{st.session_state.rf_test_score:.4f}")
        with col3:
            st.metric("训练时间", f"{st.session_state.rf_train_time:.2f}秒")

        # 特征重要性可视化
        if 'rf_feature_importance' in st.session_state:
            st.markdown("### 📈 特征重要性")
            importance_dict = st.session_state.rf_feature_importance
            importance_df = pd.DataFrame({
                '特征': list(importance_dict.keys()),
                '重要性': list(importance_dict.values())
            }).sort_values('重要性', ascending=True)
            fig = px.bar(importance_df, x='重要性', y='特征', orientation='h',
                         title='特征重要性排序',
                         color='重要性', color_continuous_scale='Blues')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # AI建议按钮（仅当有配置时显示）
    if config is not None:
        col1, col2, col3 = st.columns(3)
        with col3:
            if st.button("🤖 AI评价及建议", use_container_width=True, key="rf_ai_btn"):
                if not st.session_state.get('ai_enabled', False):
                    st.error("❌ AI功能未启用，请检查API Key配置")
                    st.stop()

                test_size_percent = int(config.get('test_size', 0.2) * 100)
                accuracy = f"{st.session_state.rf_test_score:.2%}" if is_classification else "N/A"

                ai_input = {
                    "数据概况": {
                        "样本量": len(st.session_state.processed_df),
                        "特征数": len(feature),
                        "问题类型": "分类" if is_classification else "回归",
                        "类别数": num_classes if is_classification else "N/A"
                    },
                    "当前配置": {
                        "n_estimators": config.get('n_estimators'),
                        "max_depth": config.get('max_depth'),
                        "min_samples_split": config.get('min_samples_split'),
                        "min_samples_leaf": config.get('min_samples_leaf'),
                        "max_features": config.get('max_features'),
                        "test_size": test_size_percent
                    },
                    "当前效果": {
                        "准确率": accuracy
                    }
                }

                with st.spinner("AI正在分析模型表现..."):
                    advice = get_rf_ai_advice(ai_input)

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
                # 创建打包数据（随机森林不需要scaler）
                model_package = {
                    'model': st.session_state.rf_model,
                    'scaler': None,  # 随机森林不需要标准化
                    'config': config,
                    'model_type': 'rf',
                    'feature_names': feature,
                    'num_classes': num_classes if is_classification else None,
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
                    file_name=f"rf_model_package_{time.strftime('%Y%m%d_%H%M%S')}.pkl",
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
                file_name=f"rf_config_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    st.markdown("---")

    # ===== 预测功能 =====
    st.markdown("### 🔮 使用模型预测")

    # 初始化session state
    if 'rf_show_prediction' not in st.session_state:
        st.session_state.rf_show_prediction = False
    if 'rf_pred_result' not in st.session_state:
        st.session_state.rf_pred_result = None
    if 'rf_batch_result' not in st.session_state:
        st.session_state.rf_batch_result = None
    if 'rf_manual_counter' not in st.session_state:
        st.session_state.rf_manual_counter = 0

    # 预测方式选择
    input_method = st.radio(
        "选择输入方式",
        ["手动输入", "上传文件"],
        horizontal=True,
        key=f"rf_input_method_{current_counter}"
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
                    key=f"rf_pred_feat_{i}_{current_counter}"
                )

                encoded_value = le.transform([selected_text])[0]
                input_values.append(float(encoded_value))
                input_display.append(selected_text)
            else:
                val = st.number_input(
                    f"输入 {feat}",
                    value=0.0,
                    step=0.1,
                    key=f"rf_pred_num_{i}_{current_counter}",
                    format="%f"
                )
                input_values.append(val)
                input_display.append(f"{val}")

        # 预测按钮用当前计数器
        if st.button("🚀 开始预测", type="primary", use_container_width=True,
                     key=f"rf_predict_btn_{current_counter}"):
            try:
                input_arr = np.array(input_values).reshape(1, -1)
                # 检查 scaler 是否存在
                pred = model.predict(input_arr)[0]
                proba = model.predict_proba(input_arr)[0] if is_classification else None


                st.session_state.rf_pred_result = {
                    'input_display': input_display,
                    'features': feature,
                    'prediction': pred
                }
                st.session_state.rf_show_prediction = True
                # 增加计数器，下次进入时key会变化
                st.session_state.rf_predict_counter += 1
                st.rerun()
            except Exception as e:
                st.error(f"❌ 预测出错：{str(e)}")

        # 显示预测结果
        if st.session_state.rf_show_prediction and st.session_state.rf_pred_result is not None:
            result = st.session_state.rf_pred_result
            st.markdown("---")
            st.markdown("### 📝 输入特征：")
            for i, feat in enumerate(result['features']):
                st.write(f"- {feat}：{result['input_display'][i]}")

            st.markdown("### 🎯 预测结果")
            pred = result['prediction']

            if is_classification:
                pred_int = int(pred)
                if label_encoder and 0 <= pred_int < len(label_encoder.classes_):
                    res_text = label_encoder.inverse_transform([pred_int])[0]
                    st.success(f"**预测类别：{res_text}**")
                else:
                    st.success(f"**预测类别索引：{pred_int}**")

                # 显示概率
                if result.get('probabilities') is not None:
                    st.write("各类别概率：")
                    for i, p in enumerate(result['probabilities']):
                        if label_encoder and i < len(label_encoder.classes_):
                            class_name = label_encoder.inverse_transform([i])[0]
                        else:
                            class_name = f'类别{i}'
                        st.write(f"- {class_name}: {p:.2%}")
            else:
                st.success(f"**预测值：{pred:.4f}**")


    else:  # 上传文件模式

        st.write("上传文件进行批量预测：")

        uploader_key = f"rf_pred_file_{st.session_state.get('rf_batch_counter', 0)}"

        pred_file = st.file_uploader("选择预测文件", ['csv', 'xlsx'], key=uploader_key)

        if pred_file and st.session_state.rf_batch_result is None:

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

                    if st.button("开始批量预测", type="primary", key="rf_batch_predict"):

                        with st.spinner("正在预测中..."):

                            X_pred = df_pred[feature].values

                            preds = model.predict(X_pred)

                            result_df = df_pred.copy()

                            if is_classification:

                                if num_classes == 2:  # 二分类

                                    probas = model.predict_proba(X_pred)

                                    result_df['预测概率'] = probas[:, 1]

                                    if label_encoder:

                                        result_df['预测结果'] = [label_encoder.inverse_transform([int(p)])[0] for p in
                                                                 preds]

                                    else:

                                        result_df['预测结果'] = preds

                                else:  # 多分类

                                    probas = model.predict_proba(X_pred)

                                    if label_encoder:

                                        result_df['预测结果'] = [label_encoder.inverse_transform([int(p)])[0] for p in
                                                                 preds]

                                    else:

                                        result_df['预测结果'] = preds

                                    for i in range(probas.shape[1]):

                                        if label_encoder and i < len(label_encoder.classes_):

                                            class_name = label_encoder.inverse_transform([i])[0]

                                        else:

                                            class_name = f'类别{i}'

                                        result_df[f'{class_name}_概率'] = probas[:, i]

                            else:  # 回归

                                result_df['预测值'] = preds

                            st.session_state.rf_batch_result = result_df

                            st.rerun()

            except Exception as e:

                st.error(f"❌ 预测出错：{str(e)}")

        if st.session_state.rf_batch_result is not None:
            result_df = st.session_state.rf_batch_result
            st.success(f"✅ 完成 {len(result_df)} 条数据预测")
            st.dataframe(result_df, use_container_width=True)
            csv_data = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载预测结果",
                data=csv_data,
                file_name="预测结果.csv",
                mime="text/csv",
                key="rf_download_results"
            )

    st.markdown("---")

    # 操作按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 重新训练", use_container_width=True, key="rf_retrain"):
            keys_to_remove = ['rf_model', 'rf_config', 'rf_train_score', 'rf_test_score',
                              'rf_feature_importance', 'rf_X_test', 'rf_y_test', 'rf_y_pred_test',
                              'rf_batch_result', 'rf_pred_result']
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("🔮 新预测", use_container_width=True, key="rf_new_prediction"):
            st.session_state.rf_show_prediction = False
            st.session_state.rf_pred_result = None
            st.session_state.rf_batch_result = None
            st.session_state.rf_manual_counter += 1
            if 'rf_batch_counter' in st.session_state:
                st.session_state.rf_batch_counter += 1
            st.rerun()
    with col3:
        if st.button("🏁 全新开始", use_container_width=True, key="rf_fresh"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
import time
import json
import tempfile
import os
from openai import OpenAI
import warnings
import re
warnings.filterwarnings('ignore')


# ========== AI调用函数 ==========
def get_ai_advice(data):
    """调用DeepSeek API获取模型评价和优化建议"""
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
- dropout：{data['当前配置']['dropout']}
- 优化器：{data['当前配置']['优化器']}
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
        "dropout": 建议值,     
        "优化器": "建议值",
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
                    "隐藏层": 2,
                    "神经元": 64,
                    "激活函数": "relu",
                    "dropout": 0.0,
                    "优化器": "Adam",
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
                "dropout": 0.0,
                "优化器": "Adam",
                "学习率": 0.001,
                "批次大小": 32,
                "训练轮次": 100,
                "测试集比例": 20
            },
            "预期效果": "优化后准确率预计可提升10-20%"
        }


# ========== ANN训练函数 ==========
def train_ann():
    """神经网络训练主函数"""
    st.subheader("🤖 神经网络训练")

    df = st.session_state.processed_df
    target = df.columns[-1]
    feature = df.columns[:-1].tolist()
    label_encoders = st.session_state.label_encoders
    is_classification = target in label_encoders

    # ===== 检查是否有AI建议的参数 =====
    if 'ai_suggested_params' in st.session_state:
        suggested = st.session_state.ai_suggested_params
        st.success("✨ 已应用AI建议的参数，你可以直接点击训练")

        default_hidden_layers = int(suggested.get('隐藏层', 2))
        default_neurons = int(suggested.get('神经元', 64))
        default_activation = suggested.get('激活函数', 'relu')
        default_dropout = float(suggested.get('dropout', 0.0))
        default_optimizer = suggested.get('优化器', 'Adam')
        default_learning_rate = float(suggested.get('学习率', 0.001))
        default_batch_size = int(suggested.get('批次大小', 32))
        default_epochs = int(suggested.get('训练轮次', 50))
        default_test_size = int(suggested.get('test_size', 20))
    else:
        default_hidden_layers = 2
        default_neurons = 64
        default_activation = 'relu'
        default_dropout = 0.0
        default_optimizer = 'Adam'
        default_learning_rate = 0.001
        default_batch_size = 32
        default_epochs = 50
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
    st.markdown("### 2. ANN模型配置")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**网络结构**")
        hidden_layers = st.number_input("隐藏层层数", min_value=1, max_value=10,
                                        value=default_hidden_layers, step=1, key="ann_hidden_layers")
        neurons_per_layer = st.number_input("每层神经元数", min_value=4, max_value=512,
                                            value=default_neurons, step=8, key="ann_neurons")
        activation = st.selectbox("激活函数", ["relu", "sigmoid", "tanh"],
                                  index=["relu", "sigmoid", "tanh"].index(default_activation), key="ann_activation")
        dropout_rate = st.number_input("Dropout比例", min_value=0.0, max_value=0.5,
                                       value=default_dropout, step=0.05, key="ann_dropout")

    with col2:
        st.write("**训练参数**")
        epochs = st.number_input("训练轮次", min_value=1, max_value=1000,
                                 value=default_epochs, step=10, key="ann_epochs")
        batch_size = st.number_input("批次大小", min_value=1, max_value=256,
                                     value=default_batch_size, step=4, key="ann_batch_size")
        learning_rate = st.number_input("学习率", min_value=0.00001, max_value=1.0,
                                        value=default_learning_rate, format="%.5f", key="ann_lr")
        optimizer_choice = st.selectbox("优化器", ["Adam", "SGD"],
                                        index=0 if default_optimizer == "Adam" else 1, key="ann_optimizer")

    # 数据划分
    st.markdown("### 3. 数据划分")
    test_size = st.slider("测试集比例 (%)", 10, 40, value=int(default_test_size), key="ann_test_size")

    X = df[feature].values
    y_original = df[target].values

    # 正确处理多分类标签
    if is_classification and num_classes > 2:
        y = tf.keras.utils.to_categorical(y_original, num_classes=num_classes)
        stratify = None
    else:
        y = y_original
        stratify = y_original if is_classification else None

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
        stratify=stratify
    )

    if is_classification and num_classes > 2:
        try:
            test_size_float = float(test_size) / 100.0
            if test_size_float <= 0 or test_size_float >= 1:
                test_size_float = 0.2
        except:
            test_size_float = 0.2
            st.warning(f"测试集比例格式错误，已自动调整为 20%")
        _, y_test_original = train_test_split(
            y_original, test_size=test_size_float, random_state=42,
            stratify=y_original
        )
    else:
        y_test_original = y_test

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
    if st.button("🚀 开始训练神经网络", type="primary", use_container_width=True, key="ann_train_btn"):
        # 保存配置
        st.session_state.model_config = {
            'target': target,
            'feature': feature,
            'hidden_layers': hidden_layers,
            'neurons': neurons_per_layer,
            'activation': activation,
            'dropout': dropout_rate,
            'optimizer': optimizer_choice,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'test_size': test_size / 100
        }
        st.session_state.is_classification = is_classification
        st.session_state.num_classes = num_classes
        st.session_state.label_encoder = le if is_classification else None
        st.session_state.scaler = scaler
        st.session_state.X_test = X_test_scaled
        st.session_state.y_test = y_test
        st.session_state.y_test_original = y_test_original

        # 构建模型 - 添加类型转换
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)))

        # 确保整数类型
        hidden_layers = int(hidden_layers)
        neurons_per_layer = int(neurons_per_layer)

        for _ in range(hidden_layers):
            model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation))
            if dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(dropout_rate))

        # 输出层
        if is_classification:
            if num_classes == 2:
                model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
                metric = 'accuracy'
            else:
                model.add(tf.keras.layers.Dense(int(num_classes), activation='softmax'))  # 确保是int
                loss = 'categorical_crossentropy'
                metric = 'accuracy'
        else:
            model.add(tf.keras.layers.Dense(1, activation='linear'))
            loss = 'mse'
            metric = 'mae'

        # 优化器 - 确保学习率是float
        learning_rate = float(learning_rate)
        if optimizer_choice == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

        # 编译模型
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        # 训练参数 - 确保是整数
        epochs = int(epochs)
        batch_size = int(batch_size)

        # Class Weight处理
        class_weight_dict = None
        if st.session_state.user_choices.get('balance') == 'class_weight' and is_classification and num_classes == 2:
            try:
                y_train_labels = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
                classes = np.unique(y_train_labels)
                weights = compute_class_weight('balanced', classes=classes, y=y_train_labels)
                class_weight_dict = dict(zip(classes, weights))
                st.info(f"📊 应用Class Weight：{class_weight_dict}")
            except Exception as e:
                st.warning(f"Class Weight计算失败：{str(e)}")

        # 验证集策略
        n_samples = len(X_train)
        if n_samples < 500:
            validation_split = 0.0
            st.info(f"📌 样本量较少({n_samples}行)，不使用验证集")
        elif n_samples < 2000:
            validation_split = 0.1
            st.info(f"📌 样本量适中({n_samples}行)，使用10%验证集")
        else:
            validation_split = 0.15
            st.info(f"📌 样本量充足({n_samples}行)，使用15%验证集")

        callbacks = None
        if validation_split > 0:
            callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]

        # 训练
        with st.spinner("模型训练中..."):
            history = model.fit(
                X_train_scaled, y_train,
                class_weight=class_weight_dict,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1,
                callbacks=callbacks
            )

        st.session_state.model = model
        st.session_state.history = history.history
        st.success("✅ 模型训练完成！")
        st.session_state.step = 4
        st.rerun()

    # 返回按钮
    if st.button("← 返回数据清洗", key="ann_back_clean"):
        st.session_state.ai_messages = [
            {"role": "system", "content": "你是一个机器学习调参专家，请基于对话历史给出连贯的建议。"}
        ]
        st.session_state.ai_history = []
        st.session_state.ai_advice = None
        st.session_state.step = 2
        st.rerun()


# ========== ANN预测函数 ==========
def predict_ann():
    """神经网络预测函数"""
    st.subheader("🔮 神经网络预测")
    if 'ann_predict_counter' not in st.session_state:
        st.session_state.ann_predict_counter = 0
    current_counter = st.session_state.ann_predict_counter

    # 判断模型来源：训练得到的还是上传的
    model = None
    config = None
    feature = None
    scaler = None

    # 情况1：通过训练流程得到的模型（有完整配置）
    if 'model' in st.session_state and st.session_state.model is not None:
        model = st.session_state.model
        if 'model_config' in st.session_state and st.session_state.model_config is not None:
            config = st.session_state.model_config
            feature = config['feature']
            # ✅ 修改：优先使用 ann_scaler，如果没有则用 scaler
            scaler = st.session_state.get('ann_scaler', st.session_state.get('scaler', None))
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
            scaler = st.session_state.get('ann_scaler', None)

    # 情况2：通过上传得到的模型
    elif 'uploaded_model' in st.session_state and st.session_state.uploaded_model is not None:
        if st.session_state.get('uploaded_model_type') == 'ann':
            model = st.session_state.uploaded_model
            st.info("📌 使用上传的神经网络模型进行预测")

            # ✅ 修改：从上传的模型获取scaler
            scaler = st.session_state.get('ann_scaler', st.session_state.get('uploaded_scaler', None))

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
            st.warning("请先训练模型或上传神经网络模型文件")
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

    # 如果 feature 还是 None，说明有问题
    if feature is None:
        st.error("无法确定特征信息，请返回重新训练或上传模型")
        return

    # 显示配置信息（仅当有配置时）
    if config is not None:
        with st.expander("📋 当前模型配置", expanded=False):
            st.write(f"- 隐藏层: {config.get('hidden_layers')}层")
            st.write(f"- 神经元: {config.get('neurons')}个/层")
            st.write(f"- 激活函数: {config.get('activation')}")
            st.write(f"- Dropout: {config.get('dropout')}")
            st.write(f"- 优化器: {config.get('optimizer')}")
            st.write(f"- 学习率: {config.get('learning_rate')}")
            st.write(f"- 批次大小: {config.get('batch_size')}")
            st.write(f"- 训练轮次: {config.get('epochs')}")
            st.write(f"- 测试集比例: {config.get('test_size', 0.2) * 100:.0f}%")
    else:
        st.info("ℹ️ 当前为上传模型，仅支持预测功能")

    # 训练结果
    if 'history' in st.session_state and st.session_state.history:
        st.markdown("### 📈 训练结果")
        tab1, tab2, tab3, tab4 = st.tabs(["训练曲线", "评估指标", "分类报告", "模型结构"])

        with tab1:
            fig = go.Figure()
            loss_history = st.session_state.history.get('loss', [])
            fig.add_trace(
                go.Scatter(x=list(range(1, len(loss_history) + 1)), y=loss_history, mode='lines', name='训练损失',
                           line=dict(color='blue')))

            val_loss = st.session_state.history.get('val_loss')
            if val_loss and len(val_loss) > 0:
                fig.add_trace(go.Scatter(x=list(range(1, len(val_loss) + 1)), y=val_loss, mode='lines', name='验证损失',
                                         line=dict(color='red', dash='dash')))

            fig.update_layout(title='训练损失曲线', xaxis_title='训练轮次', yaxis_title='损失值')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if hasattr(st.session_state, 'X_test') and hasattr(st.session_state, 'y_test'):
                test_loss, test_metric = model.evaluate(st.session_state.X_test, st.session_state.y_test, verbose=0)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("测试集损失", f"{test_loss:.4f}")
                with col2:
                    metric_name = "准确率" if is_classification else "MAE"
                    st.metric(f"测试集{metric_name}", f"{test_metric:.4f}")
                with col3:
                    st.metric("训练轮次", config.get('epochs', 50))

        with tab3:
            if is_classification and hasattr(st.session_state,
                                             'y_test_original') and st.session_state.y_test_original is not None:
                try:
                    y_pred = model.predict(st.session_state.X_test, verbose=0)
                    if num_classes == 2:
                        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
                    else:
                        y_pred_classes = np.argmax(y_pred, axis=1)

                    target_names = label_encoder.classes_ if label_encoder else None
                    report = classification_report(
                        st.session_state.y_test_original,
                        y_pred_classes,
                        target_names=target_names,
                        output_dict=True
                    )
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.2f}"))
                except Exception as e:
                    st.info(f"无法生成分类报告：{str(e)}")
            elif not is_classification:
                st.info("回归问题不生成分类报告")

        with tab4:
            st.write("**模型结构:**")
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text("\n".join(model_summary))
            st.metric("总参数量", f"{model.count_params():,}")

    st.markdown("---")

    # AI建议按钮
    col1, col2, col3 = st.columns(3)
    with col3:
        if st.button("🤖 AI评价及建议", use_container_width=True, key="ann_ai_btn"):
            if not st.session_state.get('ai_enabled', False):
                st.error("❌ AI功能未启用，请检查API Key配置")
                st.stop()

            test_size_value = config.get('test_size', 0.2)
            test_size_percent = int(test_size_value * 100)

            if is_classification:
                if hasattr(st.session_state, 'X_test') and hasattr(st.session_state, 'y_test'):
                    test_loss, test_metric = model.evaluate(st.session_state.X_test, st.session_state.y_test,
                                                            verbose=0)
                    accuracy = f"{test_metric:.2%}"
                else:
                    accuracy = "N/A"
            else:
                accuracy = "N/A"

            ai_input = {
                "数据概况": {
                    "样本量": len(st.session_state.processed_df),
                    "特征数": len(feature),
                    "问题类型": "分类" if is_classification else "回归",
                    "类别数": num_classes if is_classification else "N/A"
                },
                "当前配置": {
                    "隐藏层": config.get('hidden_layers'),
                    "神经元": config.get('neurons'),
                    "激活函数": config.get('activation'),
                    "dropout": config.get('dropout', 0.0),
                    "优化器": config.get('optimizer', 'Adam'),
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

            with st.spinner("AI正在分析模型表现..."):
                advice = get_ai_advice(ai_input)

            if advice:
                st.session_state.ai_advice = advice
                st.session_state.ai_advice_generated = True
                st.rerun()

    # AI提示信息
    if st.session_state.get('ai_advice_generated', False):
        st.success("✅ AI建议已生成！请查看左侧边栏下方🤖 AI模型建议，并点击✨ 应用AI建议参数。")
        st.info("💡 提示：请根据建议在步骤3中手动调整参数")

    # 模型保存（仅当有配置时显示训练好的模型，上传的模型不显示下载按钮）
    if config is not None:
        col1, col2 = st.columns(2)
        with col1:
            tmp_file_path = None
            model_bytes = None
            try:
                # 创建打包数据
                model_package = {
                    'model': model,
                    'scaler': scaler,
                    'config': config,
                    'model_type': 'ann',
                    'feature_names': feature,
                    'is_classification': is_classification,
                    'num_classes': num_classes,
                    'label_encoder_classes': label_encoder.classes_.tolist() if label_encoder else None
                }

                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    import joblib
                    joblib.dump(model_package, tmp_file.name)
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name
                    with open(tmp_file.name, 'rb') as f:
                        model_bytes = f.read()
            except Exception as e:
                st.error(f"❌ 模型保存失败：{str(e)}")
            finally:
                if tmp_file_path and os.path.exists(tmp_file_path):
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass

            if model_bytes is not None:
                st.download_button(
                    label="📥 下载完整模型包 (.pkl)",
                    data=model_bytes,
                    file_name=f"ann_model_package_{time.strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )

        with col2:
            scaler_info = {
                'mean': scaler.mean_.tolist() if scaler and hasattr(scaler, 'mean_') else None,
                'scale': scaler.scale_.tolist() if scaler and hasattr(scaler, 'scale_') else None,
                'feature_names': feature,
                'target_name': config.get('target') if config else None,
                'is_classification': is_classification,
                'num_classes': num_classes if is_classification else None,
                'label_encoder_classes': label_encoder.classes_.tolist() if label_encoder else None
            }
            scaler_json = json.dumps(scaler_info, indent=2, ensure_ascii=False)
            st.download_button(
                label="📄 下载配置 (.json)",
                data=scaler_json,
                file_name=f"model_config_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.info("ℹ️ 上传的模型不支持下载配置信息")

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
    if 'ann_show_prediction' not in st.session_state:
        st.session_state.ann_show_prediction = False
    if 'ann_pred_result' not in st.session_state:
        st.session_state.ann_pred_result = None
    if 'ann_batch_result' not in st.session_state:
        st.session_state.ann_batch_result = None
    if 'ann_manual_counter' not in st.session_state:
        st.session_state.ann_manual_counter = 0

    # 预测方式选择
    input_method = st.radio(
        "选择输入方式",
        ["手动输入", "上传文件"],
        horizontal=True,
        key=f"ann_input_method_{current_counter}"
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
                    key=f"ann_pred_feat_{i}_{current_counter}"
                )

                encoded_value = le.transform([selected_text])[0]
                input_values.append(float(encoded_value))
                input_display.append(selected_text)
            else:
                val = st.number_input(
                    f"输入 {feat}",
                    value=0.0,
                    step=0.1,
                    key=f"ann_pred_num_{i}_{current_counter}",
                    format="%f"
                )
                input_values.append(val)
                input_display.append(f"{val}")

        # 预测按钮用当前计数器
        if st.button("🚀 开始预测", type="primary", use_container_width=True,
                     key=f"ann_predict_btn_{current_counter}"):
            try:
                input_arr = np.array(input_values).reshape(1, -1)
                # 检查 scaler 是否存在
                if scaler is not None:
                    input_scaled = scaler.transform(input_arr)
                    pred = model.predict(input_scaled, verbose=0)
                else:
                    pred = model.predict(input_arr, verbose=0)
                    st.info("ℹ️ 使用原始数据进行预测（无标准化）")


                st.session_state.ann_pred_result = {
                    'input_display': input_display,
                    'features': feature,
                    'prediction': pred
                }
                st.session_state.ann_show_prediction = True
                # 增加计数器，下次进入时key会变化
                st.session_state.ann_predict_counter += 1
                st.rerun()
            except Exception as e:
                st.error(f"❌ 预测出错：{str(e)}")

        if st.session_state.get('ann_show_prediction', False) and st.session_state.ann_pred_result is not None:
            result = st.session_state.ann_pred_result
            st.markdown("---")
            st.markdown("### 📝 输入特征：")
            for i, feat in enumerate(result['features']):
                st.write(f"- {feat}：{result['input_display'][i]}")

            st.markdown("### 🎯 预测结果")
            pred = result['prediction']

            if is_classification:
                if num_classes == 2:
                    prob = pred[0][0]
                    label = 1 if prob > 0.5 else 0
                    if label_encoder and 0 <= label < len(label_encoder.classes_):
                        res_text = label_encoder.inverse_transform([label])[0]
                    else:
                        res_text = str(label)
                    st.success(f"**预测类别：{res_text}**")
                    st.progress(float(prob))
                    st.write(f"置信度：{prob:.2%}")
                else:
                    cls_idx = np.argmax(pred[0])
                    if label_encoder and 0 <= cls_idx < len(label_encoder.classes_):
                        res_text = label_encoder.inverse_transform([cls_idx])[0]
                    else:
                        res_text = str(cls_idx)
                    st.success(f"**预测类别：{res_text}**")
                    st.write("各类别概率：")
                    for i, prob in enumerate(pred[0]):
                        if label_encoder and i < len(label_encoder.classes_):
                            class_name = label_encoder.inverse_transform([i])[0]
                        else:
                            class_name = f"类别{i}"
                        st.write(f"- {class_name}: {prob:.2%}")
            else:
                st.success(f"**预测值：{pred[0][0]:.4f}**")



    else:  # 上传文件模式

        st.write("上传文件进行批量预测：")

        # 使用计数器刷新文件上传器

        uploader_key = f"ann_pred_file_{st.session_state.get('ann_batch_counter', 0)}"

        pred_file = st.file_uploader("选择预测文件", ['csv', 'xlsx'], key=uploader_key)

        if pred_file and st.session_state.ann_batch_result is None:

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

                    if st.button("开始批量预测", type="primary", key="ann_batch_predict"):

                        with st.spinner("正在预测中..."):

                            X_pred = df_pred[feature].values

                            if scaler is not None:

                                X_pred_scaled = scaler.transform(X_pred)

                                preds = model.predict(X_pred_scaled, verbose=0)

                            else:

                                preds = model.predict(X_pred, verbose=0)

                                st.info("ℹ️ 使用原始数据进行预测（无标准化）")

                            result_df = df_pred.copy()

                            # ✅ 通用维度处理

                            n_rows = len(result_df)

                            # 检查预测结果维度

                            if len(preds.shape) == 1:  # 1维数组 (n_samples,)

                                # 回归或二分类的概率输出

                                if is_classification and num_classes == 2:

                                    result_df['预测概率'] = preds

                                    result_df['预测结果'] = result_df['预测概率'].apply(

                                        lambda x: label_encoder.inverse_transform([1])[
                                            0] if x > 0.5 and label_encoder else "1"

                                    )

                                    result_df['预测结果'] = result_df.apply(

                                        lambda row: label_encoder.inverse_transform([0])[0] if row[
                                                                                                   '预测概率'] <= 0.5 and label_encoder else
                                        row['预测结果'],

                                        axis=1

                                    )

                                else:  # 回归

                                    result_df['预测值'] = preds


                            elif len(preds.shape) == 2:  # 2维数组

                                if preds.shape[1] == 1:  # (n_samples, 1)

                                    # 二分类或回归

                                    preds_flat = preds.flatten()

                                    if is_classification and num_classes == 2:

                                        result_df['预测概率'] = preds_flat

                                        result_df['预测结果'] = result_df['预测概率'].apply(

                                            lambda x: label_encoder.inverse_transform([1])[
                                                0] if x > 0.5 and label_encoder else "1"

                                        )

                                        result_df['预测结果'] = result_df.apply(

                                            lambda row: label_encoder.inverse_transform([0])[0] if row[
                                                                                                       '预测概率'] <= 0.5 and label_encoder else
                                            row['预测结果'],

                                            axis=1

                                        )

                                    else:  # 回归

                                        result_df['预测值'] = preds_flat


                                else:  # (n_samples, n_classes) 多分类

                                    # 取每个样本的最大概率类别

                                    pred_classes = np.argmax(preds, axis=1)

                                    if label_encoder:

                                        result_df['预测结果'] = [

                                            label_encoder.inverse_transform([p])[0] for p in pred_classes

                                        ]

                                    else:

                                        result_df['预测结果'] = pred_classes

                                    # 添加各类别的概率列

                                    for i in range(preds.shape[1]):

                                        if label_encoder and i < len(label_encoder.classes_):

                                            class_name = label_encoder.inverse_transform([i])[0]

                                        else:

                                            class_name = f'类别{i}'

                                        result_df[f'{class_name}_概率'] = preds[:, i]

                            else:

                                st.error(f"❌ 不支持的预测结果维度: {preds.shape}")

                            st.session_state.ann_batch_result = result_df

                            st.rerun()

            except Exception as e:

                st.error(f"❌ 预测出错：{str(e)}")

                import traceback

                st.code(traceback.format_exc())

        # 显示批量预测结果
        if st.session_state.ann_batch_result is not None:
            result_df = st.session_state.ann_batch_result
            st.success(f"✅ 完成 {len(result_df)} 条数据预测")
            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载预测结果",
                data=csv_data,
                file_name="预测结果.csv",
                mime="text/csv",
                key="ann_download_results"
            )

    st.markdown("---")

    # 操作按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 重新训练", use_container_width=True, key="ann_retrain"):
            keys_to_remove = ['model', 'scaler', 'history', 'X_test', 'y_test', 'y_test_original', 'label_encoder']
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("🔮 新预测", use_container_width=True, key="ann_new_prediction"):
            st.session_state.ann_show_prediction = False
            st.session_state.ann_pred_result = None
            st.session_state.ann_batch_result = None
            st.session_state.ann_manual_counter += 1
            if 'ann_batch_counter' in st.session_state:
                st.session_state.ann_batch_counter += 1
            st.rerun()
    with col3:
        if st.button("🏁 全新开始", use_container_width=True, key="ann_fresh"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

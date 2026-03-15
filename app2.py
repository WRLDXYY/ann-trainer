import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import time
import warnings
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import tensorflow as tf

# 页面配置
st.set_page_config(page_title="🧠 小屿屿训练器", layout="wide")
warnings.filterwarnings('ignore')

# 字体设置
font_path = os.path.join(os.path.dirname(__file__), 'simhei.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name(), 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ========== 初始化session state ==========
def init_session_state():
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
        'model_upload_counter': 0,
        'model_upload_processed': False,
        'warnings': [],
        'detection_results': {},
        'original_df': None,
        'processed_df': None,
        'temp_df': None,
        'uploaded_model': None,
        'uploaded_scaler': None,
        'uploaded_config': None,
        'uploaded_model_type': None,
        'model_loaded': False,
        'ai_enabled': False,
        'ai_advice': None,
        'ai_messages': None,
        'ai_history': [],
        'ai_advice_generated': False,
        'selected_model': '神经网络 ANN',
        'user_choices': {
            'missing': None,
            'outlier': None,
            'balance': None,
            'features': None,
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
    st.success("✅你好，我是小屿屿")
    if st.session_state.ai_messages is None:
        st.session_state.ai_messages = [
            {"role": "system", "content": "你是一个机器学习调参专家，请基于对话历史给出连贯的建议。"}
        ]
    if st.session_state.ai_history is None:
        st.session_state.ai_history = []
else:
    st.session_state.ai_enabled = False
    st.session_state.ai_advice = None

# ========== 导入模块 ==========
from utils.data_cleaner import run_data_cleaning
from models.ann_trainer import train_ann, predict_ann
from models.rf_trainer import train_rf, predict_rf
from models.lr_trainer import train_lr, predict_lr
from models.lgb_trainer import train_lgb, predict_lgb
from models.ridge_trainer import train_ridge, predict_ridge

# ========== 侧边栏 ==========
with st.sidebar:
    st.title("🧠 小屿屿训练器")
    st.markdown("---")

    # 重要提示
    st.warning(
        "⚠️ **重要提示**：上传的文件不能提前编码！\n\n请保持原始数据（文本类别请保留为字符串），系统会自动进行编码处理。"
    )
    st.markdown("---")

    # 文件上传
    uploaded_file = st.file_uploader("上传数据文件", type=['csv', 'xlsx', 'xls'], key="data_uploader")

    # 模型上传功能
    st.markdown("### 🤖 上传训练好的模型")
    st.info("上传模型文件前，请先完成数据上传和编码")

    # 显示当前数据状态
    if st.session_state.df is not None:
        st.success("✅ 数据已就绪，可以上传模型")
    else:
        st.warning("⚠️ 请先上传并编码数据文件")

    # 添加一个处理标记，避免重复处理
    if 'model_upload_processed' not in st.session_state:
        st.session_state.model_upload_processed = False

    # 在侧边栏，修改模型上传部分
    uploaded_model_file = st.file_uploader(
        "上传模型包文件 (.pkl)",  # 改为只接受 .pkl
        type=['pkl'],
        key=f"model_uploader_{st.session_state.get('model_upload_counter', 0)}"
    )

    # 逻辑1：仅上传模型文件 → 跳至步骤4
    if uploaded_model_file is not None and not st.session_state.model_upload_processed:
        # 检查是否已有编码后的数据
        if st.session_state.df is None:
            st.error("❌ 请先上传并编码数据文件！")
            st.session_state.model_upload_processed = False
            st.stop()

        # 立即标记为已处理，防止重复
        st.session_state.model_upload_processed = True

        tmp_file_path = None
        model = None
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix=os.path.splitext(uploaded_model_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_model_file.getvalue())
                tmp_file.flush()
                tmp_file_path = tmp_file.name

                # 加载模型包
                import joblib

                model_package = joblib.load(tmp_file_path)

                # 从包中解出所有组件
                model = model_package['model']
                scaler = model_package.get('scaler')  # ✅ 使用get，避免KeyError
                config = model_package['config']
                model_type = model_package['model_type']
                feature_names = model_package['feature_names']
                label_encoder_classes = model_package.get('label_encoder_classes')

                # 根据模型类型设置 - 为每种模型类型分别设置scaler
                if model_type == 'lr':
                    st.session_state.selected_model = '逻辑回归 LR'
                    st.session_state.lr_model = model
                    st.session_state.lr_scaler = scaler  # ✅ 即使是None也要保存
                    st.session_state.lr_config = config
                elif model_type == 'ann':
                    st.session_state.selected_model = '神经网络 ANN'
                    st.session_state.ann_model = model
                    st.session_state.ann_scaler = scaler  # ✅ 即使是None也要保存
                    st.session_state.ann_config = config
                elif model_type == 'ridge':
                    st.session_state.selected_model = 'Ridge回归'
                    st.session_state.ridge_model = model
                    st.session_state.ridge_scaler = scaler  # ✅ 即使是None也要保存
                    st.session_state.ridge_config = config
                elif model_type == 'rf':
                    st.session_state.selected_model = '随机森林 RF'
                    st.session_state.rf_model = model
                    # 随机森林不需要scaler，但为了统一，也保存为None
                    st.session_state.rf_scaler = None
                    st.session_state.rf_config = config
                elif model_type == 'lgb':
                    st.session_state.selected_model = 'LightGBM'
                    st.session_state.lgb_model = model
                    st.session_state.lgb_scaler = scaler  # ✅ 即使是None也要保存
                    st.session_state.lgb_config = config

                # 设置通用字段
                st.session_state.uploaded_model = model
                st.session_state.uploaded_model_type = model_type
                st.session_state.feature_names = feature_names
                st.session_state.uploaded_scaler = scaler  # ✅ 保存scaler到通用字段

                # 如果有标签编码器，重建
                if label_encoder_classes:
                    from sklearn.preprocessing import LabelEncoder

                    le = LabelEncoder()
                    le.classes_ = np.array(label_encoder_classes)
                    st.session_state.label_encoder = le

                st.session_state.model_loaded = True
                st.session_state.model_upload_counter += 1
                st.session_state.step = 4
                st.success(f"✅ {st.session_state.selected_model} 模型包已加载！")
                time.sleep(1)
                st.rerun()

        except Exception as e:
            st.error(f"❌ 模型包加载失败：{str(e)}")
            st.session_state.model_upload_processed = False
            st.stop()
        finally:
            # 清理临时文件
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

    # 逻辑2：上传数据文件（独立于模型上传）
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='gbk')
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.raw_df = df
        st.session_state.current_file = uploaded_file.name
        st.success(f"✅ 文件已上传：{df.shape[0]}行 × {df.shape[1]}列")

    # 重置模型上传处理标记
    if uploaded_model_file is None:
        st.session_state.model_upload_processed = False

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

    # AI模型建议区域
    st.markdown("### 🤖 AI模型建议")

    if not st.session_state.get('ai_enabled', False):
        st.caption("⚠️ AI功能未启用")
    elif st.session_state.ai_advice is not None:
        advice = st.session_state.ai_advice
        st.info(f"📊 **当前模型评估**: {advice['评价']}")

        st.markdown("**💡 推荐配置:**")

        # 根据当前模型显示不同的参数
        if st.session_state.selected_model == "神经网络 ANN":
            test_size = advice['优化建议'].get('测试集比例', 20)
            if isinstance(test_size, float) and test_size < 1:
                test_size_display = int(test_size * 100)
            else:
                test_size_display = test_size

            st.markdown(f"""
                - 隐藏层: {advice['优化建议'].get('隐藏层', 2)}层
                - 神经元: {advice['优化建议'].get('神经元', 64)}个
                - 激活函数: {advice['优化建议'].get('激活函数', 'relu')}
                - Dropout: {advice['优化建议'].get('dropout', 0.0)}
                - 优化器: {advice['优化建议'].get('优化器', 'Adam')}
                - 学习率: {advice['优化建议'].get('学习率', 0.001)}
                - 批次大小: {advice['优化建议'].get('批次大小', 32)}
                - 训练轮次: {advice['优化建议'].get('训练轮次', 100)}
                - 测试集比例: {test_size_display}%
                """)

        elif st.session_state.selected_model == "逻辑回归 LR":
            test_size = advice['优化建议'].get('test_size', 20)
            if isinstance(test_size, float) and test_size < 1:
                test_size_display = int(test_size * 100)
            else:
                test_size_display = test_size

            st.markdown(f"""
                - C值（正则化强度）: {advice['优化建议'].get('C', 1.0)}
                - 正则化类型: {advice['优化建议'].get('penalty', 'l2')}
                - 求解器: {advice['优化建议'].get('solver', 'lbfgs')}
                - 最大迭代次数: {advice['优化建议'].get('max_iter', 100)}
                - 测试集比例: {test_size_display}%
                """)

        elif st.session_state.selected_model == "随机森林 RF":
            test_size = advice['优化建议'].get('test_size', 20)
            if isinstance(test_size, float) and test_size < 1:
                test_size_display = int(test_size * 100)
            else:
                test_size_display = test_size

            st.markdown(f"""
                - 树的数量: {advice['优化建议'].get('n_estimators', 100)}
                - 最大深度: {advice['优化建议'].get('max_depth', 10)}
                - 最小分裂样本数: {advice['优化建议'].get('min_samples_split', 2)}
                - 最小叶子样本数: {advice['优化建议'].get('min_samples_leaf', 1)}
                - 最大特征数: {advice['优化建议'].get('max_features', 'sqrt')}
                - 测试集比例: {test_size_display}%
                """)

        elif st.session_state.selected_model == "LightGBM":
            test_size = advice['优化建议'].get('test_size', 20)
            if isinstance(test_size, float) and test_size < 1:
                test_size_display = int(test_size * 100)
            else:
                test_size_display = test_size

            st.markdown(f"""
                - 提升迭代次数: {advice['优化建议'].get('n_estimators', 100)}
                - 学习率: {advice['优化建议'].get('learning_rate', 0.1)}
                - 最大深度: {advice['优化建议'].get('max_depth', -1)}
                - 叶子节点数: {advice['优化建议'].get('num_leaves', 31)}
                - 最小叶子样本数: {advice['优化建议'].get('min_child_samples', 20)}
                - 样本采样比例: {advice['优化建议'].get('subsample', 1.0)}
                - 特征采样比例: {advice['优化建议'].get('colsample_bytree', 1.0)}
                - L1正则化: {advice['优化建议'].get('reg_alpha', 0.0)}
                - L2正则化: {advice['优化建议'].get('reg_lambda', 0.0)}
                - 最小分裂增益: {advice['优化建议'].get('min_split_gain', 0.0)}
                - 测试集比例: {test_size_display}%
                """)

        else:  # Ridge回归
            test_size = advice['优化建议'].get('test_size', 20)
            if isinstance(test_size, float) and test_size < 1:
                test_size_display = int(test_size * 100)
            else:
                test_size_display = test_size

            st.markdown(f"""
                - alpha（正则化强度）: {advice['优化建议'].get('alpha', 1.0)}
                - 是否标准化: {'是' if advice['优化建议'].get('normalize', True) else '否'}
                - 求解器: {advice['优化建议'].get('solver', 'auto')}
                - 测试集比例: {test_size_display}%
                """)

        # ===== 新增：应用参数按钮 =====
        if st.button("✨ 应用AI建议参数", use_container_width=True, key="apply_ai_params"):
            # 保存AI建议的参数到session_state
            st.session_state.ai_suggested_params = advice['优化建议']
            st.session_state.step = 3  # 跳转到训练步骤
            st.rerun()

        st.success(f"🎯 **预期效果**: {advice['预期效果']}")

        if st.session_state.ai_history and len(st.session_state.ai_history) > 0:
            with st.expander("📜 查看历史建议记录"):
                for i, h in enumerate(st.session_state.ai_history):
                    st.write(h)
    else:
        st.caption("点击主界面的【AI评价及建议】按钮获取优化建议")

    # 帮助信息
    with st.expander("❓ 使用帮助", expanded=False):
        st.markdown("""
        **数据格式：**
        - 最后一列为标签
        - 文本列需要编码
        - 数值列保持原样
        **模型上传：**
        - .h5 文件为神经网络模型
        - .pkl 文件为随机森林模型
        """)

# ========== 主界面 ==========
st.title("🧠 小屿屿训练器")

# 步骤导航显示
cols = st.columns(4)
steps_names = ["上传数据", "清洗数据", "训练模型", "预测结果"]
for i, (col, name) in enumerate(zip(cols, steps_names)):
    with col:
        if i + 1 == st.session_state.step:
            col.success(f"**{name}** ✅")
        elif i + 1 < st.session_state.step:
            col.info(f"{name} ✓")
        else:
            col.write(name)
st.markdown("---")

# ========== 步骤1：上传数据 ==========
if st.session_state.step == 1:
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df.copy()

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

        # ===== 先进行数据编码 =====
        st.subheader("🔢 数据编码")
        st.info("系统将自动检测并编码文本列（object类型）为数字")

        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        if text_cols:
            st.write(f"**检测到 {len(text_cols)} 个文本列，需要编码：**")

            # 显示各列的样本值
            sample_data = []
            for col in text_cols:
                unique_vals = df[col].dropna().unique()[:3]
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

            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🔢 开始编码", type="primary", use_container_width=True):
                    label_encoders = {}
                    encoding_log = []
                    encoding_error = False

                    for col in text_cols:
                        try:
                            le = LabelEncoder()
                            non_null_mask = df[col].notna()
                            if non_null_mask.any():
                                encoded_vals = le.fit_transform(df.loc[non_null_mask, col].astype(str))
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
                        st.session_state.df = df
                        st.session_state.label_encoders = label_encoders
                        st.session_state.encoding_log = encoding_log
                        st.success("✅ 编码完成！")

                        # 显示编码规则
                        with st.container():
                            st.markdown("### 📋 编码规则")
                            all_rules = []
                            for col, le in label_encoders.items():
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

                        with st.expander("📝 查看编码记录"):
                            for log in encoding_log:
                                st.write(log)

            with col2:
                if st.button("⏭️ 跳过编码（所有列为连续型数值）", use_container_width=True):
                    st.session_state.df = df
                    st.session_state.label_encoders = {}
                    st.warning("⚠️ 跳过编码步骤，只适用于所有列为连续型数值的数据！")
        else:
            st.success("✅ 没有检测到文本列，无需编码")
            st.session_state.df = df
            st.session_state.label_encoders = {}

        # ===== 编码完成后进行模型选择 =====
        if st.session_state.df is not None:
            st.markdown("---")
            st.subheader("🤖 选择模型")

            # 现在可以正确判断分类/回归了
            target_col = df.columns[-1]
            is_classification = target_col in st.session_state.label_encoders

            # 显示问题类型
            st.info(f"📌 检测到问题类型：{'分类' if is_classification else '回归'}")

            n_samples = len(df)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**神经网络 ANN**")
                st.caption("适合大数据集（>10000样本），能学习复杂关系")
                if n_samples < 10000:
                    st.warning(f"当前样本量 {n_samples}，ANN可能效果不佳")

                st.markdown("**逻辑回归 LR**")
                st.caption("适合分类问题，小样本（<500）表现稳定")
                if n_samples < 500 and is_classification:
                    st.success("✅ 当前样本量，分类问题推荐使用逻辑回归")

                st.markdown("**Ridge回归**")
                st.caption("适合回归问题，小样本（<500）表现最佳")
                if n_samples < 500 and not is_classification:
                    st.success("✅ 当前样本量，回归问题推荐使用Ridge回归")

            with col2:
                st.markdown("**随机森林 RF**")
                st.caption("适合中小数据集（500-2000样本），不易过拟合")
                if 500 <= n_samples <= 2000:
                    st.success("✅ 当前样本量，推荐使用随机森林")

                st.markdown("**LightGBM**")
                st.caption("适合中大数据集（>2000样本），训练快速")
                if n_samples > 2000:
                    st.success("✅ 当前样本量，推荐使用LightGBM")

            # 根据问题类型设置默认索引
            if is_classification:
                default_index = 1 if n_samples < 500 else 2 if n_samples <= 2000 else 3
            else:
                default_index = 4 if n_samples < 500 else 2 if n_samples <= 2000 else 3  # 回归问题默认选Ridge

            st.session_state.selected_model = st.radio(
                "请选择模型",
                ["神经网络 ANN", "逻辑回归 LR", "随机森林 RF", "LightGBM", "Ridge回归"],
                index=default_index,
                horizontal=True
            )

        # 显示编码后的数据预览
        if st.session_state.df is not None:
            st.markdown("---")
            st.subheader("📋 处理后数据预览")
            st.dataframe(st.session_state.df.head(10))

            if st.session_state.label_encoders:
                st.write("**编码信息：**")
                for col, le in st.session_state.label_encoders.items():
                    st.write(f"- {col}: 编码为 {len(le.classes_)} 个类别")

        # 进入清洗按钮
        if st.session_state.df is not None:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("➡️ 进入数据清洗", type="primary", use_container_width=True):
                    st.session_state.step = 2
                    st.rerun()
    else:
        st.info("请在左侧边栏上传CSV或Excel文件")

# ========== 步骤2：数据清洗 ==========
elif st.session_state.step == 2:
    if st.session_state.df is not None:
        run_data_cleaning()
    else:
        st.error("请先上传数据")
        if st.button("返回上传"):
            st.session_state.step = 1
            st.rerun()

# ========== 步骤3：模型训练 ==========
elif st.session_state.step == 3:
    if st.session_state.processed_df is not None:
        if st.session_state.selected_model == "神经网络 ANN":
            from models.ann_trainer import train_ann
            train_ann()
        elif st.session_state.selected_model == "逻辑回归 LR":
            from models.lr_trainer import train_lr
            train_lr()
        elif st.session_state.selected_model == "随机森林 RF":
            from models.rf_trainer import train_rf
            train_rf()
        elif st.session_state.selected_model == "LightGBM":
            from models.lgb_trainer import train_lgb
            train_lgb()
        else:  # Ridge回归
            from models.ridge_trainer import train_ridge
            train_ridge()
    else:
        st.error("请先完成数据清洗")
        if st.button("返回清洗"):
            st.session_state.step = 2
            st.rerun()

# ========== 步骤4：预测 ==========
elif st.session_state.step == 4:
    if st.session_state.selected_model == "神经网络 ANN":
        from models.ann_trainer import predict_ann
        predict_ann()
    elif st.session_state.selected_model == "逻辑回归 LR":
        from models.lr_trainer import predict_lr
        predict_lr()
    elif st.session_state.selected_model == "随机森林 RF":
        from models.rf_trainer import predict_rf
        predict_rf()
    elif st.session_state.selected_model == "LightGBM":
        from models.lgb_trainer import predict_lgb
        predict_lgb()
    else:  # Ridge回归
        from models.ridge_trainer import predict_ridge
        predict_ridge()
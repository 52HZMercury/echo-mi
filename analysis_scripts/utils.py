# analysis_scripts/utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob


# --- 顶刊级别的绘图风格配置 ---
def set_publication_style():
    """
    设置Matplotlib和Seaborn的绘图风格，以满足顶级期刊的要求。
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # 使用更适合出版物的颜色主题 (例如 colorblind)
    sns.set_palette("colorblind")

    # 定义字体大小
    FONT_SIZES = {
        'title': 16,
        'suptitle': 18,
        'label': 14,
        'tick': 12,
        'legend': 12,
        'annotation': 10
    }

    # 全局字体设置 (推荐使用无衬线字体，如Arial, Helvetica, 或DejaVu Sans)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': FONT_SIZES['label'],
        'axes.titlesize': FONT_SIZES['title'],
        'axes.labelsize': FONT_SIZES['label'],
        'xtick.labelsize': FONT_SIZES['tick'],
        'ytick.labelsize': FONT_SIZES['tick'],
        'legend.fontsize': FONT_SIZES['legend'],
        'figure.titlesize': FONT_SIZES['suptitle'],
        'figure.dpi': 300,  # 提高图像分辨率
        'savefig.dpi': 300,
        'savefig.format': 'pdf',  # 默认保存为矢量图格式
        'savefig.bbox': 'tight',  # 保存时自动裁剪边缘空白
    })


# --- 数据加载函数 ---
def find_results_dirs(base_dir: str, experiment_name: str) -> list:
    """
    根据实验名称查找所有5折交叉验证的实验结果目录。
    (已更新以匹配新的目录结构: outputs/EXP_NAME/DATE/Fold-X/test_results)

    Args:
        base_dir (str): 'outputs' 目录的路径。
        experiment_name (str): 实验名称 (例如 'CAMUS_FAEC_Experiment_4090_01')。

    Returns:
        list: 包含所有折数结果目录的路径列表。
    """
    # 使用通配符 * 来匹配中间的日期目录
    search_pattern = f"{base_dir}/{experiment_name}/*/Fold-*/test_results"
    result_dirs = glob.glob(search_pattern)
    if not result_dirs:
        raise FileNotFoundError(
            f"在 '{base_dir}' 中未找到与 '{experiment_name}' 相关的实验结果。\n"
            f"搜索模式为: '{search_pattern}'\n"
            "请检查路径和实验名称是否完全匹配。"
        )
    print(f"找到了 {len(result_dirs)} 个实验结果目录。")
    return result_dirs


def load_all_predictions(results_dirs: list) -> pd.DataFrame:
    """加载所有折数的 predictions.csv 文件并合并。"""
    all_preds = []
    for dir_path in results_dirs:
        pred_file = Path(dir_path) / "predictions.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            all_preds.append(df)
    return pd.concat(all_preds, ignore_index=True)


def load_all_metrics(results_dirs: list) -> pd.DataFrame:
    """
    加载所有折数的 evaluation_metrics.csv 文件并合并。
    (已更新以自动规范化列名)
    """
    all_metrics = []
    for dir_path in results_dirs:
        metric_file = Path(dir_path) / "evaluation_metrics.csv"
        if metric_file.exists():
            df = pd.read_csv(metric_file)
            # --- 核心修正点: 规范化列名 ---
            # 将所有列名中的 '/' 替换为 '_'，以处理不一致的日志记录
            df.columns = [col.replace('/', '_') for col in df.columns]
            all_metrics.append(df)
    return pd.concat(all_metrics, ignore_index=True)


def load_all_features(results_dirs: list) -> tuple[np.ndarray, np.ndarray, list]:
    """
    加载所有折数的特征向量和对应的标签。
    (已更新以匹配新的目录结构)
    """
    all_features_list = []
    all_labels_list = []
    all_sample_ids = []

    # 先加载所有预测结果以获取标签和ID
    predictions_df = load_all_predictions(results_dirs)

    for dir_path in results_dirs:
        features_dir = Path(dir_path) / "features"

        # 从 '.../Fold-5/test_results' 路径中提取折数
        fold_dir_name = Path(dir_path).parent.name  # 这将得到 'Fold-5'
        try:
            fold = int(fold_dir_name.split('-')[-1])
        except (ValueError, IndexError):
            print(f"警告: 无法从目录名 '{fold_dir_name}' 中解析折数。跳过此目录。")
            continue

        fold_preds = predictions_df[predictions_df['fold'] == fold]

        for _, row in fold_preds.iterrows():
            sample_id = row['sample_id']
            label = row['true_label']
            feature_file = features_dir / f"{sample_id}.npy"

            if feature_file.exists():
                feature = np.load(feature_file)
                all_features_list.append(feature)
                all_labels_list.append(label)
                all_sample_ids.append(sample_id)
            else:
                print(f"警告: 找不到特征文件 {feature_file}")

    return np.array(all_features_list), np.array(all_labels_list), all_sample_ids
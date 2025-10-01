# analysis_scripts/4_plot_confusion_matrix.py

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils import set_publication_style, find_results_dirs, load_all_predictions


def plot_confusion_matrix_for_data(y_true, y_pred, class_names, title, output_file):
    """
    一个通用的函数，用于计算、绘制并保存混淆矩阵。
    """
    cm = confusion_matrix(y_true, y_pred)

    # 防止因某一类别样本不存在而导致的除零错误
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)

    fig, ax = plt.subplots(figsize=(6, 5))

    # 创建标注，同时包含数量和百分比
    labels = (np.asarray(["{0:d}\n({1:.1%})".format(value, percent)
                          for value, percent in zip(cm.flatten(), cm_percent.flatten())])
              ).reshape(cm.shape)

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})  # 增大标注字体

    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.savefig(output_file)
    print(f"混淆矩阵图已保存至: {output_file}")
    plt.close()


def plot_all_confusion_matrices(base_dir: str, experiment_name: str, output_prefix: str):
    """
    计算并可视化5折交叉验证的总体及每一折的混淆矩阵。
    """
    set_publication_style()

    try:
        results_dirs = find_results_dirs(base_dir, experiment_name)
        predictions_df = load_all_predictions(results_dirs)
    except FileNotFoundError as e:
        print(e)
        return

    class_names = ['Normal', 'MI']

    # --- 1. 绘制总体的混淆矩阵 ---
    y_true_all = predictions_df['true_label']
    y_pred_all = predictions_df['predicted_label']
    plot_confusion_matrix_for_data(
        y_true_all,
        y_pred_all,
        class_names,
        f'Overall Confusion Matrix\n({experiment_name})',
        f"{output_prefix}_overall.pdf"
    )

    # --- 2. 循环绘制每一折的混淆矩阵 ---
    for fold in sorted(predictions_df['fold'].unique()):
        fold_data = predictions_df[predictions_df['fold'] == fold]
        y_true_fold = fold_data['true_label']
        y_pred_fold = fold_data['predicted_label']

        plot_confusion_matrix_for_data(
            y_true_fold,
            y_pred_fold,
            class_names,
            f'Confusion Matrix (Fold {fold})\n({experiment_name})',
            f"{output_prefix}_fold_{fold}.pdf"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot confusion matrix for 5-fold cross-validation.")
    parser.add_argument("--base_dir", type=str, default="/workdir3t/A-Echo/echo-mi/outputs")
    parser.add_argument("--experiment_name", type=str, default="HMC_FAEC_Experiment_4090_01")
    parser.add_argument("--output_prefix", type=str, default="confusion_matrix.pdf")
    args = parser.parse_args()

    plot_all_confusion_matrices(args.base_dir, args.experiment_name, args.output_prefix)
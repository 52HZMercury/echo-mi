# analysis_scripts/2_plot_roc_and_auc_ci.py

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# --- 核心修正点 1: 移除不再使用的 'interp' 导入 ---
# from scipy import interp
from utils import set_publication_style, find_results_dirs, load_all_predictions


def calculate_auc_ci(y_true, y_scores, n_bootstraps=1000, alpha=0.95):
    """使用Bootstrap方法计算AUC的置信区间。"""
    bootstrapped_aucs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true[indices], y_scores[indices])
        bootstrapped_aucs.append(auc(fpr, tpr))

    sorted_aucs = np.array(bootstrapped_aucs)
    sorted_aucs.sort()

    lower_bound = sorted_aucs[int((1.0 - alpha) / 2.0 * len(sorted_aucs))]
    upper_bound = sorted_aucs[int((1.0 + alpha) / 2.0 * len(sorted_aucs))]

    return np.mean(sorted_aucs), lower_bound, upper_bound


def plot_roc_curves(base_dir: str, experiment_name: str, output_file: str):
    """
    绘制5折交叉验证的ROC曲线，并计算AUC的95%置信区间。
    """
    set_publication_style()

    try:
        results_dirs = find_results_dirs(base_dir, experiment_name)
        predictions_df = load_all_predictions(results_dirs)
    except FileNotFoundError as e:
        print(e)
        return

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))

    # --- 打印AUC和置信区间 ---
    print("--- AUC with 95% Confidence Intervals ---")

    # 1. 绘制每一折的ROC曲线
    for fold in sorted(predictions_df['fold'].unique()):
        fold_data = predictions_df[predictions_df['fold'] == fold]
        y_true = fold_data['true_label'].values
        y_scores = fold_data['predicted_prob'].values

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # 使用Bootstrap计算置信区间
        mean_auc_bs, lower_ci, upper_ci = calculate_auc_ci(y_true, y_scores)
        print(f"Fold {fold}: AUC = {roc_auc:.4f} (95% CI: {lower_ci:.4f}-{upper_ci:.4f})")

        ax.plot(fpr, tpr, lw=1, alpha=0.4, label=f'ROC Fold {fold} (AUC = {roc_auc:.2f})')

        # --- 核心修正点 2: 使用 numpy.interp 进行插值 ---
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # 2. 绘制对角线
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.8, label='Chance')

    # 3. 绘制平均ROC曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='b',
            label=f'Mean ROC (AUC = {mean_auc:.2f} $\\pm$ {std_auc:.2f})',
            lw=2.5, alpha=0.9)

    # 4. 绘制标准差区域
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                    label=r'$\pm$ 1 std. dev.')

    # --- 整体数据的AUC和置信区间 ---
    y_true_all = predictions_df['true_label'].values
    y_scores_all = predictions_df['predicted_prob'].values
    mean_auc_all, lower_ci_all, upper_ci_all = calculate_auc_ci(y_true_all, y_scores_all)
    print("-" * 30)
    print(f"Overall: AUC = {mean_auc_all:.4f} (95% CI: {lower_ci_all:.4f}-{upper_ci_all:.4f})")
    print("-" * 30)

    # 5. 设置图表属性
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           xlabel="False Positive Rate",
           ylabel="True Positive Rate",
           title=f"Receiver Operating Characteristic\n({experiment_name})")
    ax.legend(loc="lower right")

    plt.savefig(output_file)
    print(f"\nROC曲线图已保存至: {output_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ROC curves and calculate AUC with 95% CI.")
    parser.add_argument("--base_dir", type=str, default="/workdir2/cn24/program/echo-mi/outputs")
    parser.add_argument("--experiment_name", type=str, default="Experiment_115")
    parser.add_argument("--output_file", type=str, default="Experiment_115/roc_curve.pdf")
    args = parser.parse_args()

    plot_roc_curves(args.base_dir, args.experiment_name, args.output_file)
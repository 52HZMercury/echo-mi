# analysis_scripts/1_summarize_metrics.py

import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from utils import find_results_dirs, load_all_metrics, load_all_predictions


def summarize_metrics(base_dir: str, experiment_name: str, output_file: str):
    """
    计算5折交叉验证的平均性能指标和标准差。
    """
    try:
        results_dirs = find_results_dirs(base_dir, experiment_name)
        metrics_df = load_all_metrics(results_dirs)
    except FileNotFoundError as e:
        print(e)
        return

    # 确保所有指标列都是数值类型
    metric_cols = [col for col in metrics_df.columns if col != 'fold']
    for col in metric_cols:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')

    # 计算均值和标准差
    summary_mean = metrics_df[metric_cols].mean()
    summary_std = metrics_df[metric_cols].std()

    # 创建一个漂亮的摘要DataFrame
    summary_df = pd.DataFrame({
        'Metric': metric_cols,
        'Mean': summary_mean.values,
        'StdDev': summary_std.values
    })

    # 格式化输出，例如保留4位小数
    summary_df['Mean'] = summary_df['Mean'].map('{:.4f}'.format)
    summary_df['StdDev'] = summary_df['StdDev'].map('{:.4f}'.format)

    # 这种方法可以确保所有数字都精确到4位小数，例如 0.9 会被格式化为 0.9000
    summary_df['Value (Mean ± StdDev)'] = [
        f"{mean:.4f} ± {std:.4f}"
        for mean, std in zip(summary_mean, summary_std)
    ]

    print("--- 5-Fold Cross-Validation Performance Summary ---")
    print(f"Experiment: {experiment_name}")
    print(summary_df[['Metric', 'Mean', 'StdDev', 'Value (Mean ± StdDev)']].to_string(index=False))

    # 保存到CSV
    summary_df.to_csv(output_file, index=False)
    print(f"\n摘要已保存至: {output_file}")


def calculate_metric_ci(y_true, y_pred, y_scores, metric_func, n_bootstraps=1000, alpha=0.95):
    """
    使用Bootstrap方法为给定的评估指标计算95%置信区间。
    """
    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))

        # 如果样本中只有一个类别，则跳过此次抽样
        if len(np.unique(y_true[indices])) < 2:
            continue

        # 根据指标类型选择使用预测标签或概率分数
        if metric_func == roc_auc_score:
            score = metric_func(y_true[indices], y_scores[indices])
        else:
            score = metric_func(y_true[indices], y_pred[indices])

        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower_bound = sorted_scores[int((1.0 - alpha) / 2.0 * len(sorted_scores))]
    upper_bound = sorted_scores[int((1.0 + alpha) / 2.0 * len(sorted_scores))]

    return lower_bound, upper_bound


def specificity_score(y_true, y_pred):
    """计算特异度"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def summarize_metrics_with_ci(base_dir: str, experiment_name: str, output_file: str):
    """
    计算所有评估指标的点估计值和95%置信区间。
    """
    try:
        results_dirs = find_results_dirs(base_dir, experiment_name)
        predictions_df = load_all_predictions(results_dirs)
    except FileNotFoundError as e:
        print(e)
        return

    y_true = predictions_df['true_label'].values
    y_pred = predictions_df['predicted_label'].values
    y_scores = predictions_df['predicted_prob'].values

    metrics_to_calculate = {
        "Accuracy": accuracy_score,
        "AUC": roc_auc_score,
        "F1-Score": f1_score,
        "Precision": precision_score,
        "Recall (Sensitivity)": recall_score,
        "Specificity": specificity_score
    }

    results = []

    print("--- Overall Performance with 95% Confidence Intervals ---")
    print(f"Experiment: {experiment_name}")
    print(f"Total samples: {len(y_true)}")
    print("-" * 60)

    for name, func in metrics_to_calculate.items():
        # 计算点估计值 (在全部数据上)
        if func == roc_auc_score:
            point_estimate = func(y_true, y_scores)
        else:
            point_estimate = func(y_true, y_pred)

        # 计算置信区间
        lower_bound, upper_bound = calculate_metric_ci(y_true, y_pred, y_scores, func)

        results.append({
            "Metric": name,
            "Value": point_estimate,
            "95% CI Lower": lower_bound,
            "95% CI Upper": upper_bound,
            "Formatted": f"{point_estimate:.4f} (95% CI: {lower_bound:.4f}-{upper_bound:.4f})"
        })

    summary_df = pd.DataFrame(results)

    # 打印格式化的结果
    print(summary_df[['Metric', 'Formatted']].to_string(index=False))
    print("-" * 60)

    # 保存到CSV
    summary_df.to_csv(output_file + "_95CI", index=False, float_format='%.4f')
    print(f"\n详细性能指标已保存至: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize 5-fold cross-validation metrics.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/workdir3t/A-Echo/echo-mi/outpus_20250827",
        help="实验输出的根目录 (通常是 'outputs')。"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="HMC_FAEC_Experiment_4090_01",
        help="实验名称 (不包含 '_FoldX' 后缀, 例如 'HMC_FAEC_Experiment')。"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="metrics_summary.csv",
        help="保存摘要的CSV文件名。"
    )
    args = parser.parse_args()

    summarize_metrics(args.base_dir, args.experiment_name, args.output_file)
    summarize_metrics_with_ci(args.base_dir, args.experiment_name, args.output_file)

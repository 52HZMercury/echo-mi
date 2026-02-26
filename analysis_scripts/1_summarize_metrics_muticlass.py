import pandas as pd
import numpy as np
import argparse
import os
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from utils import find_results_dirs, load_all_metrics, load_all_predictions
from functools import partial


def multiclass_specificity_score(y_true, y_pred):
    """计算多分类的宏平均特异度"""
    labels = np.unique(y_true)
    specs = []
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specs.append(spec)
    return np.mean(specs)


def summarize_metrics(base_dir: str, experiment_name: str, output_file: str):
    """计算5折交叉验证的平均性能指标和标准差"""
    try:
        results_dirs = find_results_dirs(base_dir, experiment_name)
        metrics_df = load_all_metrics(results_dirs)
    except Exception as e:
        print(f"Error: {e}")
        return

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metric_cols = [col for col in metrics_df.columns if col != 'fold']
    for col in metric_cols:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')

    summary_mean = metrics_df[metric_cols].mean()
    summary_std = metrics_df[metric_cols].std()

    summary_df = pd.DataFrame({
        'Metric': metric_cols,
        'Mean ± StdDev': [f"{m:.4f} ± {s:.4f}" for m, s in zip(summary_mean, summary_std)]
    })

    print(f"\n--- 5-Fold Summary: {experiment_name} ---")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(output_file, index=False)


def calculate_metric_ci(y_true, y_pred, y_probs, metric_func, name, n_bootstraps=1000, alpha=0.95):
    """使用Bootstrap计算95%置信区间"""
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    unique_labels = np.unique(y_true)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        # 确保采样包含所有类别
        if len(np.unique(y_true[indices])) < len(unique_labels):
            continue

        if "AUC" in name:
            score = metric_func(y_true[indices], y_probs[indices])
        else:
            score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    lower = sorted_scores[int((1.0 - alpha) / 2.0 * len(sorted_scores))]
    upper = sorted_scores[int((1.0 + alpha) / 2.0 * len(sorted_scores))]
    return lower, upper


def summarize_metrics_with_ci(base_dir: str, experiment_name: str, output_file: str):
    """计算三分类指标点估计及95% CI"""
    try:
        results_dirs = find_results_dirs(base_dir, experiment_name)
        df = load_all_predictions(results_dirs)
    except Exception as e:
        print(f"Error: {e}")
        return

    y_true = df['true_label'].values
    y_pred = df['predicted_label'].values
    # 提取三列概率
    y_probs = df[['prob_0', 'prob_1', 'prob_2']].values

    # 配置三分类宏平均指标
    metrics_to_calculate = {
        "Accuracy": accuracy_score,
        "AUC": lambda yt, yp: roc_auc_score(yt, yp, multi_class='ovr', average='macro'),
        "F1-Score": partial(f1_score, average='macro'),
        "Precision": partial(precision_score, average='macro'),
        "Recall (Sensitivity)": partial(recall_score, average='macro'),
        "Specificity": multiclass_specificity_score
    }

    results = []
    print(f"\n--- Overall Performance (3-Class) with 95% CI ---")

    for name, func in metrics_to_calculate.items():
        # 点估计
        if "AUC" in name:
            val = func(y_true, y_probs)
        else:
            val = func(y_true, y_pred)

        # 置信区间
        low, high = calculate_metric_ci(y_true, y_pred, y_probs, func, name)

        results.append({
            "Metric": name,
            "Value": val,
            "Formatted": f"{val:.4f} (95% CI: {low:.4f}-{high:.4f})"
        })

    summary_df = pd.DataFrame(results)
    print(summary_df[['Metric', 'Formatted']].to_string(index=False))

    output_path = output_file.replace('.csv', '_95CI.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"\nDetailed metrics saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/workdir2/cn24/program/echo-mi/outputs")
    parser.add_argument("--experiment_name", type=str, default="Experiment_provincial_17")
    parser.add_argument("--output_file", type=str, default="Experiment_provincial_17/metrics_summary.csv")
    args = parser.parse_args()

    summarize_metrics(args.base_dir, args.experiment_name, args.output_file)
    summarize_metrics_with_ci(args.base_dir, args.experiment_name, args.output_file)
# analysis_scripts/3_plot_embeddings.py

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
from utils import set_publication_style, find_results_dirs, load_all_features, load_all_predictions


def plot_embeddings(base_dir: str, experiment_name: str, output_prefix: str, method: str = 'tsne'):
    """
    使用t-SNE或UMAP对特征向量进行降维并可视化。

    完美方案包含两种视角：
    1. 整体样本可视化: 展示模型间差异（inter-fold variance）。
    2. 类别中心点可视化: 评估类别区分度及其稳定性。
    """
    set_publication_style()

    try:
        results_dirs = find_results_dirs(base_dir, experiment_name)
        features, labels, _ = load_all_features(results_dirs)
        predictions_df = load_all_predictions(results_dirs)
        folds = predictions_df['fold'].values
    except FileNotFoundError as e:
        print(e)
        return

    if features.shape[0] == 0:
        print("未加载到任何特征向量，无法进行可视化。")
        return

    # --- 准备工作: 标准化特征和定义视觉元素 ---
    print("对整体特征进行标准化...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    class_labels = {0: 'Normal', 1: 'MI'}
    colors = sns.color_palette("colorblind", n_colors=len(class_labels))
    markers = ['o', 's', '^', 'D', 'P']  # 为5折准备不同的标记

    # ==============================================================================
    # 视角一: 整体样本可视化 (展示并解释模型间差异)
    # ==============================================================================
    print(f"视角一: 正在为整体数据（按折区分）使用 {method.upper()} 进行降维...")

    if method == 'tsne':
        # --- t-SNE 参数调整建议 ---
        # perplexity: 困惑度, 理解为每个点的有效近邻数。这是最重要的参数。
        #   - 典型范围: 5 到 50 之间。
        #   - 较小值 (如 5-15): 更关注数据的局部结构，可能形成多个小而紧密的簇。如果您的图中有很多小碎块，可以尝试增大它。
        #   - 较大值 (如 30-50): 更关注数据的全局结构，簇会更分散，但能更好地反映整体分布。如果您的簇都挤在一起，可以尝试增大它。
        # max_iter: 最大迭代次数。默认值通常足够，但如果簇没有很好地分离，可以尝试增加此值 (例如 2000 或 5000)。
        # learning_rate: 学习率。通常不需要调整，但如果点都挤成一团，可以尝试减小 (如 100-200)。
        # reducer_all = TSNE(n_components=2, perplexity=20, max_iter=2000, learning_rate=150, random_state=42)
        # 修改后（适用于较新的scikit-learn版本）
        reducer_all = TSNE(n_components=2, perplexity=20, n_iter=2000, learning_rate=150, random_state=42,init='random')
    else: # umap
        # --- UMAP 参数调整建议 ---
        # n_neighbors: 近邻数。控制局部与全局结构的平衡，类似于t-SNE的perplexity。
        #   - 较小值 (如 2-10): 聚焦局部细节，可能会把大簇拆成许多小簇，强调数据的细微结构。
        #   - 较大值 (如 20-100): 聚焦全局结构，更好地保留数据的宏观拓扑，使大簇更完整。
        # min_dist: 嵌入点之间的最小距离。控制簇的紧密程度。
        #   - 较小值 (如 0.0-0.1): 簇会非常紧密，点会挤在一起，适合寻找清晰的聚类。
        #   - 较大值 (如 0.5-0.9): 簇会更松散，点会散开，适合观察数据的拓扑结构和流形。
        reducer_all = umap.UMAP(n_neighbors=10, min_dist=0.3, n_components=2, random_state=42)

    embeddings_all = reducer_all.fit_transform(features_scaled)

    fig1, ax1 = plt.subplots(figsize=(5, 5))

    for fold_val in sorted(np.unique(folds)):
        fold_indices = np.where(folds == fold_val)
        marker = markers[fold_val - 1]  # 假设折数从1开始

        # 在每一折内部，再按类别绘制
        for label_val, class_name in class_labels.items():
            class_indices_in_fold = np.where(labels[fold_indices] == label_val)

            # 组合图例标签，仅为一个类别的一个折叠添加标签以避免图例重复
            legend_label = f'Fold {fold_val} - {class_name}' if label_val == 0 else None

            ax1.scatter(
                embeddings_all[fold_indices][class_indices_in_fold, 0],
                embeddings_all[fold_indices][class_indices_in_fold, 1],
                label=legend_label,
                alpha=0.7,
                s=30,
                color=colors[label_val],
                marker=marker
            )

    ax1.set_title(f'Overall {method.upper()} Visualization by Fold')
    ax1.set_xlabel(f'{method.upper()} Component 1')
    ax1.set_ylabel(f'{method.upper()} Component 2')
    ax1.legend(title="Data Source", markerscale=1.5)
    ax1.grid(True)
    ax1.set_xticks([])
    ax1.set_yticks([])

    output_file1 = f"{output_prefix}_overall_by_fold_{method}.pdf"
    plt.savefig(output_file1)
    print(f"视角一的可视化图已保存至: {output_file1}")
    plt.close(fig1)

    # # ==============================================================================
    # # 视角二: 类别中心点可视化 (评估类别区分度)
    # # ==============================================================================
    # print(f"视角二: 正在为类别中心点使用 {method.upper()} 进行降维...")
    #
    # # 计算每个类别在每一折中的特征中心点
    # all_data = pd.DataFrame(features_scaled)
    # all_data['label'] = labels
    # all_data['fold'] = folds
    #
    # centroids = all_data.groupby(['fold', 'label']).mean().reset_index()
    # centroid_features = centroids.drop(['fold', 'label'], axis=1).values
    # centroid_labels = centroids['label'].values
    # centroid_folds = centroids['fold'].values
    #
    # if method == 'tsne':
    #     reducer_centroids = TSNE(n_components=2, perplexity=min(5, len(centroid_features) - 1), random_state=42)
    # else:  # umap
    #     reducer_centroids = umap.UMAP(n_neighbors=min(5, len(centroid_features) - 1), min_dist=0.1, random_state=42)
    #
    # embeddings_centroids = reducer_centroids.fit_transform(centroid_features)
    #
    # fig2, ax2 = plt.subplots(figsize=(8, 7))
    #
    # for label_val, class_name in class_labels.items():
    #     indices = np.where(centroid_labels == label_val)
    #     ax2.scatter(
    #         embeddings_centroids[indices, 0],
    #         embeddings_centroids[indices, 1],
    #         label=class_name,
    #         alpha=0.9,
    #         s=150,  # 中心点使用更大的标记
    #         color=colors[label_val]
    #     )
    #     # 为每个中心点添加折数标注
    #     for i in indices[0]:
    #         ax2.text(embeddings_centroids[i, 0], embeddings_centroids[i, 1], str(centroid_folds[i]),
    #                  ha='center', va='center', color='white', weight='bold', fontsize=10)
    #
    # ax2.set_title(f'{method.upper()} Visualization of Class Centroids')
    # ax2.set_xlabel(f'{method.upper()} Component 1')
    # ax2.set_ylabel(f'{method.upper()} Component 2')
    # ax2.legend(title="Class")
    # ax2.grid(True)
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    #
    # output_file2 = f"{output_prefix}_centroids_{method}.pdf"
    # plt.savefig(output_file2)
    # print(f"视角二的可视化图已保存至: {output_file2}")
    # plt.close(fig2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize feature embeddings using t-SNE or UMAP.")
    parser.add_argument("--base_dir", type=str, default="/workdir2/cn24/program/echo-mi/outputs")
    parser.add_argument("--experiment_name", type=str, default="Experiment_115")
    parser.add_argument("--output_prefix", type=str, default="Experiment_115/embedding_visualization")

    args = parser.parse_args()

    plot_embeddings(args.base_dir, args.experiment_name, args.output_prefix, method='tsne')
    plot_embeddings(args.base_dir, args.experiment_name, args.output_prefix, method='umap')
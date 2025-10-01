# analysis_scripts/5_run_shap_analysis.py

# 1. 获取此脚本文件所在的目录
import os
from pathlib import Path
script_dir = Path(__file__).resolve().parent
# 2. 获取项目的根目录 (即脚本所在目录的上一级)
project_root = script_dir.parent
# 3. 将当前工作目录更改为项目根目录
os.chdir(project_root)

import torch
import hydra
from omegaconf import OmegaConf
import shap
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from utils import set_publication_style


# 警告: SHAP分析非常耗时，通常只在一小部分有代表性的样本上进行。

def run_shap(config_path: str, checkpoint_path: str, num_samples: int, output_prefix: str):
    """
    对模型进行SHAP分析，以解释模型对特征的依赖性。
    """
    set_publication_style()

    # --- 1. 加载配置、数据和模型 ---
    print("正在加载配置和模型...")
    # 我们使用Hydra来加载配置，这样可以轻松地实例化所有组件
    # @hydra.main() 装饰器不适用于非主脚本，所以我们手动加载
    from hydra.experimental import initialize, compose

    with initialize(config_path=config_path):
        # 假设我们正在分析FAEC HMC实验
        cfg = compose(config_name="train_faec_hmc.yaml")

        # 减少batch_size以避免内存问题
        cfg.data.batch_size = 1

        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup('test')
        test_loader = datamodule.test_dataloader()

        # 加载系统 (LightningModule)
        # 您需要根据您分析的模型来修改这里的System类
        from src.systems.faec_system import FAECSystem
        system = FAECSystem.load_from_checkpoint(checkpoint_path)
        system.eval()
        model = system.model.cuda() if torch.cuda.is_available() else system.model

    # --- 2. 准备数据和预测函数 ---
    print(f"正在从测试集中准备 {num_samples} 个样本用于SHAP分析...")

    background_data = []
    test_samples = []

    # 从数据加载器中获取样本
    for i, batch in enumerate(test_loader):
        # (a2c, a4c), labels, sample_ids
        (a2c, a4c), _, _ = batch

        # 获取模型的输入特征 (对于FAEC是最终的信念状态)
        # 注意：这里的逻辑需要根据您分析的模型进行调整
        with torch.no_grad():
            _, features = model(a2c.cuda(), a4c.cuda(), system.prompts, return_features=True)

        if i < num_samples:
            test_samples.append(features.cpu().numpy())

        # 通常使用训练集的一个子集作为背景数据，这里为简化使用测试集的前100个样本
        if i < 100:
            background_data.append(features.cpu().numpy())

        if len(background_data) >= 100 and len(test_samples) >= num_samples:
            break

    background_data = np.concatenate(background_data, axis=0)
    test_samples = np.concatenate(test_samples, axis=0)

    # 定义一个包装好的预测函数，输入为numpy数组，输出也为numpy数组
    def f(x):
        x_tensor = torch.from_numpy(x).float().cuda()
        with torch.no_grad():
            # 这里的逻辑也需要根据模型调整
            # 对于FAEC，特征就是信念状态，直接送入最终分类器
            logits = model.classifiers['head_5'](x_tensor)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    # --- 3. 运行SHAP解释器 ---
    print("正在计算SHAP值 (这可能需要很长时间)...")
    # 使用KernelExplainer，因为它与模型无关
    explainer = shap.KernelExplainer(f, shap.kmeans(background_data, 10))
    shap_values = explainer.shap_values(test_samples)

    # --- 4. 可视化结果 ---
    print("正在生成SHAP可视化图...")

    # 摘要图 (Summary Plot)
    shap.summary_plot(shap_values, test_samples, show=False,
                      feature_names=[f'Feature_{i}' for i in range(test_samples.shape[1])])
    plt.title(f'SHAP Summary Plot ({cfg.experiment_name})')
    plt.savefig(f"{output_prefix}_summary_plot.pdf", bbox_inches='tight')
    plt.close()
    print(f"SHAP摘要图已保存至: {output_prefix}_summary_plot.pdf")

    # 依赖图 (Dependence Plot) - 示例：解释第0个特征
    shap.dependence_plot(0, shap_values, test_samples, show=False)
    plt.title(f'SHAP Dependence Plot for Feature 0')
    plt.savefig(f"{output_prefix}_dependence_plot_0.pdf", bbox_inches='tight')
    plt.close()
    print(f"SHAP依赖图已保存至: {output_prefix}_dependence_plot_0.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP analysis on a trained model.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="../configs",
        help="Hydra配置文件的相对路径。"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/workdir3t/A-Echo/echo-mi/outputs/CAMUS_FAEC_Experiment_4090_01/2025-08-21/Fold-1/checkpoints/epochepoch=06-val_accuracyval_auroc=0.8542.ckpt",
        help="要分析的模型权重文件 (.ckpt) 的路径。"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="用于SHAP分析的样本数量。"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="shap_analysis",
        help="保存SHAP图表的文件名前缀。"
    )

    args = parser.parse_args()
    run_shap(args.config_path, args.checkpoint_path, args.num_samples, args.output_prefix)
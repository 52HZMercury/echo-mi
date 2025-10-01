# visualize_flexible_features.py (Final Corrected Version with Full Comments)

import torch
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import torch.nn.functional as F
import argparse

# --- 项目模块导入 ---
# 从您的项目中导入核心的PyTorch Lightning系统模块。
# 这个模块封装了模型、优化器、损失函数以及训练/验证逻辑。
from src.systems.faec_advanced_system import FAECAdvancedSystem
# 导入您项目中使用的数据模块。
# 这个模块负责数据的加载、预处理和封装成DataLoader。
from src.data.hmc import HMCMultiTaskDataModule

# --- 全局变量 ---
# 创建一个全局字典，用于存储PyTorch hook捕获的中间层激活值（特征图）。
# hook是一种在不修改模型代码的情况下，从模型中提取数据的强大机制。
# 字典的键是自定义的名称(如此处的'feature_map')，值是将是捕获到的张量。
activations = {}


def get_activation(name):
    """
    一个闭包函数，用于创建并返回一个标准的PyTorch前向hook函数。
    闭包的特性使得内部的hook函数可以访问外部传入的`name`变量。

    Args:
        name (str): 用于在全局`activations`字典中存储特征图的键名。

    Returns:
        function: 一个可以被注册到模型层上的hook函数。
    """

    def hook(model, input, output):
        """
        这个内部函数是实际的hook。每当注册了此hook的层完成一次前向传播时，它就会被自动调用。

        Args:
            model (torch.nn.Module): hook所在的模型层。
            input (tuple): 输入到该层的张量元组。
            output: 该层前向传播的输出。可能是单个张量，也可能是元组。
        """
        # 模型的某些层（如注意力层）可能返回一个元组 (特征, 其他值)。
        # 我们需要确保只捕获我们感兴趣的特征张量，通常是元组中的第一个元素。
        if isinstance(output, tuple):
            feature_tensor = output[0]
        else:
            feature_tensor = output

        # 使用.detach()来分离张量，这样它就不再参与梯度计算，可以安全地存储而不会导致内存泄漏。
        activations[name] = feature_tensor.detach()

    return hook


def normalize_for_display(video_tensor: torch.Tensor) -> np.ndarray:
    """
    将经过标准化处理的视频张量转换回可供显示的单帧图像（NumPy数组）。
    这个函数执行的是数据加载时标准化操作的逆过程。

    Args:
        video_tensor (torch.Tensor): 一个批次为1的视频张量，形状通常为 (1, C, T, H, W)。

    Returns:
        np.ndarray: 一个形状为 (H, W, C) 且像素值在[0, 1]范围内的NumPy数组。
    """
    video_tensor = video_tensor.clone().cpu()  # 复制并移动到CPU
    # 从时间维度(T)中选择中间的一帧进行可视化。
    middle_frame = video_tensor[:, :, video_tensor.shape[2] // 2, :, :]

    # --- 逆标准化 ---
    # 这两个值是您在数据预处理中使用的均值和标准差，这里需要用它们来恢复原始像素范围。
    # 可调整参数: 如果您的数据集标准化参数不同，请务必在此处更新。
    mean = torch.tensor([29.1106, 28.0768, 29.0964]).view(1, 3, 1, 1)
    std = torch.tensor([47.9892, 46.4569, 47.2008]).view(1, 3, 1, 1)
    middle_frame = middle_frame * std + mean  # 逆标准化公式: original = normalized * std + mean

    # --- 格式转换 ---
    # squeeze(0)移除批次维度, permute(1, 2, 0)将形状从 (C, H, W) 转换为 (H, W, C) 以便显示。
    frame_np = middle_frame.squeeze(0).permute(1, 2, 0).numpy()

    # 将像素值重新缩放到[0, 1]范围，这是matplotlib.pyplot.imshow所期望的浮点图像格式。
    frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min())
    return np.clip(frame_np, 0, 1)  # 使用clip确保值不会因浮点误差超出[0, 1]范围


def create_superimposed_image(heatmap: np.ndarray, original_image: np.ndarray, alpha=0.35, cmap_name='jet'):
    """
    将热力图以半透明的方式叠加到原始图像上，生成特征激活的覆盖图。

    Args:
        heatmap (np.ndarray): 2D的特征热力图。
        original_image (np.ndarray): 原始的RGB图像 (H, W, C)。
        alpha (float): 热力图的透明度。值越小，原始图像越清晰。
        cmap_name (str): Matplotlib的颜色映射方案，用于给热力图上色。

    Returns:
        np.ndarray: 叠加后的RGB图像。
    """
    # 可调整参数:
    # - alpha: 控制叠加的强度，常用范围0.3-0.6。
    # - cmap_name: 更改颜色方案，如 'viridis', 'plasma', 'coolwarm'。'jet'色彩丰富，对比度高。
    cmap = plt.get_cmap(cmap_name)
    # 将热力图的值归一化到[0, 1]之间，以便颜色映射可以正确应用。
    norm = Normalize(vmin=heatmap.min(), vmax=heatmap.max())
    # 将归一化后的热力图通过颜色映射转换为RGB颜色图，并丢弃alpha通道。
    heatmap_colored = cmap(norm(heatmap))[:, :, :3]

    original_image_float = original_image.astype(np.float32)

    # 核心叠加公式: superimposed = alpha * foreground + (1 - alpha) * background
    superimposed_img = alpha * heatmap_colored + (1 - alpha) * original_image_float
    return np.clip(superimposed_img, 0, 1)


def get_spatial_dims_for_patches(num_patches: int):
    """
    根据Vision Transformer(ViT)中patch的数量，动态推断其原始的时空维度 (T, H, W)。
    ViT将输入展平为patch序列，这个函数的作用就是将其“还原”回空间结构。

    Args:
        num_patches (int): 从模型层输出的patch token的总数。

    Returns:
        tuple or None: (T, H, W) 维度元组，如果未知则返回None。
    """
    # 这是一个硬编码的查找表。键是模型不同阶段的patch数量，值是对应的(T, H, W)维度。
    # 可调整参数: 如果您的模型结构不同（例如输入分辨率、patch大小或降采样策略改变），
    # 您需要根据新模型的结构来更新这个字典。
    PATCH_DIMS = {
        6272: (16, 22, 14),  # 早期层，分辨率较高
        1568: (16, 14, 7),  # 中间层，经过降采样
        392: (8, 7, 7)  # 深层，时空维度可能都已降采样
    }
    if num_patches in PATCH_DIMS:
        return PATCH_DIMS[num_patches]
    else:
        # 如果遇到未在字典中定义的patch数量，打印警告。
        print(f"警告: 未知的patch数量 {num_patches}。无法自动推断维度。")
        return None


def visualize(args):
    """主可视化函数，执行所有加载、处理和绘图操作。"""
    pl.seed_everything(args.seed)  # 设置随机种子以保证结果可复现

    # --- 1. 初始化和加载 ---
    # 根据实验名称设置输出目录，并确保该目录存在。
    output_dir = os.path.join(args.output_base_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"--> [灵活版] 可视化结果将保存至: {output_dir}")

    # 直接实例化DataModule，而不是通过Hydra。
    # 这样可以使脚本更独立，易于在没有完整Hydra配置的情况下运行。
    datamodule = HMCMultiTaskDataModule(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        fold=args.fold,
        batch_size=1,  # 强制batch_size为1，因为我们是逐个样本进行可视化
        num_workers=args.num_workers
    )
    datamodule.setup(stage='test')  # 准备测试数据集
    test_loader = datamodule.test_dataloader()

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 从指定的checkpoint文件加载训练好的模型系统
    system = FAECAdvancedSystem.load_from_checkpoint(args.checkpoint_path, map_location=device)
    system.eval()  # 将模型设置为评估模式（这会禁用dropout等）
    model = system.model  # 从系统中获取原始的模型结构

    # --- 2. 注册 Hook ---
    # 尝试在用户指定的目标层上注册我们之前定义的hook。
    try:
        # dict(model.named_modules()) 可以通过名称快速查找模型中的任何一个层。
        target_layer = dict(model.named_modules())[args.target_layer_name]
        hook_handle = target_layer.register_forward_hook(get_activation('feature_map'))
        print(f"成功在目标层 '{args.target_layer_name}' 上注册hook。")
    except KeyError:
        # 如果用户提供的层名称不存在，捕获异常并给出友好提示。
        print(f"错误: 在模型中未找到目标层 '{args.target_layer_name}'。")
        print("可用层包括 (部分示例):")
        # 打印出模型中所有符合特定模式的层的名称，帮助用户找到正确的名字。
        for name, _ in model.named_modules():
            if 'video_encoder.model.blocks' in name and ('norm' in name or 'attn' in name):
                print(f"- {name}")
        return

    # --- 3. 循环处理数据 ---
    # 确定要处理的样本数量。如果args.num_samples为-1，则处理所有样本。
    num_to_process = min(args.num_samples, len(test_loader)) if args.num_samples != -1 else len(test_loader)
    # 使用tqdm创建进度条，方便跟踪处理进度。
    progress_bar = tqdm(enumerate(test_loader), total=num_to_process, desc=f"分析层 {args.target_layer_name}")

    for i, batch in progress_bar:
        if i >= num_to_process:
            break

        (a4c_video, a2c_video), _, sample_ids = batch
        a4c_video, a2c_video = a4c_video.to(device), a2c_video.to(device)
        current_sample_id = sample_ids[0]

        # 将A2C和A4C视图的视频在批次维度上拼接，以便一次性送入模型。
        video_batch = torch.cat([a4c_video, a2c_video], dim=0)

        # 使用torch.no_grad()上下文管理器，临时禁用梯度计算，以节省内存并加速前向传播。
        with torch.no_grad():
            # 执行前向传播。我们不关心这里的返回值，其唯一目的是触发我们注册的hook。
            model.video_encoder(video_batch)

        # 从全局字典中获取hook捕获的特征图。
        feature_map_flat_batch = activations.get('feature_map')
        if feature_map_flat_batch is None: continue  # 如果没捕获到，跳过此样本

        # 创建一个2x3的子图网格用于显示结果。
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        plt.subplots_adjust(wspace=0.1, hspace=0.3)

        # 分别处理A2C和A4C视图
        views_to_process = {"A2C": (feature_map_flat_batch[0:1], a2c_video),
                            "A4C": (feature_map_flat_batch[1:2], a4c_video)}

        for view_idx, (view_name, (feature_map_flat, video_tensor)) in enumerate(views_to_process.items()):
            # 对于ViT，输出的第一个token通常是[CLS] token，我们需要将其移除，只保留patch tokens。
            # 这个判断逻辑可能需要根据具体模型输出调整。
            patch_tokens = feature_map_flat[:, 1:, :] if feature_map_flat.shape[1] % 2 != 0 else feature_map_flat

            # 使用辅助函数将展平的patch序列还原为时空维度。
            spatial_dims = get_spatial_dims_for_patches(patch_tokens.shape[1])
            if not spatial_dims: continue  # 如果无法还原，跳过

            T, H, W = spatial_dims
            D = patch_tokens.shape[-1]  # 特征维度

            # --- 特征图处理与可视化 ---
            # 1. Reshape: 将patch序列重新组织为 (B, T, H, W, D) 的空间形式。
            feature_map_spatial = patch_tokens.reshape(1, T, H, W, D)
            # 2. Select Frame: 选择中间时间帧的特征进行可视化。
            frame_features = feature_map_spatial[0, T // 2, :, :, :]
            # 3. Create Heatmap: 沿着特征维度(D)取平均值，生成一个2D热力图。
            heatmap = torch.mean(frame_features, dim=-1)

            # 4. Upsample: 使用双线性插值将热力图上采样到原始视频帧的大小。
            original_size = (video_tensor.shape[-2], video_tensor.shape[-1])
            heatmap_upsampled = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=original_size, mode='bilinear',
                                              align_corners=False).squeeze().cpu().numpy()

            # 5. Get Original Frame: 使用辅助函数获取可显示的原始图像。
            original_frame = normalize_for_display(video_tensor)

            # --- 绘图 ---
            # 绘制原始帧
            axs[view_idx, 0].imshow(original_frame, aspect='equal')
            axs[view_idx, 0].set_title(f"Original {view_name}")
            axs[view_idx, 0].axis('off')

            # 绘制热力图
            axs[view_idx, 1].imshow(heatmap_upsampled, cmap='viridis', aspect='equal')
            axs[view_idx, 1].set_title(f"Feature Map")
            axs[view_idx, 1].axis('off')

            # 绘制叠加图
            overlay = create_superimposed_image(heatmap_upsampled, original_frame)
            axs[view_idx, 2].imshow(overlay, aspect='equal')
            axs[view_idx, 2].set_title(f"Activation Overlay")
            axs[view_idx, 2].axis('off')

        # --- 保存图像 ---
        # 清理层名称中的'.'，使其成为一个有效的文件名。
        sanitized_layer_name = args.target_layer_name.replace('.', '_')
        fig.suptitle(f"Feature Viz - Layer: {args.target_layer_name}\nSample: {current_sample_id}", fontsize=16)
        save_path = os.path.join(output_dir, f"sample_{current_sample_id}_layer_{sanitized_layer_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)  # 关闭图像以释放内存，在循环中非常重要

    # --- 清理 ---
    # 在所有操作完成后，务必移除hook，否则它会一直驻留在内存中。
    hook_handle.remove()
    print(f"\n--> {i + 1} 个样本的可视化图已成功保存至 '{output_dir}'.")


if __name__ == "__main__":
    # --- 使用 argparse 集中管理所有可调参数 ---
    # 这使得您可以直接从命令行运行脚本并轻松更改配置，而无需修改代码。
    parser = argparse.ArgumentParser(description="Flexible Feature Map Visualization for FAEC-Advanced Model.")

    # --- 可调整参数 ---
    parser.add_argument('--checkpoint_path', type=str,
                        default='outputs/HMC_FAEC_Advanced_Experiment/2025-08-28/Fold-4/checkpoints/epoch=19_val_accuracy=0.8125.ckpt',
                        help='要加载的模型检查点文件 (.ckpt) 的路径。')

    parser.add_argument('--target_layer_name', type=str, default='video_encoder.model.norm',
                        help='要可视化的目标层的完整名称。如果名称错误，程序会打印可用层名。')

    parser.add_argument('--experiment_name', type=str, default='norm',
                        help='用于保存结果的输出子目录的名称，建议与层名相关。')

    parser.add_argument('--output_base_dir', type=str, default='outputs_cam',
                        help='保存所有可视化结果的基础目录。')

    parser.add_argument('--num_samples', type=int, default=-1,
                        help='要可视化的样本数量。设为 -1 表示处理测试集中的所有样本。')

    parser.add_argument('--seed', type=int, default=3407, help='全局随机种子。')

    # --- 数据相关参数 ---
    # 这些参数通常由Hydra从YAML文件加载，这里我们用argparse手动指定它们。
    # 您需要根据您的项目结构和数据集位置调整这些默认值。
    parser.add_argument('--data_dir', type=str, default="/workdir3t/A-Echo/Dataset/MI-DATA/HMC-QU/pt_data/",
                        help='数据目录的路径。')
    parser.add_argument('--metadata_path', type=str, default="data/160version.csv",
                        help='元数据CSV文件的路径。')
    parser.add_argument('--fold', type=int, default=4, help='交叉验证的折数。')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载器使用的工作进程数。')

    args = parser.parse_args()

    # (此部分不再需要，因为我们直接将args传递给DataModule)

    visualize(args)

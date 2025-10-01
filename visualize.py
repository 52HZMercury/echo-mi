import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from src.utils.visualizations import generate_gradcam_for_transformer, reshape_heatmap_to_2d, superimpose_heatmap
from src.systems.gef_system import GEFSystem
from src.utils.prompts import segment_prompts

@hydra.main(config_path="configs", config_name="visualize", version_base=None)
def visualize(cfg: DictConfig) -> None:
    print("------ Configuration ------")
    OmegaConf.update(cfg, "data.batch_size", 1, merge=True)
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------")

    pl.seed_everything(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage='test')

    system = GEFSystem.load_from_checkpoint(cfg.checkpoint_path)
    system.to('cuda' if cfg.trainer.accelerator == 'gpu' else 'cpu')
    system.eval()

    activations = {}
    gradients = {}
    def save_activation(name):
        def hook(model, input, output):
            activations[name] = output[0].detach()
        return hook

    def save_gradient(name):
        def hook(model, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook

    try:
        module_dict = dict(system.model.named_modules())
        target_layer = module_dict[cfg.grad_cam_target_layer_name]
    except KeyError:
        print(f"Error: Layer '{cfg.grad_cam_target_layer_name}' not found in model.")
        print("Available layers are:")
        for name in module_dict.keys():
            print(name)
        return

    target_layer.register_forward_hook(save_activation('target'))
    target_layer.register_full_backward_hook(save_gradient('target'))

    prompts = segment_prompts["A4C"]
    test_loader = datamodule.test_dataloader()

    for i, batch in enumerate(tqdm(test_loader, desc="Visualizing Samples")):
        if i >= cfg.num_samples_to_viz:
            break

        (a2c, a4c), labels = batch
        targets = labels["mi_label"]
        a2c, a4c, targets = a2c.to(system.device), a4c.to(system.device), targets.to(system.device)

        with torch.enable_grad():
            a2c.requires_grad = True
            a4c.requires_grad = True
            context = system.model.get_context(a2c, a4c, prompts)
            b_final, _, _ = system.model.solve_ode(context, cfg.viz_ode_steps, enable_grad=True)
            system.model.zero_grad()
            b_final.sum().backward()

        grad = gradients['target'][0]
        acti = activations['target'][0]

        heatmap_1d = generate_gradcam_for_transformer(grad, acti)
        heatmap_patches = heatmap_1d[1:]
        heatmap_2d = reshape_heatmap_to_2d(heatmap_patches)

        # --- **核心修正点** ---
        # 1. 提取中间帧，但保持通道维度
        # `a4c.detach()[0]` -> [C, T, H, W]
        # `[:, a4c.shape[2] // 2, :, :]` -> [C, H, W]
        original_frame_tensor = a4c.detach()[0, :, a4c.shape[2] // 2, :, :]

        # 2. 将张量转换为NumPy数组
        original_frame = original_frame_tensor.cpu().numpy()

        # 3. 执行正确的轴变换: (C, H, W) -> (H, W, C)
        original_frame = np.transpose(original_frame, (1, 2, 0))
        # --- 修正结束 ---

        # 归一化以便显示
        original_frame = (original_frame - original_frame.min()) / (original_frame.max() - original_frame.min() + 1e-8)

        # 如果是单通道灰度图，复制通道以适应热力图叠加
        if original_frame.shape[2] == 1:
            original_frame = np.concatenate([original_frame] * 3, axis=2)

        superimposed_image = superimpose_heatmap(heatmap_2d, original_frame)

        final_prob = torch.sigmoid(b_final[0]).item()
        target_label = targets[0].item()
        save_path = output_dir / f"sample_{i}_prob{final_prob:.2f}_target{target_label}.png"
        plt.imsave(save_path, superimposed_image)

        del grad, acti, context, b_final
        torch.cuda.empty_cache()

    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    visualize()

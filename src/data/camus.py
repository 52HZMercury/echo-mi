import random
from pathlib import Path
from .base import BaseEchoDataset, BaseDataModule
import torch
import torch.nn.functional as F
import random
from pathlib import Path
from .base import BaseEchoDataset


class CAMUSDataset(BaseEchoDataset):
    """CAMUS 数据集（支持多种帧扩展策略）"""

    def __init__(self, data_dir, metadata_path, split, fold, view, expand_mode="repeat"):
        """
        Args:
            data_dir (str): 数据目录
            metadata_path (str): 元数据文件路径
            split (str): "train" / "test"
            fold (int): 交叉验证折数
            view (str): "A2C", "A4C" 或 "both"
            expand_mode (str): 帧扩展模式，可选：
                ["repeat", "interpolate", "random_repeat", "cyclic", "reflect", "noise", "random"]
                | 模式名               | 方法         | 特点            |
                | `"repeat"`         | 简单帧复制    | 稳定、无噪声、简单     |
                | `"interpolate"`    | 3D 插值      | 平滑、适合时序连续     |
                | `"random_repeat"`  | 随机重复帧    | 数据增强，增加时间抖动   |
                | `"cyclic"`         | 循环播放填充   | 保持周期性（心脏视频推荐） |
                | `"reflect"`        | 时间镜像扩展   | 增强首尾边界一致性     |
                | `"noise"`          | 插值+噪声     | 保留时序同时加扰动     |
                | `"random"`         | 随机选择一种策略 | 强数据增强方式       |

        """
        super().__init__(data_dir, metadata_path, split, fold)
        self.view = view
        self.expand_mode = expand_mode.lower()
        self.target_frames = 32  # 可调整

    # --------------------------
    # 主函数：根据 view 加载数据
    # --------------------------
    def __getitem__(self, idx):
        patient_info = self.patients[idx]
        name = patient_info['Number']

        if self.view == "both":
            a2c_path = Path(self.data_dir) / 'A2C' / f"{name}.pt"
            a4c_path = Path(self.data_dir) / 'A4C' / f"{name}.pt"
            a2c_tensor = torch.load(a2c_path)
            a4c_tensor = torch.load(a4c_path)

            a2c_tensor = self._expand_temporal_dim(a2c_tensor)
            a4c_tensor = self._expand_temporal_dim(a4c_tensor)

            label = int(patient_info["Both"])
            return a2c_tensor, a4c_tensor, label, name
        else:
            view_path = Path(self.data_dir) / self.view / f"{name}.pt"
            video_tensor = torch.load(view_path)
            video_tensor = self._expand_temporal_dim(video_tensor)
            label = int(patient_info[self.view])
            return video_tensor, label, name

    # --------------------------
    # 主调度函数
    # --------------------------
    def _expand_temporal_dim(self, tensor):
        """根据配置选择不同扩展策略"""
        methods = {
            "repeat": self._expand_temporal_repeat,
            "interpolate": self._expand_temporal_interpolate,
            "random_repeat": self._expand_temporal_random_repeat,
            "cyclic": self._expand_temporal_cyclic,
            "reflect": self._expand_temporal_reflect,
            "noise": self._expand_temporal_interpolate_noise,
        }

        if self.expand_mode == "random":
            method = random.choice(list(methods.values()))
        else:
            method = methods.get(self.expand_mode, self._expand_temporal_interpolate)

        return method(tensor, self.target_frames)

    # =========================================================
    # 各种扩展策略（均输出 [C, 32, H, W]）
    # =========================================================

    def _expand_temporal_repeat(self, tensor, target_frames):
        """重复帧扩展"""
        C, T, H, W = tensor.shape
        repeat_factor = target_frames // T
        remainder = target_frames % T
        expanded = tensor.repeat_interleave(repeat_factor, dim=1)
        if remainder > 0:
            expanded = torch.cat([expanded, tensor[:, :remainder]], dim=1)
        return expanded

    def _expand_temporal_interpolate(self, tensor, target_frames):
        """时间插值扩展"""
        C, T, H, W = tensor.shape
        tensor = tensor.unsqueeze(0)  # [1, C, T, H, W]
        expanded = F.interpolate(
            tensor,
            size=(target_frames, H, W),
            mode='trilinear',
            align_corners=False
        )
        return expanded.squeeze(0)

    def _expand_temporal_random_repeat(self, tensor, target_frames):
        """随机重复部分帧"""
        C, T, H, W = tensor.shape
        if T >= target_frames:
            return tensor[:, :target_frames]
        indices = list(range(T))
        extra_indices = random.choices(indices, k=target_frames - T)
        all_indices = sorted(indices + extra_indices)
        return tensor[:, all_indices]

    def _expand_temporal_cyclic(self, tensor, target_frames):
        """循环播放扩展"""
        C, T, H, W = tensor.shape
        repeats = (target_frames + T - 1) // T
        expanded = tensor.repeat(1, repeats, 1, 1)
        return expanded[:, :target_frames]

    def _expand_temporal_reflect(self, tensor, target_frames):
        """镜像反射扩展"""
        C, T, H, W = tensor.shape
        if T >= target_frames:
            return tensor[:, :target_frames]
        reflect = torch.flip(tensor, dims=[1])
        expanded = torch.cat([tensor, reflect], dim=1)
        if expanded.shape[1] < target_frames:
            expanded = torch.cat([expanded, tensor[:, :target_frames - expanded.shape[1]]], dim=1)
        return expanded[:, :target_frames]

    def _expand_temporal_interpolate_noise(self, tensor, target_frames, noise_scale=0.02):
        """插值 + 噪声扩展"""
        tensor = self._expand_temporal_interpolate(tensor, target_frames)
        noise = torch.randn_like(tensor) * noise_scale
        return tensor + noise



class CAMUSMultiTaskDataset(BaseEchoDataset):
    """用于多任务学习的CAMUS数据集"""

    def __init__(self, data_dir, metadata_path, split, fold):
        super().__init__(data_dir, metadata_path, split, fold)

    def __getitem__(self, idx):
        patient_info = self.patients[idx]
        name = patient_info['Number']
        mi_label = int(patient_info["Both"])

        a2c_path = Path(self.data_dir) / 'A2C' / f"{name}.pt"
        a4c_path = Path(self.data_dir) / 'A4C' / f"{name}.pt"
        tensor1 = torch.load(a2c_path)
        tensor2 = torch.load(a4c_path)

        # 视图分类标签: A2C为0, A4C为1
        view_labels = [0, 1]

        # 随机交换顺序以进行数据增强
        if random.random() > 0.5:
            tensor1, tensor2 = tensor2, tensor1
            view_labels = [1, 0]

        # 返回样本ID
        return (tensor1, tensor2), {"view_label": torch.tensor(view_labels), "mi_label": mi_label}, name


class CAMUSDataModule(BaseDataModule):
    def __init__(self, data_dir, metadata_path, fold, view, batch_size, num_workers):
        super().__init__(data_dir, metadata_path, fold, batch_size, num_workers)
        self.view = view

    def setup(self, stage=None):
        self.train_dataset = CAMUSDataset(self.data_dir, self.metadata_path, "train", self.fold, self.view)
        self.val_dataset = CAMUSDataset(self.data_dir, self.metadata_path, "test", self.fold, self.view)
        self.test_dataset = self.val_dataset

class CAMUSDoubleViewDataModule(CAMUSDataModule):
    def __init__(self, data_dir, metadata_path, fold, batch_size, num_workers):
        super().__init__(data_dir, metadata_path, fold, "both", batch_size, num_workers)


class CAMUSMultiTaskDataModule(BaseDataModule):
    def setup(self, stage=None):
        self.train_dataset = CAMUSMultiTaskDataset(self.data_dir, self.metadata_path, "train", self.fold)
        self.val_dataset = CAMUSMultiTaskDataset(self.data_dir, self.metadata_path, "test", self.fold)
        self.test_dataset = self.val_dataset
import torch
import random
from pathlib import Path
import os
from .base import BaseEchoDataset, BaseDataModule
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

class HMCDataset(BaseEchoDataset):
    """HMC 数据集 (加载 .pt 文件)"""

    def __init__(self, data_dir, metadata_path, split, fold, view):

        super().__init__(data_dir, metadata_path, split, fold)
        self.view = view
        # 定义数据增强变换
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]) if split == "train" else None

    def augment_video(self, video_tensor):
        """对视频张量进行数据增强"""
        if self.transform is None:
            return video_tensor

        c, t, h, w = video_tensor.shape

        # 对每一帧应用空间增强
        augmented_frames = []
        for i in range(t):
            frame = video_tensor[:, i, :, :]  # [3, 224, 224]
            # 添加通道维度以便使用transforms
            frame = frame.unsqueeze(0)  # [1, 3, 224, 224]
            augmented_frame = self.transform(frame)
            augmented_frames.append(augmented_frame.squeeze(0))  # [3, 224, 224]

        # 重新组合为视频张量
        augmented_video = torch.stack(augmented_frames, dim=1)  # [3, 16, 224, 224]

        # 时间维度增强：保持帧数一致
        if random.random() > 0.5 and t >= 16:
            # 使用固定的帧数采样策略，确保输出始终是16帧
            indices = sorted(random.sample(range(t), k=16))
            augmented_video = augmented_video[:, indices, :, :]

        # 添加小量噪声
        if random.random() > 0.7:
            noise = torch.randn_like(augmented_video) * 0.01
            augmented_video = augmented_video + noise

        return augmented_video

    def __getitem__(self, idx):
        patient_info = self.patients[idx]
        name = patient_info['name']

        if self.view == "both":
            a2c_path = Path(self.data_dir) / 'A2C' / f"{name}.pt"
            a4c_path = Path(self.data_dir) / 'A4C' / f"{name}.pt"
            a2c_tensor = torch.load(a2c_path)  # [3,16,224,224]
            a4c_tensor = torch.load(a4c_path)  # [3,16,224,224]

            # 应用数据增强
            if self.transform is not None:
                a2c_tensor = self.augment_video(a2c_tensor)
                a4c_tensor = self.augment_video(a4c_tensor)

            label = int(patient_info["seg_Bi"])
            # 返回样本ID
            return a2c_tensor, a4c_tensor, label, name
        else:
            view_path = Path(self.data_dir) / self.view / f"{name}.pt"
            video_tensor = torch.load(view_path)
            label = int(patient_info[f"seg_{self.view}"])
            # 返回样本ID
            return video_tensor, label, name


class HMCViewClsDataset(torch.utils.data.Dataset):
    """用于视图分类的HMC数据集"""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_paths = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".pt"):
                self.file_paths.append(os.path.join(data_dir, filename))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path)
        label = 0 if '_A2C' in file_path else 1
        return data, label


# --- 关键修改点：更新 HMCMultiTaskDataset 的实现 ---
class HMCMultiTaskDataset(BaseEchoDataset):
    """
    用于多任务学习的HMC数据集。
    (新) 直接加载预处理好的 .pt 文件以提高效率。
    """

    def __init__(self, data_dir, metadata_path, split, fold):
        # 这里的 data_dir 应该是 .pt 文件的根目录
        super().__init__(data_dir, metadata_path, split, fold)

    def __getitem__(self, idx):
        patient_info = self.patients[idx]
        name = patient_info['name']
        mi_label = int(patient_info["seg_Bi"])

        # 直接加载预处理好的 .pt 文件
        a2c_path = Path(self.data_dir) / 'A2C' / f"{name}.pt"
        tensor1 = torch.load(a2c_path)

        a4c_path = Path(self.data_dir) / 'A4C' / f"{name}.pt"
        tensor2 = torch.load(a4c_path)

        # 创建视图分类标签 (A2C: 0, A4C: 1)
        view_labels = [0, 1]

        # 随机交换顺序以进行数据增强
        if random.random() > 0.5:
            tensor1, tensor2 = tensor2, tensor1
            view_labels = [1, 0]

        # 返回样本ID
        return (tensor1, tensor2), {"view_label": torch.tensor(view_labels), "mi_label": mi_label}, name


# --- DataModule 部分保持不变 ---
class HMCDataModule(BaseDataModule):
    def __init__(self, data_dir, metadata_path, fold, view, batch_size, num_workers):
        super().__init__(data_dir, metadata_path, fold, batch_size, num_workers)
        self.view = view

    def setup(self, stage=None):
        self.train_dataset = HMCDataset(self.data_dir, self.metadata_path, "train", self.fold, self.view)
        self.val_dataset = HMCDataset(self.data_dir, self.metadata_path, "test", self.fold, self.view)
        self.test_dataset = self.val_dataset


class HMCSingleViewDataModule(HMCDataModule):
    def __init__(self, data_dir, metadata_path, fold, view, batch_size, num_workers):
        if view not in ['A2C', 'A4C']:
            raise ValueError("HMCSingleViewDataModule requires view to be 'A2C' or 'A4C'.")
        super().__init__(data_dir, metadata_path, fold, view, batch_size, num_workers)


class HMCDoubleViewDataModule(HMCDataModule):
    def __init__(self, data_dir, metadata_path, fold, batch_size, num_workers):
        super().__init__(data_dir, metadata_path, fold, "both", batch_size, num_workers)


class HMCMultiTaskDataModule(BaseDataModule):
    """用于HMC多任务学习的LightningDataModule"""

    def setup(self, stage=None):
        self.train_dataset = HMCMultiTaskDataset(self.data_dir, self.metadata_path, "train", self.fold)
        self.val_dataset = HMCMultiTaskDataset(self.data_dir, self.metadata_path, "test", self.fold)
        self.test_dataset = self.val_dataset

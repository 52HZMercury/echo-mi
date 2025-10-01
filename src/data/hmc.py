import torch
import random
from pathlib import Path
import os
from .base import BaseEchoDataset, BaseDataModule
from torch.utils.data import DataLoader


class HMCDataset(BaseEchoDataset):
    """HMC 数据集 (加载 .pt 文件)"""

    def __init__(self, data_dir, metadata_path, split, fold, view):
        super().__init__(data_dir, metadata_path, split, fold)
        self.view = view

    def __getitem__(self, idx):
        patient_info = self.patients[idx]
        name = patient_info['name']

        if self.view == "both":
            a2c_path = Path(self.data_dir) / 'A2C' / f"{name}.pt"
            a4c_path = Path(self.data_dir) / 'A4C' / f"{name}.pt"
            a2c_tensor = torch.load(a2c_path)
            a4c_tensor = torch.load(a4c_path)
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
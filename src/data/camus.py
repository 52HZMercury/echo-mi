import torch
import torch.nn.functional as F
import random
from pathlib import Path
from .base import BaseEchoDataset, BaseDataModule



class CAMUSDataset(BaseEchoDataset):
    """CAMUS 数据集"""

    def __init__(self, data_dir, metadata_path, split, fold, view):
        super().__init__(data_dir, metadata_path, split, fold)
        self.view = view


    def __getitem__(self, idx):
        patient_info = self.patients[idx]
        name = patient_info['Number']

        if self.view == "both":
            a2c_path = Path(self.data_dir) / 'A2C' / f"{name}.pt"
            a4c_path = Path(self.data_dir) / 'A4C' / f"{name}.pt"
            a2c_tensor = torch.load(a2c_path)
            a4c_tensor = torch.load(a4c_path)

            # 复制张量到指定尺寸 [3, 32, 224, 224]
            a2c_tensor = self._expand_temporal_dim(a2c_tensor)
            a4c_tensor = self._expand_temporal_dim(a4c_tensor)


            label = int(patient_info["Both"])
            # 返回样本ID
            return a2c_tensor, a4c_tensor, label, name
        else:
            view_path = Path(self.data_dir) / self.view / f"{name}.pt"
            video_tensor = torch.load(view_path)

            # 复制张量到指定尺寸 [3, 32, 224, 224]
            video_tensor = self._expand_temporal_dim(video_tensor)

            label = int(patient_info[self.view])
            # 返回样本ID
            return video_tensor, label, name

    def _expand_temporal_dim(self, tensor):
        """将张量的时间维度从16扩展到32"""
        if tensor.shape[1] == 16:
            # 通过复制帧来扩展时间维度
            expanded_tensor = torch.repeat_interleave(tensor, 2, dim=1)
            return expanded_tensor
        elif tensor.shape[1] == 32:
            return tensor
        else:
            # 如果不是期望的尺寸，使用插值方法
            return F.interpolate(tensor, size=(32, 224, 224), mode='trilinear', align_corners=False)


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


class CAMUSMultiTaskDataModule(BaseDataModule):
    def setup(self, stage=None):
        self.train_dataset = CAMUSMultiTaskDataset(self.data_dir, self.metadata_path, "train", self.fold)
        self.val_dataset = CAMUSMultiTaskDataset(self.data_dir, self.metadata_path, "test", self.fold)
        self.test_dataset = self.val_dataset
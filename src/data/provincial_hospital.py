import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path

from .base import BaseEchoDataset, BaseDataModule

class ProvincialHospitalDataset(BaseEchoDataset):

    def __init__(self, data_dir, metadata_path, split, fold, view):
        super().__init__(data_dir, metadata_path, split, fold)
        self.view = view


    def __len__(self):
        """返回有效样本的总数。"""
        return len(self.patients)

    def __getitem__(self, idx):
        """
        从数据集中获取单个样本。

        返回:
            tuple: (a2c_tensor, a4c_tensor, label, record_id)
        """
        patient_info = self.patients[idx]

        # 加载标签
        record_id = patient_info["recordID"]

        if self.view == "2C":
            a2c_path = Path(self.data_dir) / f"{record_id}" / 'A2C.npy'
            a4c_path = Path(self.data_dir) / f"{record_id}" / 'A4C.npy'

            a2c_tensor = torch.from_numpy(np.load(a2c_path)).float()
            a4c_tensor = torch.from_numpy(np.load(a4c_path)).float()

            label = int(patient_info["severe"])
            # 返回样本ID
            return a2c_tensor, a4c_tensor, label, record_id

        elif self.view == "6C":
            a2c_path = Path(self.data_dir) / f"{record_id}" / 'A2C.npy'
            a3c_path = Path(self.data_dir) / f"{record_id}" / 'A3C.npy'
            a4c_path = Path(self.data_dir) / f"{record_id}" / 'A4C.npy'
            apsax_path = Path(self.data_dir) / f"{record_id}" / 'APSAX.npy'
            mvsax_path = Path(self.data_dir) / f"{record_id}" / 'MVSAX.npy'
            pmsax_path = Path(self.data_dir) / f"{record_id}" / 'PMSAX.npy'

            a2c_tensor = torch.from_numpy(np.load(a2c_path)).float()
            a3c_tensor = torch.from_numpy(np.load(a3c_path)).float()
            a4c_tensor = torch.from_numpy(np.load(a4c_path)).float()
            apsax_tensor = torch.from_numpy(np.load(apsax_path)).float()
            mvsax_tensor = torch.from_numpy(np.load(mvsax_path)).float()
            pmsax_tensor = torch.from_numpy(np.load(pmsax_path)).float()

            label = int(patient_info["severe"])
            # 返回样本ID
            return a2c_tensor, a3c_tensor, a4c_tensor, apsax_tensor, mvsax_tensor, pmsax_tensor, label, record_id


class ProvincialHospitalDataModule(BaseDataModule):
    def __init__(self, data_dir, metadata_path, fold, view, batch_size, num_workers, drop_last=False):
        super().__init__(data_dir, metadata_path, fold, batch_size, num_workers, drop_last)
        self.view = view

    def setup(self, stage=None):
        self.train_dataset = ProvincialHospitalDataset(self.data_dir, self.metadata_path, "train", self.fold, self.view)
        self.val_dataset = ProvincialHospitalDataset(self.data_dir, self.metadata_path, "test", self.fold, self.view)
        self.test_dataset = self.val_dataset

class ProvincialHospitalDoubleViewDataModule(ProvincialHospitalDataModule):
    def __init__(self, data_dir, metadata_path, fold, batch_size, num_workers, drop_last=False):
        super().__init__(data_dir, metadata_path, fold, "2C", batch_size, num_workers, drop_last)

class ProvincialHospitalhexaViewDataModule(ProvincialHospitalDataModule):
    def __init__(self, data_dir, metadata_path, fold, batch_size, num_workers, drop_last=False):
        super().__init__(data_dir, metadata_path, fold, "6C", batch_size, num_workers, drop_last)
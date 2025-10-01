import csv
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class BaseEchoDataset(Dataset):
    """
    超声心动图数据集的基类.
    处理数据集划分 (train/test) 和交叉验证折叠 (fold) 的通用逻辑.
    """
    def __init__(self, data_dir, metadata_path, split, fold):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.fold = fold
        self.patients = []
        self._load_metadata(metadata_path)

    def _load_metadata(self, metadata_path):
        """从CSV文件中加载元数据."""
        with open(metadata_path) as mfile:
            reader = csv.DictReader(mfile)
            for row in reader:
                # 根据 split 和 fold 筛选数据
                is_train = self.split == "train" and self.fold != int(row['fold'])
                is_test = self.split == "test" and self.fold == int(row['fold'])

                # 如果 fold 为 1024, 则加载所有数据
                if self.fold == 1024 or is_train or is_test:
                    self.patients.append(row)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        raise NotImplementedError("子类必须实现 __getitem__ 方法")

class BaseDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule 的基类.
    """
    def __init__(self, data_dir, metadata_path, fold, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        raise NotImplementedError("子类必须实现 setup 方法")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

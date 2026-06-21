import csv
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

SUPPORTED_NUM_FRAMES = (8, 16, 32)


def resample_video_tensor(video_tensor, num_frames):
    """Uniformly sample or cyclically pad a [C, T, H, W] video tensor."""
    if num_frames not in SUPPORTED_NUM_FRAMES:
        raise ValueError(
            f"num_frames must be one of {SUPPORTED_NUM_FRAMES}, got {num_frames}"
        )
    if video_tensor.ndim != 4:
        raise ValueError(
            "video_tensor must have shape [C, T, H, W], "
            f"got {tuple(video_tensor.shape)}"
        )

    source_frames = video_tensor.shape[1]
    if source_frames <= 0:
        raise ValueError("video_tensor must contain at least one frame")
    if source_frames == num_frames:
        return video_tensor

    if source_frames < num_frames:
        indices = torch.arange(num_frames, device=video_tensor.device) % source_frames
    else:
        indices = torch.linspace(
            0,
            source_frames - 1,
            steps=num_frames,
            device=video_tensor.device,
        ).long()
    return video_tensor.index_select(1, indices)


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
    def __init__(self, data_dir, metadata_path, fold, batch_size, num_workers, drop_last):
        super().__init__()
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

    def setup(self, stage=None):
        raise NotImplementedError("子类必须实现 setup 方法")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=True)

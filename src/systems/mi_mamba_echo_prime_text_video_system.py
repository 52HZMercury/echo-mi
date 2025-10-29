import torch
import torchmetrics
import hydra
import torch.nn.functional as F
from .base_system import BaseSystem


class MIMambaEchoPrimeTextVideoSystem(BaseSystem):
    """双视图二分类训练系统，可配置多种损失函数"""

    def __init__(self, model_cfg, learning_rate=1e-5, loss_type="bce"):
        """
        Args:
            model_cfg: 已由 Hydra 实例化的模型对象
            learning_rate: 学习率
            loss_type: 损失类型，可选：
                ["bce", "focal", "dice", "bce_dice", "tversky", "asymmetric"]
        """
        super().__init__(learning_rate=learning_rate)
        self.save_hyperparameters(ignore=['model_cfg'])
        self.model = model_cfg
        self.loss_type = loss_type.lower()

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, a2c_video, a4c_video):
        return self.model(a2c_video, a4c_video)

    # ----------------------------------------------------------------------
    # --- Loss Functions ---
    # ----------------------------------------------------------------------
    def bce_loss(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets)

    def focal_loss(self, logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
        prob = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_weight = alpha * (1 - p_t) ** gamma
        loss = focal_weight * ce_loss
        return loss.mean() if reduction == 'mean' else loss.sum()

    def dice_loss(self, logits, targets, eps=1e-6):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2. * intersection + eps) / (union + eps)
        return 1 - dice

    def bce_dice_loss(self, logits, targets, bce_weight=0.5):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = self.dice_loss(logits, targets)
        return bce_weight * bce + (1 - bce_weight) * dice

    def tversky_loss(self, logits, targets, alpha=0.7, beta=0.3, eps=1e-6):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fp = ((1 - targets) * probs).sum()
        fn = (targets * (1 - probs)).sum()
        tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
        return 1 - tversky

    def asymmetric_loss(self, logits, targets, gamma_pos=0, gamma_neg=4, clip=0.05):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, clip, 1 - clip)
        pos_loss = targets * torch.log(probs) * ((1 - probs) ** gamma_pos)
        neg_loss = (1 - targets) * torch.log(1 - probs) * (probs ** gamma_neg)
        return - (pos_loss + neg_loss).mean()

    # ----------------------------------------------------------------------
    # --- Unified loss selector ---
    # ----------------------------------------------------------------------
    def compute_loss(self, logits, targets):
        if self.loss_type == "bce":
            return self.bce_loss(logits, targets)
        elif self.loss_type == "focal":
            return self.focal_loss(logits, targets)
        elif self.loss_type == "dice":
            return self.dice_loss(logits, targets)
        elif self.loss_type == "bce_dice":
            return self.bce_dice_loss(logits, targets)
        elif self.loss_type == "tversky":
            return self.tversky_loss(logits, targets)
        elif self.loss_type == "asymmetric":
            return self.asymmetric_loss(logits, targets)
        elif self.loss_type == "asymmetric_bce":
            return 0.5 * self.asymmetric_loss(logits, targets) + 0.5 * self.bce_loss(logits, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    # ----------------------------------------------------------------------
    # --- Steps ---
    # ----------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        a2c, a4c, targets, _ = batch
        logits = self(a2c, a4c).squeeze()
        loss = self.compute_loss(logits, targets.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        a2c, a4c, targets, _ = batch
        logits = self(a2c, a4c).squeeze()
        loss = self.compute_loss(logits, targets.float())
        preds = torch.sigmoid(logits)
        self.val_metrics.update(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        a2c, a4c, targets, sample_ids = batch
        logits, features = self.model(a2c, a4c, return_features=True)
        logits = logits.squeeze()
        loss = self.compute_loss(logits, targets.float())
        preds = torch.sigmoid(logits)
        self.test_metrics.update(preds, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return {
            'sample_ids': sample_ids,
            'targets': targets,
            'preds': preds,
            'features': features
        }

import torch
import torchmetrics
import hydra
import torch.nn.functional as F
from .base_system import BaseSystem


class MIMamba6C3System(BaseSystem):
    """双视图二分类训练系统，可配置多种损失函数"""

    def __init__(self, model_cfg, learning_rate=1e-5, loss_type="bce", num_classes=2):
        """
        Args:
            model_cfg: 已由 Hydra 实例化的模型对象
            learning_rate: 学习率
            loss_type: 损失类型，可选：
                ["bce", "focal", "dice", "bce_dice", "tversky", "asymmetric"]
        """
        super().__init__(learning_rate=learning_rate, num_classes=num_classes)
        self.save_hyperparameters(ignore=['model_cfg'])
        self.model = model_cfg
        self.loss_type = loss_type.lower()

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, a2c_video, a3c_video, a4c_video, apsax_video, mvsax_video, pmsax_video):
        return self.model(a2c_video, a3c_video, a4c_video, apsax_video, mvsax_video, pmsax_video)

    # ----------------------------------------------------------------------
    # --- Loss Functions ---
    # ----------------------------------------------------------------------
    def cross_entropy_loss(self, logits, targets):
        return F.cross_entropy(logits, targets)

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
        elif self.loss_type == "cross_entropy":  # 新增交叉熵损失支持
            return self.cross_entropy_loss(logits, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    # ----------------------------------------------------------------------
    # --- Steps ---
    # ----------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        a2c, a3c, a4c, apsax, mvsax, pmsax, targets, _ = batch
        logits = self(a2c, a3c, a4c, apsax, mvsax, pmsax).squeeze()

        # 根据损失类型决定是否转换目标类型
        if self.loss_type == "cross_entropy":
            loss = self.compute_loss(logits, targets.long())  # 使用 long 类型
        else:
            loss = self.compute_loss(logits, targets.float())  # 保持 float 类型

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        a2c, a3c, a4c, apsax, mvsax, pmsax, targets, _ = batch
        logits = self(a2c, a3c, a4c, apsax, mvsax, pmsax).squeeze()
        # 根据损失类型决定是否转换目标类型
        if self.loss_type == "cross_entropy":
            loss = self.compute_loss(logits, targets.long())  # 使用 long 类型
            preds = torch.softmax(logits, dim=-1)  # 多分类使用 softmax
        else:
            loss = self.compute_loss(logits, targets.float())  # 保持 float 类型
            preds = torch.sigmoid(logits)  # 二分类使用 sigmoid

        self.val_metrics.update(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        a2c, a3c, a4c, apsax, mvsax, pmsax, targets, sample_ids = batch
        # 获取 logits 和 features
        logits, features = self.model(a2c, a3c, a4c, apsax, mvsax, pmsax, return_features=True)

        # 移除多余维度，保持 [Batch, 3]
        if logits.dim() > 2:
            logits = logits.squeeze()

        if self.loss_type == "cross_entropy":
            loss = self.compute_loss(logits, targets.long())
            # 计算 Softmax 概率分布 [Batch, 3]
            probs = torch.softmax(logits, dim=-1)
        else:
            loss = self.compute_loss(logits, targets.float())
            probs = torch.sigmoid(logits)

        self.test_metrics.update(probs, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return {
            'sample_ids': sample_ids,
            'targets': targets,
            'preds': probs,  # 传递整个概率矩阵，形状为 [Batch, 3]
            'features': features
        }


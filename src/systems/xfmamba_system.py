import torch

from .mi_mamba_echo_prime_text_video_system import MIMambaEchoPrimeTextVideoSystem


class XFMambaSystem(MIMambaEchoPrimeTextVideoSystem):
    """Binary-classification training system shared by XF-Mamba variants."""

    @staticmethod
    def _flatten_binary_batch(logits, targets):
        return logits.reshape(-1), targets.float().reshape(-1)

    def training_step(self, batch, batch_idx):
        a2c, a4c, targets, _ = batch
        logits, targets = self._flatten_binary_batch(self(a2c, a4c), targets)
        loss = self.compute_loss(logits, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        a2c, a4c, targets, _ = batch
        logits, targets = self._flatten_binary_batch(self(a2c, a4c), targets)
        loss = self.compute_loss(logits, targets)
        self.val_metrics.update(torch.sigmoid(logits), targets.long())
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        a2c, a4c, targets, sample_ids = batch
        logits, features = self.model(a2c, a4c, return_features=True)
        logits, targets = self._flatten_binary_batch(logits, targets)
        loss = self.compute_loss(logits, targets)
        preds = torch.sigmoid(logits)
        self.test_metrics.update(preds, targets.long())
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return {
            "sample_ids": sample_ids,
            "targets": targets,
            "preds": preds,
            "features": features,
        }


__all__ = ["XFMambaSystem"]

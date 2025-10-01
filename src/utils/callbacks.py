from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class ConditionalEarlyStopping(EarlyStopping):
    """
    一个条件化的早停回调.
    它只在指定的 epoch 之后才开始监控指标.
    """

    def __init__(self, start_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def _run_early_stopping_check(self, trainer):
        """
        只有当 trainer.current_epoch >= start_epoch 时才执行检查.
        """
        if trainer.current_epoch >= self.start_epoch:
            super()._run_early_stopping_check(trainer)


class ConditionalModelCheckpoint(ModelCheckpoint):
    """
    一个条件化的模型保存回调.
    它只在指定的 epoch 之后才开始监控和保存模型.
    """

    def __init__(self, start_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        """
        只有当 trainer.current_epoch >= start_epoch 时才执行检查和保存逻辑.
        """
        if trainer.current_epoch >= self.start_epoch:
            # --- 关键修改点：移除了最后一个 'None' 参数 ---
            super().on_train_epoch_end(trainer, pl_module)

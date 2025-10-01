# train.py

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import os
from src.utils.analysis_saver import ResultsSaver
from hydra.core.hydra_config import HydraConfig  # <--- 核心改动 1: 导入HydraConfig


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    使用Hydra配置驱动的训练入口函数.
    """
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------")

    pl.seed_everything(cfg.seed, workers=True)

    print("--> Instantiating DataModule...")
    datamodule = hydra.utils.instantiate(cfg.data)

    print("--> Instantiating System (LightningModule)...")
    system = hydra.utils.instantiate(cfg.system)

    print("--> Instantiating Callbacks...")
    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))

    print("--> Instantiating Logger...")
    logger = hydra.utils.instantiate(cfg.logger)

    print("--> Instantiating Trainer...")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial"
    )

    print("--> Starting training...")
    trainer.fit(system, datamodule=datamodule)

    if cfg.get("test_after_training", False):
        print("--> Starting testing...")

        # --- 核心改动 2: 使用HydraConfig安全地获取输出目录 ---
        save_dir = HydraConfig.get().run.dir
        fold = cfg.data.fold
        results_saver_callback = ResultsSaver(save_dir=save_dir, fold=fold)

        # 将回调添加到trainer
        trainer.callbacks.append(results_saver_callback)

        # 从回调列表中找到 ModelCheckpoint
        checkpoint_callback = None
        for cb in trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                checkpoint_callback = cb
                break

        if checkpoint_callback and checkpoint_callback.best_model_path:
            best_ckpt_path = checkpoint_callback.best_model_path
            print(f"--> Found best model, testing with checkpoint: {best_ckpt_path}")
            trainer.test(datamodule=datamodule, ckpt_path='best')
        else:
            print("--> WARNING: Could not find the best checkpoint path. Testing with the model's last state.")
            trainer.test(model=system, datamodule=datamodule)

    print("--> Training finished!")


if __name__ == "__main__":
    train()
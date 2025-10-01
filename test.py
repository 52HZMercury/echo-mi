import torch
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from src.utils.analysis_saver import ResultsSaver # 导入回调
import os

@hydra.main(config_path="configs", config_name="test", version_base=None)
def test(cfg: DictConfig) -> None:
    """
    使用Hydra配置驱动的测试入口函数.

    Args:
        cfg (DictConfig): 从YAML文件加载的Hydra配置对象.
    """
    # 设定随机种子
    pl.seed_everything(cfg.seed)

    # 实例化数据模块
    datamodule = hydra.utils.instantiate(cfg.data)

    # 实例化系统 (LightningModule)
    system = hydra.utils.instantiate(cfg.system)

    # --- 解决方案: 从配置中获取信息并实例化 ResultsSaver ---
    # 从checkpoint路径推断出实验的根目录
    # .../outputs/EXP_NAME/RUN_TIME/checkpoints/model.ckpt -> .../outputs/EXP_NAME/RUN_TIME/
    checkpoint_dir = os.path.dirname(cfg.checkpoint_path)
    save_dir = os.path.dirname(checkpoint_dir)
    fold = cfg.data.fold
    results_saver_callback = ResultsSaver(save_dir=save_dir, fold=fold)

    # 实例化训练器，并传入回调
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[results_saver_callback],
        _convert_="partial"
    )

    # 运行测试
    print(f"--> Starting testing with checkpoint: {cfg.checkpoint_path}")
    trainer.test(model=system, datamodule=datamodule, ckpt_path=cfg.checkpoint_path)
    print("--> Testing finished!")


if __name__ == "__main__":
    test()
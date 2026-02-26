# src/utils/analysis_saver.py

import os
import pandas as pd
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


class ResultsSaver(Callback):
    """
    一个在测试结束后自动保存详细预测结果、特征向量和评估指标的回调。
    (新版：由外部配置直接传入保存路径和折数，更加稳定)
    """

    def __init__(self, save_dir: str, fold: int):
        super().__init__()
        self.save_dir = os.path.join(save_dir, "test_results")
        self.fold = fold
        self.predictions = []
        self.features = {}

        os.makedirs(self.save_dir, exist_ok=True)
        self.features_dir = os.path.join(self.save_dir, "features")
        os.makedirs(self.features_dir, exist_ok=True)
        print(f"ResultsSaver initialized. Results will be saved to: {self.save_dir}")
        print(f"Current fold for testing: {self.fold}")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """在每个测试批次结束后，从test_step的返回中收集数据。"""
        # --- FIX: Add robustness check for empty outputs ---
        if outputs is None:
            print("Warning: Received empty outputs from test_step. Skipping batch.")
            return

        sample_ids = outputs['sample_ids']
        targets = outputs['targets'].cpu()
        preds = outputs['preds'].cpu()

        # Only process features if they exist and are not None
        if 'features' in outputs and outputs['features'] is not None:
            features = outputs['features'].cpu()

            # 二分类
            for i in range(len(sample_ids)):
                self.predictions.append({
                    'fold': self.fold,
                    'sample_id': sample_ids[i],
                    'true_label': targets[i].item(),
                    'predicted_prob': preds[i].item(),
                    'predicted_label': int(preds[i].item() > 0.5)
                })

            # 三分类
            # for i in range(len(sample_ids)):
            #     prob_vec = preds[i]  # 这是一个包含 3 个元素的张量
            #
            #     # 1. 获取预测标签 (概率最大的索引：0, 1, 或 2)
            #     pred_label = torch.argmax(prob_vec).item()
            #
            #     # 2. 提取三个独立的概率
            #     p0 = prob_vec[0].item()
            #     p1 = prob_vec[1].item()
            #     p2 = prob_vec[2].item()
            #
            #     self.predictions.append({
            #         'fold': self.fold,
            #         'sample_id': sample_ids[i],
            #         'true_label': int(targets[i].item()),
            #         'predicted_label': int(pred_label),
            #         'prob_0': p0,
            #         'prob_1': p1,
            #         'prob_2': p2,
            #         # predicted_score 可以存预测类别的概率值，或者最大概率
            #         'predicted_score': float(prob_vec[pred_label].item())
            #     })
            #

                self.features[sample_ids[i]] = features[i].numpy()
        else:
            # If features are not available, still save predictions
            print("Warning: 'features' key not found or is None in test_step output. Saving predictions only.")
            # 二分类
            for i in range(len(sample_ids)):
                self.predictions.append({
                    'fold': self.fold,
                    'sample_id': sample_ids[i],
                    'true_label': targets[i].item(),
                    'predicted_prob': preds[i].item(),
                    'predicted_label': int(preds[i].item() > 0.5)
                })


            # 三分类
            # for i in range(len(sample_ids)):
            #     prob_vec = preds[i]  # 这是一个包含 3 个元素的张量
            #
            #     # 1. 获取预测标签 (概率最大的索引：0, 1, 或 2)
            #     pred_label = torch.argmax(prob_vec).item()
            #
            #     # 2. 提取三个独立的概率
            #     p0 = prob_vec[0].item()
            #     p1 = prob_vec[1].item()
            #     p2 = prob_vec[2].item()
            #
            #     self.predictions.append({
            #         'fold': self.fold,
            #         'sample_id': sample_ids[i],
            #         'true_label': int(targets[i].item()),
            #         'predicted_label': int(pred_label),
            #         'prob_0': p0,
            #         'prob_1': p1,
            #         'prob_2': p2,
            #         # predicted_score 可以存预测类别的概率值，或者最大概率
            #         'predicted_score': float(prob_vec[pred_label].item())
            #     })


    def on_test_end(self, trainer, pl_module):
        """在所有测试批次结束后，将收集到的数据写入文件。"""
        if not self.predictions:
            print("No test predictions to save.")
            return

        pred_df = pd.DataFrame(self.predictions)
        pred_df.to_csv(os.path.join(self.save_dir, "predictions.csv"), index=False)
        print(f"Saved {len(self.predictions)} predictions to predictions.csv")

        if self.features:
            for sample_id, feature_vec in self.features.items():
                np.save(os.path.join(self.features_dir, f"{sample_id}.npy"), feature_vec)
            print(f"Saved {len(self.features)} feature vectors to features/ directory.")

        metrics = trainer.logged_metrics
        test_metrics = {k: v.item() for k, v in metrics.items() if k.startswith('test')}
        test_metrics['fold'] = self.fold

        metrics_df = pd.DataFrame([test_metrics])
        metrics_df.to_csv(os.path.join(self.save_dir, "evaluation_metrics.csv"), index=False)
        print(f"Saved evaluation metrics to evaluation_metrics.csv")

        self.predictions.clear()
        self.features.clear()

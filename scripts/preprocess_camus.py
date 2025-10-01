import os
import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.data.utils import process_video_frames

def preprocess_camus_dataset(source_dir, dest_dir, metadata_path):
    """
    预处理CAMUS数据集，从AVI视频中提取帧并保存为.pt文件。
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    metadata = pd.read_csv(metadata_path)

    os.makedirs(dest_dir / "A2C", exist_ok=True)
    os.makedirs(dest_dir / "A4C", exist_ok=True)

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing CAMUS"):
        patient_id = row['Number']

        # 处理 A2C
        a2c_video_path = source_dir / 'A2C' / f"{patient_id}.avi"
        a2c_pt_path = dest_dir / 'A2C' / f"{patient_id}.pt"
        if a2c_video_path.exists() and not a2c_pt_path.exists():
            try:
                start, end = int(row["Start_A2C"]), int(row["End_A2C"])
                tensor = process_video_frames(a2c_video_path, start, end)
                torch.save(tensor, a2c_pt_path)
            except Exception as e:
                print(f"Error processing {a2c_video_path}: {e}")

        # 处理 A4C
        a4c_video_path = source_dir / 'A4C' / f"{patient_id}.avi"
        a4c_pt_path = dest_dir / 'A4C' / f"{patient_id}.pt"
        if a4c_video_path.exists() and not a4c_pt_path.exists():
            try:
                start, end = int(row["Start_A4C"]), int(row["End_A4C"])
                tensor = process_video_frames(a4c_video_path, start, end)
                torch.save(tensor, a4c_pt_path)
            except Exception as e:
                print(f"Error processing {a4c_video_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CAMUS dataset.")
    parser.add_argument('--source_dir', type=str, required=True, help='Directory containing the raw AVI videos.')
    parser.add_argument('--dest_dir', type=str, required=True, help='Directory to save the processed .pt files.')
    parser.add_argument('--metadata_path', type=str, default='data/label_select160.csv', help='Path to the metadata CSV file.')

    args = parser.parse_args()

    preprocess_camus_dataset(args.source_dir, args.dest_dir, args.metadata_path)

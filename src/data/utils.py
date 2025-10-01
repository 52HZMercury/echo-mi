import cv2
import numpy as np
import torch
from pathlib import Path


def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    """裁剪并缩放图像"""
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    img = cv2.resize(img, res, interpolation=interpolation)
    return img


def mask_outside_ultrasound_avi(original_pixels: np.array) -> np.array:
    """对AVI视频中的超声波区域外的所有像素进行掩膜处理"""
    try:
        vid = np.copy(original_pixels)
        testarray = np.copy(original_pixels)

        frame_sum = testarray[0].astype(np.float32)
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_BGR2GRAY)
        frame_sum = np.where(frame_sum > 0, 1, 0)

        for i in range(testarray.shape[0]):
            frame = testarray[i, :, :, :].astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.where(frame > 0, 1, 0)
            frame_sum = np.add(frame_sum, frame)

        kernel = np.ones((3, 3), np.uint8)
        frame_sum = cv2.erode(np.uint8(frame_sum), kernel, iterations=10)
        frame_sum = np.where(frame_sum > 0, 1, 0)

        frame0 = testarray[0].astype(np.uint8)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        frame_last = testarray[-1].astype(np.uint8)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_BGR2GRAY)
        frame_diff = abs(np.subtract(frame0, frame_last))
        frame_diff = np.where(frame_diff > 0, 1, 0)
        frame_diff[0:20, 0:20] = 0

        frame_overlap = np.add(frame_sum, frame_diff)
        frame_overlap = np.where(frame_overlap > 1, 1, 0)
        frame_overlap = cv2.dilate(np.uint8(frame_overlap), kernel, iterations=10).astype(np.uint8)

        cv2.floodFill(frame_overlap, None, (0, 0), 100)
        frame_overlap = np.where(frame_overlap != 100, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(frame_overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame_overlap, [hull], -1, 255, -1)

        frame_overlap = np.array(frame_overlap, dtype=bool)

        for i in range(len(vid)):
            frame = vid[i, :, :, :].astype('uint8')
            frame = cv2.bitwise_and(frame, frame, mask=frame_overlap.astype(np.uint8))
            vid[i, :, :, :] = frame

        return vid
    except Exception as e:
        print(f"Masking failed: {e}. Returning original video.")
        return original_pixels


def process_video_frames(video_path, start_frame, end_frame, num_out_frames=16, video_size=224):
    """
    从视频文件中读取指定帧范围，进行预处理，并返回一个torch张量。
    """
    mean = torch.tensor([29.1106, 28.0768, 29.0964]).reshape(3, 1, 1, 1)
    std = torch.tensor([47.9892, 46.4569, 47.2008]).reshape(3, 1, 1, 1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from {video_path} for range {start_frame}-{end_frame}")

    frames = mask_outside_ultrasound_avi(np.array(frames))

    num_frames = len(frames)
    if num_frames < num_out_frames:
        # 循环填充
        indices = np.arange(num_frames)
        indices = np.tile(indices, (num_out_frames // num_frames) + 1)[:num_out_frames]
        frames = frames[indices]
    else:
        # 均匀采样
        indices = np.linspace(0, num_frames - 1, num_out_frames, dtype=int)
        frames = frames[indices]

    # 预处理
    x = np.zeros((num_out_frames, video_size, video_size, 3), dtype=np.float32)
    for i in range(num_out_frames):
        x[i] = crop_and_scale(frames[i], res=(video_size, video_size))

    x = torch.from_numpy(x).permute(3, 0, 1, 2)  # C, T, H, W
    x.sub_(mean).div_(std)  # 归一化

    return x

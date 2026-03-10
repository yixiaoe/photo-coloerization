"""
评估指标
包含 PSNR、SSIM 等图像质量指标
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_psnr(pred, target, data_range=None):
    """
    计算 PSNR (Peak Signal-to-Noise Ratio)

    Args:
        pred: 预测图像 (B, C, H, W) tensor 或 (H, W, C) numpy
        target: 目标图像，同上
        data_range: 数据范围

    Returns:
        psnr_value: PSNR 值 (dB)
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
    else:
        pred_np = pred
        target_np = target

    if len(pred_np.shape) == 4:
        # 批次维度，计算平均
        batch_psnr = 0.0
        for i in range(pred_np.shape[0]):
            p = pred_np[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            t = target_np[i].transpose(1, 2, 0)
            if data_range is None:
                data_range_val = max(t.max() - t.min(), 1e-8)
            else:
                data_range_val = data_range
            batch_psnr += psnr(t, p, data_range=data_range_val)
        return batch_psnr / pred_np.shape[0]
    else:
        if data_range is None:
            data_range = max(target_np.max() - target_np.min(), 1e-8)
        return psnr(target_np, pred_np, data_range=data_range)


def calculate_ssim(pred, target, data_range=None):
    """
    计算 SSIM (Structural Similarity Index)

    Args:
        pred: 预测图像 (B, C, H, W) tensor 或 (H, W, C) numpy
        target: 目标图像，同上
        data_range: 数据范围

    Returns:
        ssim_value: SSIM 值
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
    else:
        pred_np = pred
        target_np = target

    if len(pred_np.shape) == 4:
        batch_ssim = 0.0
        for i in range(pred_np.shape[0]):
            p = pred_np[i].transpose(1, 2, 0)
            t = target_np[i].transpose(1, 2, 0)
            if data_range is None:
                data_range_val = max(t.max() - t.min(), 1e-8)
            else:
                data_range_val = data_range
            batch_ssim += ssim(t, p, data_range=data_range_val, channel_axis=2)
        return batch_ssim / pred_np.shape[0]
    else:
        if data_range is None:
            data_range = max(target_np.max() - target_np.min(), 1e-8)
        return ssim(target_np, pred_np, data_range=data_range, channel_axis=2)


def calculate_l2_distance(pred_ab, target_ab):
    """
    计算 ab 通道的 L2 距离

    Args:
        pred_ab: 预测的 ab 通道 (B, 2, H, W)
        target_ab: 目标 ab 通道 (B, 2, H, W)

    Returns:
        l2_dist: 平均 L2 距离
    """
    if isinstance(pred_ab, torch.Tensor):
        diff = pred_ab - target_ab
        l2 = torch.sqrt((diff ** 2).sum(dim=1)).mean()
        return l2.item()
    else:
        diff = pred_ab - target_ab
        l2 = np.sqrt((diff ** 2).sum(axis=1)).mean()
        return l2

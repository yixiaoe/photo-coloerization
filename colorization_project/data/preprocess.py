"""
数据预处理工具
包含 RGB 和 LAB 色彩空间转换、归一化等功能
"""

import numpy as np
from skimage import color
import torch


def rgb_to_lab(rgb_image):
    """
    将 RGB 图像转换为 LAB 色彩空间

    Args:
        rgb_image: RGB 图像，范围 [0, 255] 或 [0, 1]

    Returns:
        lab_image: LAB 图像，L 范围 [0, 100]，ab 范围 [-128, 127]
    """
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0

    lab_image = color.rgb2lab(rgb_image)
    return lab_image


def lab_to_rgb(lab_image):
    """
    将 LAB 图像转换为 RGB 色彩空间

    Args:
        lab_image: LAB 图像

    Returns:
        rgb_image: RGB 图像，范围 [0, 1]
    """
    rgb_image = color.lab2rgb(lab_image)
    return np.clip(rgb_image, 0, 1)


def normalize_lab(lab_image, l_cent=50.0, l_norm=100.0, ab_norm=110.0):
    """
    归一化 LAB 图像

    Args:
        lab_image: LAB 图像 (H, W, 3)
        l_cent: L 通道中心值
        l_norm: L 通道归一化因子
        ab_norm: ab 通道归一化因子

    Returns:
        normalized_lab: 归一化后的 LAB 图像
    """
    lab_normalized = lab_image.copy()
    lab_normalized[:, :, 0] = (lab_image[:, :, 0] - l_cent) / l_norm
    lab_normalized[:, :, 1:] = lab_image[:, :, 1:] / ab_norm
    return lab_normalized


def denormalize_lab(lab_normalized, l_cent=50.0, l_norm=100.0, ab_norm=110.0):
    """
    反归一化 LAB 图像

    Args:
        lab_normalized: 归一化的 LAB 图像
        l_cent: L 通道中心值
        l_norm: L 通道归一化因子
        ab_norm: ab 通道归一化因子

    Returns:
        lab_image: 反归一化后的 LAB 图像
    """
    lab_image = lab_normalized.copy()
    if isinstance(lab_normalized, torch.Tensor):
        lab_image = lab_normalized.clone()
        if len(lab_image.shape) == 4:  # (B, C, H, W)
            lab_image[:, 0, :, :] = lab_normalized[:, 0, :, :] * l_norm + l_cent
            lab_image[:, 1:, :, :] = lab_normalized[:, 1:, :, :] * ab_norm
        else:  # (C, H, W)
            lab_image[0, :, :] = lab_normalized[0, :, :] * l_norm + l_cent
            lab_image[1:, :, :] = lab_normalized[1:, :, :] * ab_norm
    else:
        lab_image[:, :, 0] = lab_normalized[:, :, 0] * l_norm + l_cent
        lab_image[:, :, 1:] = lab_normalized[:, :, 1:] * ab_norm
    return lab_image


def split_lab_channels(lab_image):
    """
    分离 LAB 图像的 L 和 ab 通道

    Args:
        lab_image: LAB 图像 (H, W, 3) 或 (B, 3, H, W)

    Returns:
        l_channel: L 通道
        ab_channels: ab 通道
    """
    if isinstance(lab_image, torch.Tensor):
        if len(lab_image.shape) == 4:  # (B, C, H, W)
            l_channel = lab_image[:, 0:1, :, :]
            ab_channels = lab_image[:, 1:3, :, :]
        else:  # (C, H, W)
            l_channel = lab_image[0:1, :, :]
            ab_channels = lab_image[1:3, :, :]
    else:
        l_channel = lab_image[:, :, 0]
        ab_channels = lab_image[:, :, 1:3]
    return l_channel, ab_channels


def merge_lab_channels(l_channel, ab_channels):
    """
    合并 L 和 ab 通道为完整的 LAB 图像

    Args:
        l_channel: L 通道
        ab_channels: ab 通道

    Returns:
        lab_image: 完整的 LAB 图像
    """
    if isinstance(l_channel, torch.Tensor):
        lab_image = torch.cat([l_channel, ab_channels], dim=1 if len(l_channel.shape) == 4 else 0)
    else:
        lab_image = np.concatenate([l_channel[..., np.newaxis] if len(l_channel.shape) == 2 else l_channel,
                                     ab_channels], axis=-1)
    return lab_image

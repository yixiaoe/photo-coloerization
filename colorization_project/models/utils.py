"""
模型工具函数
"""

import torch
import numpy as np
from scipy.spatial import distance


def load_ab_quantization_centers(pts_in_hull_path=None):
    """
    加载 ab 空间的 313 个聚类中心

    Args:
        pts_in_hull_path: pts_in_hull.npy 文件路径

    Returns:
        ab_centers: (313, 2) 的 ab 聚类中心
    """
    if pts_in_hull_path is None:
        # 如果没有提供文件，生成一个简化的网格
        # 实际使用时应该使用论文提供的预计算聚类中心
        a_range = np.linspace(-110, 110, 18)
        b_range = np.linspace(-110, 110, 18)
        aa, bb = np.meshgrid(a_range, b_range)
        ab_centers = np.stack([aa.flatten(), bb.flatten()], axis=1)
        # 只保留在色域内的点（简化处理）
        ab_centers = ab_centers[:313]
    else:
        ab_centers = np.load(pts_in_hull_path)

    return ab_centers


def ab_to_class(ab_values, ab_centers, sigma=5.0):
    """
    将 ab 值转换为类别分布（soft encoding）

    Args:
        ab_values: (H, W, 2) 的 ab 值
        ab_centers: (313, 2) 的聚类中心
        sigma: 高斯核的标准差

    Returns:
        class_distribution: (H, W, 313) 的类别分布
    """
    H, W = ab_values.shape[:2]
    ab_flat = ab_values.reshape(-1, 2)  # (H*W, 2)

    # 计算每个像素到所有聚类中心的距离
    dists = distance.cdist(ab_flat, ab_centers, metric='euclidean')  # (H*W, 313)

    # 使用高斯核转换为概率分布
    weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

    class_distribution = weights.reshape(H, W, -1)
    return class_distribution


def class_to_ab(class_probs, ab_centers, temperature=0.38):
    """
    将类别概率分布转换为 ab 值

    Args:
        class_probs: (B, 313, H, W) 的类别概率
        ab_centers: (313, 2) 的聚类中心
        temperature: annealed-mean 的温度参数

    Returns:
        ab_values: (B, 2, H, W) 的 ab 值
    """
    if isinstance(class_probs, torch.Tensor):
        device = class_probs.device
        ab_centers_tensor = torch.from_numpy(ab_centers).float().to(device)

        # Annealed-mean: 对概率进行温度调整
        adjusted_probs = class_probs ** (1.0 / temperature)
        adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)

        # 加权平均
        B, _, H, W = class_probs.shape
        adjusted_probs_flat = adjusted_probs.permute(0, 2, 3, 1).reshape(-1, 313)  # (B*H*W, 313)
        ab_flat = torch.matmul(adjusted_probs_flat, ab_centers_tensor)  # (B*H*W, 2)
        ab_values = ab_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)  # (B, 2, H, W)
    else:
        # NumPy 版本
        adjusted_probs = class_probs ** (1.0 / temperature)
        adjusted_probs = adjusted_probs / (adjusted_probs.sum(axis=1, keepdims=True) + 1e-8)

        B, _, H, W = class_probs.shape
        adjusted_probs_flat = adjusted_probs.transpose(0, 2, 3, 1).reshape(-1, 313)
        ab_flat = np.dot(adjusted_probs_flat, ab_centers)
        ab_values = ab_flat.reshape(B, H, W, 2).transpose(0, 3, 1, 2)

    return ab_values


def compute_class_weights(dataset, ab_centers, num_samples=10000):
    """
    从数据集统计计算类别权重

    Args:
        dataset: 数据集对象
        ab_centers: ab 聚类中心
        num_samples: 采样数量

    Returns:
        class_weights: (313,) 的类别权重
    """
    class_counts = np.zeros(len(ab_centers))

    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in sample_indices:
        _, ab_channels = dataset[idx]
        ab_np = ab_channels.permute(1, 2, 0).numpy()  # (H, W, 2)

        # 转换为类别
        class_dist = ab_to_class(ab_np, ab_centers)
        class_counts += class_dist.sum(axis=(0, 1))

    # 归一化为概率
    class_probs = class_counts / class_counts.sum()

    return class_probs

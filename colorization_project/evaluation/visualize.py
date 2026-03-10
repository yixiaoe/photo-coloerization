"""
可视化工具
用于展示上色结果
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import torch


def visualize_results(l_channel, pred_ab, target_ab=None, save_path=None, title=None):
    """
    可视化上色结果：灰度图 | 预测上色 | 真实彩色（如有）

    Args:
        l_channel: L 通道 (1, H, W) 或 (H, W)，归一化前的值
        pred_ab: 预测的 ab 通道 (2, H, W)
        target_ab: 真实 ab 通道 (2, H, W)，可选
        save_path: 保存路径
        title: 图片标题
    """
    # 转换为 numpy
    if isinstance(l_channel, torch.Tensor):
        l_channel = l_channel.detach().cpu().numpy()
    if isinstance(pred_ab, torch.Tensor):
        pred_ab = pred_ab.detach().cpu().numpy()
    if target_ab is not None and isinstance(target_ab, torch.Tensor):
        target_ab = target_ab.detach().cpu().numpy()

    # 调整维度
    if len(l_channel.shape) == 3:
        l_channel = l_channel[0]  # (1, H, W) -> (H, W)
    if len(pred_ab.shape) == 3:
        pred_ab = pred_ab.transpose(1, 2, 0)  # (2, H, W) -> (H, W, 2)
    if target_ab is not None and len(target_ab.shape) == 3:
        target_ab = target_ab.transpose(1, 2, 0)

    # 组装 LAB 图像
    H, W = l_channel.shape
    pred_lab = np.zeros((H, W, 3))
    pred_lab[:, :, 0] = l_channel
    pred_lab[:, :, 1:] = pred_ab
    pred_rgb = np.clip(color.lab2rgb(pred_lab), 0, 1)

    # 灰度图
    gray_rgb = np.stack([l_channel / 100.0] * 3, axis=-1)

    num_cols = 3 if target_ab is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

    # 灰度图
    axes[0].imshow(gray_rgb, cmap='gray')
    axes[0].set_title('灰度输入')
    axes[0].axis('off')

    # 预测上色
    axes[1].imshow(pred_rgb)
    axes[1].set_title('预测上色')
    axes[1].axis('off')

    # 真实彩色
    if target_ab is not None:
        target_lab = np.zeros((H, W, 3))
        target_lab[:, :, 0] = l_channel
        target_lab[:, :, 1:] = target_ab
        target_rgb = np.clip(color.lab2rgb(target_lab), 0, 1)

        axes[2].imshow(target_rgb)
        axes[2].set_title('真实彩色')
        axes[2].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_batch(l_channels, pred_abs, target_abs=None, save_dir=None,
                    num_samples=4):
    """
    可视化一个 batch 中的多个样本

    Args:
        l_channels: L 通道 (B, 1, H, W)
        pred_abs: 预测 ab 通道 (B, 2, H, W)
        target_abs: 真实 ab 通道 (B, 2, H, W)，可选
        save_dir: 保存目录
        num_samples: 展示样本数量
    """
    if isinstance(l_channels, torch.Tensor):
        l_channels = l_channels.detach().cpu().numpy()
    if isinstance(pred_abs, torch.Tensor):
        pred_abs = pred_abs.detach().cpu().numpy()
    if target_abs is not None and isinstance(target_abs, torch.Tensor):
        target_abs = target_abs.detach().cpu().numpy()

    batch_size = min(num_samples, l_channels.shape[0])

    for i in range(batch_size):
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"sample_{i}.png")

        target_ab_i = target_abs[i] if target_abs is not None else None
        visualize_results(
            l_channels[i], pred_abs[i], target_ab_i,
            save_path=save_path,
            title=f"Sample {i+1}"
        )


def plot_training_curves(log_dir, save_path=None):
    """
    绘制训练曲线（从 TensorBoard 日志读取）
    也可以直接从训练中记录的数据绘制

    Args:
        log_dir: TensorBoard 日志目录
        save_path: 保存路径
    """
    print(f"请使用 TensorBoard 查看训练曲线：")
    print(f"  tensorboard --logdir {log_dir}")
    print(f"  然后在浏览器中打开 http://localhost:6006")

"""
推理脚本
用于对新的灰度图像进行上色
"""

import os
import torch
import numpy as np
from PIL import Image
from skimage import color
from torchvision import transforms

from models.colorization_net import ColorizationNet


def load_model(checkpoint_path, device='cuda', num_classes=313):
    """
    加载训练好的模型

    Args:
        checkpoint_path: 检查点路径
        device: 设备
        num_classes: 颜色类别数

    Returns:
        model: 加载权重后的模型
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = ColorizationNet(num_classes=num_classes).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, device


def preprocess_image(image_path, target_size=256):
    """
    预处理输入图像

    Args:
        image_path: 图像路径
        target_size: 目标大小

    Returns:
        l_channel: L 通道 tensor (1, 1, H, W)
        original_l: 原始 L 通道（用于合并输出）
        original_size: 原始图像大小
    """
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)

    # 调整大小
    image_resized = image.resize((target_size, target_size), Image.BILINEAR)
    image_np = np.array(image_resized).astype(np.float32) / 255.0

    # RGB 转 LAB
    lab_image = color.rgb2lab(image_np)

    # 提取 L 通道
    l_channel = lab_image[:, :, 0]  # (H, W)

    # 转换为 tensor
    l_tensor = torch.from_numpy(l_channel).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)

    return l_tensor, l_channel, original_size


def postprocess_output(l_channel, pred_ab, original_size=None):
    """
    后处理模型输出

    Args:
        l_channel: L 通道 numpy (H, W)
        pred_ab: 预测的 ab 通道 tensor (1, 2, H, W)
        original_size: 原始图像大小 (W, H)

    Returns:
        rgb_image: RGB 图像 numpy (H, W, 3)，范围 [0, 255]
    """
    # tensor 转 numpy
    if isinstance(pred_ab, torch.Tensor):
        pred_ab = pred_ab.detach().cpu().numpy()

    pred_ab = pred_ab[0].transpose(1, 2, 0)  # (2, H, W) -> (H, W, 2)

    # 组装 LAB 图像
    H, W = l_channel.shape
    lab_image = np.zeros((H, W, 3))
    lab_image[:, :, 0] = l_channel
    lab_image[:, :, 1:] = pred_ab

    # LAB 转 RGB
    rgb_image = np.clip(color.lab2rgb(lab_image), 0, 1)

    # 恢复原始大小
    if original_size is not None:
        rgb_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))
        rgb_pil = rgb_pil.resize(original_size, Image.BILINEAR)
        rgb_image = np.array(rgb_pil).astype(np.float32) / 255.0

    return (rgb_image * 255).astype(np.uint8)


@torch.no_grad()
def colorize_image(image_path, model, device, target_size=256):
    """
    对单张灰度图像上色

    Args:
        image_path: 输入图像路径
        model: 训练好的模型
        device: 设备
        target_size: 处理大小

    Returns:
        rgb_output: 上色后的 RGB 图像 (H, W, 3)，范围 [0, 255]
    """
    l_tensor, l_channel, original_size = preprocess_image(image_path, target_size)
    l_tensor = l_tensor.to(device)

    # 模型推理
    pred_ab = model(l_tensor)

    # 后处理
    rgb_output = postprocess_output(l_channel, pred_ab, original_size)

    return rgb_output


@torch.no_grad()
def colorize_batch(image_paths, model, device, target_size=256, output_dir=None):
    """
    批量上色

    Args:
        image_paths: 图像路径列表
        model: 训练好的模型
        device: 设备
        target_size: 处理大小
        output_dir: 输出目录

    Returns:
        results: 上色结果列表
    """
    results = []

    for i, image_path in enumerate(image_paths):
        print(f"处理 [{i+1}/{len(image_paths)}]: {image_path}")

        rgb_output = colorize_image(image_path, model, device, target_size)
        results.append(rgb_output)

        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{filename}_colorized.png")
            Image.fromarray(rgb_output).save(output_path)
            print(f"  保存到: {output_path}")

    return results

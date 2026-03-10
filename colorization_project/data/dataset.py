"""
数据集类定义
支持 COCO 和 ImageNet 数据集的加载和预处理
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from .preprocess import rgb_to_lab, normalize_lab, split_lab_channels


class ColorizationDataset(Dataset):
    """
    图像上色数据集类
    """

    def __init__(self, image_dir, image_size=256, crop_size=224, mode='train'):
        """
        Args:
            image_dir: 图像目录路径
            image_size: 调整后的图像大小
            crop_size: 裁剪大小（训练时使用）
            mode: 'train' 或 'val'
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.crop_size = crop_size
        self.mode = mode

        # 获取所有图像文件
        self.image_files = self._get_image_files()

        # 数据增强
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(crop_size),
            ])

    def _get_image_files(self):
        """获取目录下所有图像文件"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []

        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    image_files.append(os.path.join(root, file))

        return sorted(image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        返回一个样本

        Returns:
            l_channel: 归一化的 L 通道 (1, H, W)
            ab_channels: 归一化的 ab 通道 (2, H, W)
        """
        # 加载图像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')

        # 数据增强
        image = self.transform(image)

        # 转换为 numpy 数组
        image_np = np.array(image).astype(np.float32) / 255.0

        # RGB 转 LAB
        lab_image = rgb_to_lab(image_np)

        # 归一化
        lab_normalized = normalize_lab(lab_image)

        # 分离 L 和 ab 通道
        l_channel = lab_normalized[:, :, 0]
        ab_channels = lab_normalized[:, :, 1:3]

        # 转换为 torch tensor 并调整维度 (H, W, C) -> (C, H, W)
        l_channel = torch.from_numpy(l_channel).unsqueeze(0).float()  # (1, H, W)
        ab_channels = torch.from_numpy(ab_channels).permute(2, 0, 1).float()  # (2, H, W)

        return l_channel, ab_channels


class COCOColorizationDataset(ColorizationDataset):
    """COCO 数据集"""

    def __init__(self, coco_root, split='train2017', **kwargs):
        """
        Args:
            coco_root: COCO 数据集根目录
            split: 'train2017' 或 'val2017'
        """
        image_dir = os.path.join(coco_root, split)
        mode = 'train' if 'train' in split else 'val'
        super().__init__(image_dir, mode=mode, **kwargs)


class ImageNetColorizationDataset(ColorizationDataset):
    """ImageNet 数据集"""

    def __init__(self, imagenet_root, split='train', **kwargs):
        """
        Args:
            imagenet_root: ImageNet 数据集根目录
            split: 'train' 或 'val'
        """
        image_dir = os.path.join(imagenet_root, split)
        super().__init__(image_dir, mode=split, **kwargs)

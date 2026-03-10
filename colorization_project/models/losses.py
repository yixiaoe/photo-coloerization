"""
损失函数定义
包含加权交叉熵损失和可选的感知损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ColorRebalancedLoss(nn.Module):
    """
    颜色重平衡交叉熵损失
    对稀有颜色赋予更高权重
    """

    def __init__(self, num_classes=313, lambda_=0.5):
        super(ColorRebalancedLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_ = lambda_

        # 这里需要预先计算的类别权重
        # 实际使用时应该从数据集统计得到
        # 这里先用均匀权重作为占位符
        self.register_buffer('class_weights', torch.ones(num_classes))

    def set_class_weights(self, class_probs):
        """
        根据数据集统计设置类别权重

        Args:
            class_probs: 每个类别的出现概率 (num_classes,)
        """
        # 权重计算公式: w = ((1-λ) * p + λ/Q)^(-1)
        weights = ((1 - self.lambda_) * class_probs + self.lambda_ / self.num_classes) ** (-1)
        # 归一化
        weights = weights / weights.sum() * self.num_classes
        self.class_weights = weights

    def forward(self, pred_logits, target_ab, target_classes=None):
        """
        计算损失

        Args:
            pred_logits: 模型预测的类别 logits (B, num_classes, H, W)
            target_ab: 目标 ab 通道 (B, 2, H, W)
            target_classes: 目标类别索引 (B, H, W)，如果为 None 则需要从 target_ab 计算

        Returns:
            loss: 加权交叉熵损失
        """
        if target_classes is None:
            # 这里需要将 ab 值转换为类别索引
            # 实际实现需要使用预计算的 ab 聚类中心
            # 这里简化处理
            raise NotImplementedError("需要提供 target_classes 或实现 ab 到类别的转换")

        # 计算交叉熵损失
        loss = F.cross_entropy(
            pred_logits,
            target_classes,
            weight=self.class_weights,
            reduction='mean'
        )

        return loss


class SimplifiedColorLoss(nn.Module):
    """
    简化的颜色损失（直接回归 ab 通道）
    用于快速原型验证
    """

    def __init__(self):
        super(SimplifiedColorLoss, self).__init__()

    def forward(self, pred_ab, target_ab):
        """
        计算 L2 损失

        Args:
            pred_ab: 预测的 ab 通道 (B, 2, H, W)
            target_ab: 目标 ab 通道 (B, 2, H, W)

        Returns:
            loss: L2 损失
        """
        loss = F.mse_loss(pred_ab, target_ab)
        return loss


class PerceptualLoss(nn.Module):
    """
    感知损失（可选）
    使用预训练 VGG 提取特征
    """

    def __init__(self, feature_layers=[3, 8, 15, 22]):
        super(PerceptualLoss, self).__init__()
        from torchvision import models

        vgg = models.vgg16(pretrained=True).features
        self.feature_extractors = nn.ModuleList()

        prev_layer = 0
        for layer_idx in feature_layers:
            self.feature_extractors.append(vgg[prev_layer:layer_idx + 1])
            prev_layer = layer_idx + 1

        # 冻结 VGG 参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred_rgb, target_rgb):
        """
        计算感知损失

        Args:
            pred_rgb: 预测的 RGB 图像 (B, 3, H, W)
            target_rgb: 目标 RGB 图像 (B, 3, H, W)

        Returns:
            loss: 感知损失
        """
        loss = 0.0

        pred_features = pred_rgb
        target_features = target_rgb

        for extractor in self.feature_extractors:
            pred_features = extractor(pred_features)
            target_features = extractor(target_features)
            loss += F.mse_loss(pred_features, target_features)

        return loss / len(self.feature_extractors)

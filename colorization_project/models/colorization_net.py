"""
图像上色网络模型
基于 ECCV 2016 论文的实现
"""

import torch
import torch.nn as nn


class ColorizationNet(nn.Module):
    """
    图像上色网络
    输入：L 通道 (1, H, W)
    输出：ab 通道分布 (313, H, W) 或直接输出 ab (2, H, W)
    """

    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=313):
        super(ColorizationNet, self).__init__()

        self.num_classes = num_classes
        self.l_cent = 50.0
        self.l_norm = 100.0
        self.ab_norm = 110.0

        # 编码器 Block 1: 1 -> 64, stride=2
        model1 = [
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64),
        ]

        # 编码器 Block 2: 64 -> 128, stride=2
        model2 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]

        # 编码器 Block 3: 128 -> 256, stride=2
        model3 = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        ]

        # 编码器 Block 4: 256 -> 512, stride=1
        model4 = [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # 中间层 Block 5: 512 -> 512, dilation=2
        model5 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # 中间层 Block 6: 512 -> 512, dilation=2
        model6 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # 中间层 Block 7: 512 -> 512, stride=1
        model7 = [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # 解码器 Block 8: 512 -> 256, upsample
        model8 = [
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            # 输出层：256 -> num_classes
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(num_classes, 2, kernel_size=1, padding=0, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def normalize_l(self, in_l):
        """归一化 L 通道"""
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_ab(self, in_ab):
        """反归一化 ab 通道"""
        return in_ab * self.ab_norm

    def forward(self, input_l, return_class_probs=False):
        """
        前向传播

        Args:
            input_l: 输入 L 通道 (B, 1, H, W)
            return_class_probs: 是否返回类别概率分布

        Returns:
            如果 return_class_probs=False: ab 通道预测 (B, 2, H, W)
            如果 return_class_probs=True: (ab 通道, 类别概率)
        """
        # 归一化输入
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)  # (B, 313, H/4, W/4)

        # Softmax 得到类别概率
        class_probs = self.softmax(conv8_3)

        # 转换为 ab 通道
        out_reg = self.model_out(class_probs)  # (B, 2, H/4, W/4)

        # 上采样到原始分辨率
        out_ab = self.unnormalize_ab(self.upsample4(out_reg))  # (B, 2, H, W)

        if return_class_probs:
            return out_ab, class_probs
        else:
            return out_ab

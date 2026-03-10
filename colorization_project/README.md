# 黑白照片上色神经网络项目实现方案

## Context（背景）

这是一个计算机视觉课程项目，目标是从零训练一个黑白照片自动上色的深度学习模型。

**用户选择：**
- 方案：经典 CNN 分类方法（基于 2016 年论文）
- 功能：纯自动上色，无需用户交互
- 资源：单卡消费级 GPU（RTX 3090/4090），2 周内完成
- 数据：COCO 和 ImageNet-1000 数据集

**现有资源：**
- 参考代码：colorization-master（包含 ECCV16 模型实现）
- 参考论文：paper/ 文件夹中的 4 篇论文
- 开发环境：PyTorch

## 项目架构设计

### 1. 目录结构

```
CV/
├── paper/                          # 参考论文（已存在）
├── colorization-master/            # 参考代码（已存在）
├── colorization_project/           # 新建：主项目目录
│   ├── data/                       # 数据相关
│   │   ├── __init__.py
│   │   ├── dataset.py              # 数据集类
│   │   ├── download_data.py        # 数据下载脚本
│   │   └── preprocess.py           # 数据预处理工具
│   ├── models/                     # 模型定义
│   │   ├── __init__.py
│   │   ├── colorization_net.py     # 主网络架构
│   │   ├── losses.py               # 损失函数
│   │   └── utils.py                # 模型工具函数
│   ├── training/                   # 训练相关
│   │   ├── __init__.py
│   │   ├── trainer.py              # 训练器类
│   │   └── config.py               # 训练配置
│   ├── evaluation/                 # 评估相关
│   │   ├── __init__.py
│   │   ├── metrics.py              # 评估指标
│   │   └── visualize.py            # 可视化工具
│   ├── inference/                  # 推理相关
│   │   ├── __init__.py
│   │   └── colorize.py             # 推理脚本
│   ├── train.py                    # 训练入口脚本
│   ├── test.py                     # 测试脚本
│   └── requirements.txt            # 依赖列表
├── datasets/                       # 新建：数据集存放目录
│   ├── coco/                       # COCO 数据集
│   └── imagenet/                   # ImageNet 数据集
├── checkpoints/                    # 新建：模型检查点
└── outputs/                        # 新建：输出结果
    ├── logs/                       # 训练日志
    └── results/                    # 上色结果
```

### 2. 核心模型架构

基于 ECCV16 论文和现有代码，模型架构如下：

**输入输出：**
- 输入：灰度图像的 L 通道（1×H×W）
- 输出：预测的 ab 通道（2×H×W）
- 色彩空间：LAB（L 范围 0-100，ab 范围 -110 到 110）

**网络结构：**
```
编码器（下采样）：
- Conv Block 1: 1→64 channels, stride=2 (H/2, W/2)
- Conv Block 2: 64→128 channels, stride=2 (H/4, W/4)
- Conv Block 3: 128→256 channels, stride=2 (H/8, W/8)
- Conv Block 4: 256→512 channels, stride=1

中间层（空洞卷积保持分辨率）：
- Conv Block 5: 512→512, dilation=2
- Conv Block 6: 512→512, dilation=2
- Conv Block 7: 512→512, stride=1

解码器（上采样）：
- Conv Block 8: 512→256, ConvTranspose (H/4, W/4)
- Output Layer: 256→313 (颜色分类)
- Softmax + 1×1 Conv: 313→2 (ab 通道)
- Upsample 4x: 恢复到原始分辨率
```

**关键技术点：**
1. 将 ab 通道量化为 313 个颜色类别（基于 ab 空间的聚类）
2. 使用 Softmax 输出颜色分布，而非直接回归
3. 类别重平衡：对稀有颜色赋予更高权重
4. BatchNorm 用于稳定训练

### 3. 数据处理流程

**数据下载：**
- COCO 2017 Train/Val：~25GB（118K 训练图像）
- ImageNet ILSVRC2012：~150GB（1.28M 训练图像）
- 建议先用 COCO 训练原型，验证后再加入 ImageNet

**数据预处理：**
```python
1. 加载 RGB 图像
2. 调整大小到 256×256（训练时）
3. 转换到 LAB 色彩空间
4. 归一化：
   - L: (L - 50) / 100
   - ab: ab / 110
5. 数据增强：
   - 随机水平翻转
   - 随机裁剪（224×224）
   - 可选：轻微旋转、缩放
```

**颜色量化：**
- 使用预计算的 313 个 ab 聚类中心（pts_in_hull.npy）
- 每个像素的 ab 值映射到最近的聚类中心
- 生成 soft encoding（高斯核平滑）而非 hard label

### 4. 训练策略

**损失函数：**
```python
主损失：加权交叉熵
- 对每个像素的 313 类颜色分类
- 类别权重：稀有颜色权重更高（基于数据集统计）
- 权重计算：w = ((1-λ) * p + λ/Q)^(-1)
  其中 p 是类别频率，λ=0.5，Q=313

可选：感知损失（提升视觉质量）
- 使用预训练 VGG 提取特征
- 比较预测图像和真实图像的特征距离
```

**优化器配置：**
- 优化器：Adam (β1=0.9, β2=0.999)
- 初始学习率：1e-4
- 学习率调度：
  - Warmup：前 5 epochs 线性增长
  - CosineAnnealing 或 ReduceLROnPlateau
- 批次大小：16-32（根据 GPU 显存调整）
- 训练轮数：50-100 epochs

**训练技巧：**
1. 梯度裁剪：防止梯度爆炸
2. 混合精度训练（FP16）：加速训练，节省显存
3. 每 5 epochs 保存检查点
4. 每 1000 iterations 验证并可视化
5. TensorBoard 记录损失、学习率、样本图像

### 5. 评估指标

**定量指标：**
- PSNR（Peak Signal-to-Noise Ratio）：衡量像素级准确度
- SSIM（Structural Similarity Index）：衡量结构相似度
- 色彩准确度：ab 通道的 L2 距离

**定性评估：**
- 随机采样验证集图像进行可视化
- 对比：灰度输入 | 预测上色 | 真实彩色
- 关注：肤色、天空、植被等常见物体的上色质量

### 6. 推理流程

```python
1. 加载训练好的模型
2. 读取灰度图像（或 RGB 转灰度）
3. 转换到 LAB 空间，提取 L 通道
4. 归一化并输入模型
5. 模型输出 ab 通道预测
6. 反归一化 ab 通道
7. 合并 L 和 ab，转换回 RGB
8. 保存结果图像
```

### 7. 实现时间线（2 周）

**第 1-2 天：环境搭建和数据准备**
- 创建项目目录结构
- 安装依赖：PyTorch, torchvision, scikit-image, tensorboard
- 下载 COCO 数据集（先用小规模验证）
- 实现数据加载和预处理代码

**第 3-4 天：模型实现**
- 实现 ColorizationNet 模型（参考 eccv16.py）
- 实现损失函数（加权交叉熵）
- 编写训练器类（Trainer）
- 单元测试：验证数据流和模型前向传播

**第 5-7 天：初步训练**
- 在 COCO 子集上训练（10K 图像）
- 调试训练流程
- 监控损失曲线和样本输出
- 调整超参数（学习率、批次大小）

**第 8-10 天：完整训练**
- 在完整 COCO 数据集上训练
- 可选：加入 ImageNet 数据（如果时间充裕）
- 训练 50+ epochs
- 定期评估和保存最佳模型

**第 11-12 天：评估和优化**
- 在验证集上计算 PSNR/SSIM
- 可视化大量样本，分析失败案例
- 可选优化：调整损失权重、增加感知损失
- 编写推理脚本

**第 13-14 天：文档和展示**
- 整理代码和注释
- 生成展示用的上色结果
- 编写项目报告（方法、实验、结果）
- 准备演示材料

### 8. 关键文件实现要点

**models/colorization_net.py：**
- 参考 [colorization-master/colorizers/eccv16.py](../colorization-master/colorizers/eccv16.py)
- 修改为可训练版本（移除预训练加载）
- 添加权重初始化（Xavier/Kaiming）

**data/dataset.py：**
- 继承 torch.utils.data.Dataset
- 实现 RGB→LAB 转换
- 实现颜色量化（ab→313 类）
- 支持数据增强

**training/trainer.py：**
- 封装训练循环
- 实现检查点保存/恢复
- 集成 TensorBoard 日志
- 支持分布式训练（可选）

**train.py：**
- 解析命令行参数
- 初始化数据加载器、模型、优化器
- 调用 Trainer 开始训练

### 9. 预期效果

**训练资源估算：**
- 单卡 RTX 3090（24GB）：批次大小 32，约 2-3 天训练 50 epochs
- 单卡 RTX 4090（24GB）：批次大小 32-48，约 1.5-2 天训练 50 epochs

**性能预期：**
- PSNR：24-28 dB（验证集）
- SSIM：0.85-0.92
- 视觉效果：大部分场景合理上色，部分复杂场景可能偏灰或颜色不准

**常见问题：**
- 偏灰问题：增加稀有颜色权重，或加入对抗损失
- 颜色溢出：检查归一化和反归一化是否正确
- 训练不稳定：降低学习率，增加 warmup

## 验证方案

**训练过程验证：**
1. 检查损失是否持续下降
2. 每 1000 steps 可视化训练样本
3. 监控验证集 PSNR/SSIM 曲线
4. 确保没有过拟合（训练/验证损失差距）

**最终模型验证：**
1. 在验证集上计算定量指标
2. 对 colorization-master/imgs/ 中的测试图像进行上色
3. 与 ECCV16 预训练模型对比效果
4. 测试不同类型图像：人像、风景、建筑、动物

**成功标准：**
- 模型能够收敛（损失稳定下降）
- 验证集 PSNR > 24 dB
- 视觉效果：天空蓝色、草地绿色、肤色自然
- 推理速度：单张 256×256 图像 < 0.1 秒（GPU）

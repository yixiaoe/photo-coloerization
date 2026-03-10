"""
训练配置
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """训练配置参数"""

    # 数据相关
    data_root: str = "../datasets"
    dataset_type: str = "coco"  # 'coco' 或 'imagenet'
    image_size: int = 256
    crop_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

    # 模型相关
    num_classes: int = 313
    use_pretrained: bool = False

    # 训练相关
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # 学习率调度
    lr_scheduler: str = "cosine"  # 'cosine' 或 'plateau'
    lr_decay_factor: float = 0.5
    lr_patience: int = 5

    # 损失函数
    loss_type: str = "simplified"  # 'rebalanced' 或 'simplified'
    use_perceptual_loss: bool = False
    perceptual_weight: float = 0.1

    # 优化器
    optimizer: str = "adam"  # 'adam' 或 'sgd'
    momentum: float = 0.9
    betas: tuple = (0.9, 0.999)

    # 训练技巧
    use_amp: bool = True  # 混合精度训练
    gradient_clip: float = 1.0

    # 检查点和日志
    checkpoint_dir: str = "../checkpoints"
    log_dir: str = "../outputs/logs"
    save_freq: int = 5  # 每 N 个 epoch 保存一次
    log_freq: int = 100  # 每 N 个 iteration 记录一次
    val_freq: int = 1000  # 每 N 个 iteration 验证一次

    # 设备
    device: str = "cuda"
    gpu_ids: list = None

    # 恢复训练
    resume: Optional[str] = None  # 检查点路径

    # 其他
    seed: int = 42
    deterministic: bool = False

    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = [0]

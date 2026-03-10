"""
训练入口脚本
"""

import os
import argparse
import random
import numpy as np
import torch

from data.dataset import COCOColorizationDataset, ImageNetColorizationDataset
from training.config import TrainingConfig
from training.trainer import Trainer


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if seed == 42:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="训练图像上色模型")

    # 数据相关
    parser.add_argument("--data_root", type=str, default="../datasets",
                        help="数据集根目录")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=['coco', 'imagenet'],
                        help="数据集类型")
    parser.add_argument("--image_size", type=int, default=256,
                        help="图像大小")
    parser.add_argument("--crop_size", type=int, default=224,
                        help="裁剪大小")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")

    # 训练相关
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Warmup 轮数")

    # 模型相关
    parser.add_argument("--num_classes", type=int, default=313,
                        help="颜色类别数")

    # 损失函数
    parser.add_argument("--loss_type", type=str, default="simplified",
                        choices=['simplified', 'rebalanced'],
                        help="损失函数类型")

    # 优化器
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=['adam', 'sgd'],
                        help="优化器")

    # 学习率调度
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=['cosine', 'plateau'],
                        help="学习率调度器")

    # 训练技巧
    parser.add_argument("--use_amp", action="store_true",
                        help="使用混合精度训练")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="梯度裁剪")

    # 检查点和日志
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints",
                        help="检查点保存目录")
    parser.add_argument("--log_dir", type=str, default="../outputs/logs",
                        help="日志目录")
    parser.add_argument("--save_freq", type=int, default=5,
                        help="保存频率（epoch）")
    parser.add_argument("--log_freq", type=int, default=100,
                        help="日志频率（iteration）")

    # 设备
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=[0],
                        help="GPU IDs")

    # 恢复训练
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")

    # 其他
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建配置
    config = TrainingConfig(
        data_root=args.data_root,
        dataset_type=args.dataset,
        image_size=args.image_size,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        lr_scheduler=args.lr_scheduler,
        loss_type=args.loss_type,
        optimizer=args.optimizer,
        use_amp=args.use_amp,
        gradient_clip=args.gradient_clip,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        device=args.device,
        gpu_ids=args.gpu_ids,
        resume=args.resume,
        seed=args.seed,
    )

    # 加载数据集
    print("加载数据集...")
    if args.dataset == 'coco':
        train_dataset = COCOColorizationDataset(
            coco_root=os.path.join(args.data_root, 'coco'),
            split='train2017',
            image_size=args.image_size,
            crop_size=args.crop_size,
        )
        val_dataset = COCOColorizationDataset(
            coco_root=os.path.join(args.data_root, 'coco'),
            split='val2017',
            image_size=args.image_size,
            crop_size=args.crop_size,
        )
    else:  # imagenet
        train_dataset = ImageNetColorizationDataset(
            imagenet_root=os.path.join(args.data_root, 'imagenet'),
            split='train',
            image_size=args.image_size,
            crop_size=args.crop_size,
        )
        val_dataset = ImageNetColorizationDataset(
            imagenet_root=os.path.join(args.data_root, 'imagenet'),
            split='val',
            image_size=args.image_size,
            crop_size=args.crop_size,
        )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 创建训练器
    trainer = Trainer(config, train_dataset, val_dataset)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()

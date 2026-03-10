"""
训练器类
封装完整的训练循环
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .config import TrainingConfig
from ..models.colorization_net import ColorizationNet
from ..models.losses import SimplifiedColorLoss, PerceptualLoss
from ..evaluation.metrics import calculate_psnr, calculate_ssim


class Trainer:
    """训练器"""

    def __init__(self, config: TrainingConfig, train_dataset, val_dataset=None):
        self.config = config

        # 设备
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )

        # 模型
        self.model = ColorizationNet(num_classes=config.num_classes).to(self.device)

        # 损失函数
        self.criterion = SimplifiedColorLoss().to(self.device)
        self.perceptual_loss = None
        if config.use_perceptual_loss:
            self.perceptual_loss = PerceptualLoss().to(self.device)

        # 优化器
        if config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                betas=config.betas,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )

        # 学习率调度器
        if config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.num_epochs
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.lr_decay_factor,
                patience=config.lr_patience,
            )

        # 混合精度
        self.scaler = GradScaler() if config.use_amp else None

        # 日志
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(config.log_dir)

        # 训练状态
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # 恢复训练
        if config.resume:
            self._load_checkpoint(config.resume)

    def train(self):
        """执行完整训练循环"""
        print(f"开始训练，设备：{self.device}")
        print(f"训练集大小：{len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"验证集大小：{len(self.val_loader.dataset)}")
        print(f"总 Epochs：{self.config.num_epochs}")
        print(f"批次大小：{self.config.batch_size}")
        print("-" * 50)

        for epoch in range(self.start_epoch, self.config.num_epochs):
            # 训练一个 epoch
            train_loss = self._train_epoch(epoch)

            # 验证
            val_loss = None
            if self.val_loader and (epoch + 1) % self.config.save_freq == 0:
                val_loss = self._validate(epoch)

            # 更新学习率
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss if val_loss is not None else train_loss)
            else:
                self.scheduler.step()

            # 记录日志
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/learning_rate', current_lr, epoch)
            if val_loss is not None:
                self.writer.add_scalar('Epoch/val_loss', val_loss, epoch)

            # 保存检查点
            if (epoch + 1) % self.config.save_freq == 0:
                self._save_checkpoint(epoch, val_loss)

            # 保存最佳模型
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)

            print(f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                  f"Train Loss: {train_loss:.4f} "
                  f"{'Val Loss: ' + f'{val_loss:.4f}' if val_loss else ''} "
                  f"LR: {current_lr:.6f}")

        self.writer.close()
        print("训练完成！")

    def _train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (l_channel, ab_channels) in enumerate(pbar):
            l_channel = l_channel.to(self.device)
            ab_channels = ab_channels.to(self.device)

            # Warmup 学习率
            if epoch < self.config.warmup_epochs:
                warmup_lr = self.config.learning_rate * (
                    self.global_step / (self.config.warmup_epochs * len(self.train_loader))
                )
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            self.optimizer.zero_grad()

            if self.config.use_amp:
                with autocast():
                    pred_ab = self.model(l_channel)
                    loss = self.criterion(pred_ab, ab_channels)

                    if self.perceptual_loss is not None:
                        loss += self.config.perceptual_weight * self.perceptual_loss(
                            pred_ab, ab_channels
                        )

                self.scaler.scale(loss).backward()
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_ab = self.model(l_channel)
                loss = self.criterion(pred_ab, ab_channels)

                if self.perceptual_loss is not None:
                    loss += self.config.perceptual_weight * self.perceptual_loss(
                        pred_ab, ab_channels
                    )

                loss.backward()
                if self.config.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # 记录日志
            if self.global_step % self.config.log_freq == 0:
                self.writer.add_scalar(
                    'Step/train_loss', loss.item(), self.global_step
                )

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    @torch.no_grad()
    def _validate(self, epoch):
        """在验证集上评估"""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = 0

        for l_channel, ab_channels in tqdm(self.val_loader, desc="Validating"):
            l_channel = l_channel.to(self.device)
            ab_channels = ab_channels.to(self.device)

            pred_ab = self.model(l_channel)
            loss = self.criterion(pred_ab, ab_channels)
            total_loss += loss.item()

            # 计算 PSNR
            psnr = calculate_psnr(pred_ab, ab_channels)
            total_psnr += psnr

            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches

        self.writer.add_scalar('Val/loss', avg_loss, epoch)
        self.writer.add_scalar('Val/psnr', avg_psnr, epoch)

        print(f"  验证 - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f} dB")
        return avg_loss

    def _save_checkpoint(self, epoch, val_loss=None, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss

        filename = f"checkpoint_epoch_{epoch+1}.pth"
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"  保存最佳模型 (val_loss: {val_loss:.4f})")

    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"从 {checkpoint_path} 恢复训练...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"恢复成功，从 Epoch {self.start_epoch} 继续训练")

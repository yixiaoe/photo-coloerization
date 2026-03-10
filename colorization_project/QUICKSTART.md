# 黑白照片上色项目快速开始指南

## 环境安装

```bash
cd colorization_project
pip install -r requirements.txt
```

## 数据准备

### 下载 COCO 数据集
```bash
cd colorization_project
python -m data.download_data --dataset coco --data_root ../datasets
```

### ImageNet 数据集
ImageNet 需要手动下载，请参考 `data/download_data.py` 中的说明。

## 训练模型

### 基础训练（COCO 数据集）
```bash
cd colorization_project
python train.py \
    --data_root ../datasets \
    --dataset coco \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 1e-4 \
    --use_amp
```

### 查看训练日志
```bash
tensorboard --logdir ../outputs/logs
```

## 测试模型

```bash
python test.py \
    --checkpoint ../checkpoints/best_model.pth \
    --data_root ../datasets \
    --dataset coco \
    --output_dir ../outputs/results
```

## 推理（对新图像上色）

```python
from inference.colorize import load_model, colorize_image
from PIL import Image

# 加载模型
model, device = load_model('../checkpoints/best_model.pth')

# 上色
rgb_output = colorize_image('path/to/grayscale_image.jpg', model, device)

# 保存结果
Image.fromarray(rgb_output).save('colorized_output.png')
```

## 项目结构

详细的项目架构和实现方案请参考 [README.md](README.md)

## 常见问题

1. **CUDA 内存不足**：减小 `batch_size`
2. **训练速度慢**：启用 `--use_amp` 混合精度训练
3. **上色效果偏灰**：增加训练轮数或调整损失权重

## 参考论文

- Colorful Image Colorization (ECCV 2016)
- 详见 `../paper/` 目录

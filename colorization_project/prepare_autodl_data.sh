#!/bin/bash
# 准备 AutoDL COCO 数据集

echo "=========================================="
echo "准备 AutoDL COCO 数据集"
echo "=========================================="

# 检查 AutoDL 环境
if [ ! -d ~/autodl-pub/COCO2017 ]; then
    echo "错误：未找到 AutoDL COCO2017 数据集"
    echo "当前不在 AutoDL 环境中"
    exit 1
fi

# 创建目标目录
mkdir -p ~/autodl-tmp/datasets/coco
cd ~/autodl-tmp/datasets/coco

# 检查是否已解压
if [ -d "train2017" ] && [ -d "val2017" ]; then
    echo "数据集已存在，跳过解压"
    train_count=$(ls train2017 | wc -l)
    val_count=$(ls val2017 | wc -l)
    echo "训练集: $train_count 张"
    echo "验证集: $val_count 张"
    exit 0
fi

# 解压训练集
if [ ! -d "train2017" ]; then
    echo "解压训练集（约 18GB，需要几分钟）..."
    unzip -q ~/autodl-pub/COCO2017/train2017.zip
    echo "✓ 训练集解压完成"
fi

# 解压验证集
if [ ! -d "val2017" ]; then
    echo "解压验证集（约 1GB）..."
    unzip -q ~/autodl-pub/COCO2017/val2017.zip
    echo "✓ 验证集解压完成"
fi

# 验证
train_count=$(ls train2017 | wc -l)
val_count=$(ls val2017 | wc -l)

echo ""
echo "=========================================="
echo "数据集准备完成！"
echo "=========================================="
echo "训练集: $train_count 张图像"
echo "验证集: $val_count 张图像"
echo "位置: ~/autodl-tmp/datasets/coco/"
echo ""

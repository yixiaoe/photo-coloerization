#!/bin/bash
# 一键训练脚本
# 在 GPU 服务器上运行此脚本开始训练

set -e

echo "=========================================="
echo "黑白照片上色 - 开始训练"
echo "=========================================="

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 检查是否在正确的目录
if [ ! -f "train.py" ]; then
    echo -e "${RED}错误：请在 colorization_project 目录下运行此脚本${NC}"
    exit 1
fi

# 检查数据集
if [ ! -d "../datasets/coco/train2017" ]; then
    echo -e "${RED}错误：未找到 COCO 数据集${NC}"
    echo "请先下载数据集："
    echo "  python -m data.download_data --dataset coco --data_root ../datasets"
    exit 1
fi

# 检查 GPU
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${RED}错误：CUDA 不可用${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 环境检查通过${NC}"
echo ""

# 训练配置
DATASET="coco"
BATCH_SIZE=32
NUM_EPOCHS=50
LEARNING_RATE=1e-4
USE_AMP="--use_amp"

# 显示配置
echo "训练配置："
echo "  数据集: $DATASET"
echo "  批次大小: $BATCH_SIZE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  学习率: $LEARNING_RATE"
echo "  混合精度: 启用"
echo ""

# 询问是否修改配置
read -p "是否使用默认配置? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    read -p "批次大小 (默认 32): " input_batch
    BATCH_SIZE=${input_batch:-32}

    read -p "训练轮数 (默认 50): " input_epochs
    NUM_EPOCHS=${input_epochs:-50}

    echo "更新后的配置："
    echo "  批次大小: $BATCH_SIZE"
    echo "  训练轮数: $NUM_EPOCHS"
fi

echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""

# 创建日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="train_${TIMESTAMP}.log"

# 运行训练
python train.py \
    --dataset $DATASET \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    $USE_AMP \
    --checkpoint_dir ../checkpoints \
    --log_dir ../outputs/logs \
    2>&1 | tee $LOG_FILE

echo ""
echo "=========================================="
echo -e "${GREEN}训练完成！${NC}"
echo "=========================================="
echo ""
echo "日志文件: $LOG_FILE"
echo "检查点目录: ../checkpoints/"
echo "TensorBoard 日志: ../outputs/logs/"
echo ""
echo "查看 TensorBoard:"
echo "  tensorboard --logdir ../outputs/logs --port 6006 --bind_all"
echo ""

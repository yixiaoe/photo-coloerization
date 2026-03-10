#!/bin/bash
# GPU 平台一键部署脚本
# 在 GPU 服务器上运行此脚本完成环境配置

set -e  # 遇到错误立即退出

echo "=========================================="
echo "黑白照片上色项目 - GPU 平台部署脚本"
echo "=========================================="

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查是否在正确的目录
if [ ! -f "train.py" ]; then
    echo -e "${RED}错误：请在 colorization_project 目录下运行此脚本${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 当前目录正确${NC}"

# 1. 检查 Python 版本
echo ""
echo "步骤 1/7: 检查 Python 版本..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python 版本: $PYTHON_VERSION"

# 2. 检查 CUDA 和 GPU
echo ""
echo "步骤 2/7: 检查 GPU 和 CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU 检测成功${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        echo "CUDA 版本: $CUDA_VERSION"
    else
        echo -e "${YELLOW}⚠ nvcc 未找到，但 nvidia-smi 可用${NC}"
    fi
else
    echo -e "${RED}✗ 未检测到 NVIDIA GPU！${NC}"
    echo "请确保在有 GPU 的服务器上运行此脚本"
    exit 1
fi

# 3. 创建必要的目录
echo ""
echo "步骤 3/7: 创建项目目录..."
cd ..
mkdir -p datasets/coco
mkdir -p checkpoints
mkdir -p outputs/logs
mkdir -p outputs/results
echo -e "${GREEN}✓ 目录创建完成${NC}"

# 4. 安装 PyTorch
echo ""
echo "步骤 4/7: 安装 PyTorch..."
echo "检查 PyTorch 是否已安装..."

if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    echo "PyTorch 版本: $TORCH_VERSION"
    echo "CUDA 可用: $CUDA_AVAILABLE"

    if [ "$CUDA_AVAILABLE" = "True" ]; then
        echo -e "${GREEN}✓ PyTorch 已安装且 CUDA 可用${NC}"
    else
        echo -e "${YELLOW}⚠ PyTorch 已安装但 CUDA 不可用，需要重新安装${NC}"
        read -p "是否重新安装 PyTorch? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --upgrade
        fi
    fi
else
    echo "PyTorch 未安装，开始安装..."
    echo "选择 CUDA 版本："
    echo "1) CUDA 11.8 (推荐)"
    echo "2) CUDA 12.1"
    read -p "请选择 (1/2): " cuda_choice

    if [ "$cuda_choice" = "2" ]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    fi

    echo -e "${GREEN}✓ PyTorch 安装完成${NC}"
fi

# 5. 安装其他依赖
echo ""
echo "步骤 5/7: 安装项目依赖..."
cd colorization_project

# 使用国内镜像加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo -e "${GREEN}✓ 依赖安装完成${NC}"

# 6. 验证安装
echo ""
echo "步骤 6/7: 验证环境..."
python -c "
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib
import skimage

print('✓ 所有依赖包导入成功')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 设备: {torch.cuda.get_device_name(0)}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 环境验证成功${NC}"
else
    echo -e "${RED}✗ 环境验证失败${NC}"
    exit 1
fi

# 7. 下载数据集（可选）
echo ""
echo "步骤 7/7: 数据集准备..."
echo "是否现在下载 COCO 数据集？（约 25GB，需要一些时间）"
read -p "下载数据集? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始下载 COCO 数据集..."
    python -m data.download_data --dataset coco --data_root ../datasets
    echo -e "${GREEN}✓ 数据集下载完成${NC}"
else
    echo -e "${YELLOW}⚠ 跳过数据集下载，请稍后手动下载${NC}"
    echo "手动下载命令："
    echo "  python -m data.download_data --dataset coco --data_root ../datasets"
fi

# 完成
echo ""
echo "=========================================="
echo -e "${GREEN}部署完成！${NC}"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 如果还没下载数据集，运行："
echo "   python -m data.download_data --dataset coco --data_root ../datasets"
echo ""
echo "2. 开始训练："
echo "   bash run_train.sh"
echo ""
echo "   或直接运行："
echo "   python train.py --dataset coco --batch_size 32 --num_epochs 50 --use_amp"
echo ""
echo "3. 使用 tmux 后台运行（推荐）："
echo "   tmux new -s training"
echo "   python train.py --dataset coco --batch_size 32 --num_epochs 50 --use_amp"
echo "   # 按 Ctrl+B 然后按 D 分离会话"
echo ""
echo "4. 监控训练："
echo "   tensorboard --logdir ../outputs/logs --port 6006 --bind_all"
echo ""

# AutoDL GPU 服务器部署步骤

## 当前环境信息
- GPU: NVIDIA GeForce RTX 3090 (1 卡)
- CPU: 15 核心
- 内存: 90 GB
- 系统盘: 30GB (/root)
- 数据盘: 50GB (/root/autodl-tmp) - 速度更快

## 部署步骤

### 1. 克隆 Git 仓库

```bash
# 进入系统盘（存放代码）
cd ~

# 克隆你的仓库（替换为你的仓库地址）
git clone <your-repo-url> colorization_project

# 如果是 GitHub
# git clone https://github.com/your-username/colorization.git colorization_project

# 如果是 Gitee
# git clone https://gitee.com/your-username/colorization.git colorization_project

# 进入项目目录
cd colorization_project
ls -la
```

### 2. 检查 GPU 和环境

```bash
# 检查 GPU
nvidia-smi

# 检查 Python
python --version

# 检查 CUDA
nvcc --version

# 检查 PyTorch（AutoDL 通常预装）
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. 运行一键部署脚本

```bash
# 给脚本添加执行权限
chmod +x setup_gpu.sh run_train.sh

# 运行部署脚本
bash setup_gpu.sh
```

**部署脚本会自动：**
- ✅ 检查 GPU 和 CUDA
- ✅ 创建必要目录
- ✅ 安装/验证 PyTorch
- ✅ 安装项目依赖
- ✅ 询问是否下载数据集

### 4. 下载 COCO 数据集

**方法 A：使用 AutoDL 数据集缓存（最快，推荐）**

```bash
# AutoDL 已经提供了 COCO 数据集
# 创建软链接到数据盘（速度更快）
cd ~
mkdir -p autodl-tmp/datasets
cd autodl-tmp/datasets

# 检查 AutoDL 是否有 COCO 缓存
ls ~/autodl-pub/ | grep -i coco

# 如果有 COCO2017，创建软链接
ln -s ~/autodl-pub/COCO2017 coco

# 验证
ls coco/
```

**方法 B：手动下载（如果缓存不可用）**

```bash
cd ~/autodl-tmp/datasets
mkdir -p coco
cd coco

# 下载训练集（约 18GB）
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# 下载验证集（约 1GB）
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# 清理压缩包
rm *.zip
```

**方法 C：使用项目脚本下载**

```bash
cd ~/colorization_project
python -m data.download_data --dataset coco --data_root ~/autodl-tmp/datasets
```

### 5. 验证数据集

```bash
# 检查训练集
ls ~/autodl-tmp/datasets/coco/train2017 | wc -l
# 应该显示 118287

# 检查验证集
ls ~/autodl-tmp/datasets/coco/val2017 | wc -l
# 应该显示 5000
```

### 6. 创建必要目录

```bash
cd ~
mkdir -p checkpoints
mkdir -p outputs/logs
mkdir -p outputs/results
```

### 7. 开始训练（使用 tmux 后台运行）

```bash
# 安装 tmux（如果没有）
apt-get update && apt-get install -y tmux

# 创建 tmux 会话
tmux new -s training

# 在 tmux 中进入项目目录
cd ~/colorization_project

# 开始训练
bash run_train.sh

# 或者直接运行 Python 命令
python train.py \
    --dataset coco \
    --data_root ~/autodl-tmp/datasets \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 1e-4 \
    --use_amp \
    --checkpoint_dir ~/checkpoints \
    --log_dir ~/outputs/logs

# 分离 tmux 会话：按 Ctrl+B 然后按 D
# 重新连接：tmux attach -t training
```

### 8. 监控训练

**查看训练日志：**
```bash
# 实时查看日志
tail -f ~/colorization_project/train_*.log

# 或查看最近的输出
tail -100 ~/colorization_project/train_*.log
```

**监控 GPU 使用：**
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或安装 gpustat
pip install gpustat
gpustat -i 1
```

**启动 TensorBoard：**
```bash
# 在新的 tmux 窗口或 SSH 会话中
tensorboard --logdir ~/outputs/logs --port 6006 --bind_all

# 在浏览器访问（AutoDL 会提供访问地址）
# http://<your-instance-url>:6006
```

### 9. 常用 tmux 命令

```bash
# 创建新会话
tmux new -s training

# 列出所有会话
tmux ls

# 连接到会话
tmux attach -t training

# 分离会话（在 tmux 内）
Ctrl+B, 然后按 D

# 杀死会话
tmux kill-session -t training

# 在 tmux 内创建新窗口
Ctrl+B, 然后按 C

# 切换窗口
Ctrl+B, 然后按 数字键
```

### 10. 检查训练进度

```bash
# 查看保存的检查点
ls -lh ~/checkpoints/

# 查看训练 epoch
grep "Epoch" ~/colorization_project/train_*.log | tail -10

# 查看损失
grep "Loss:" ~/colorization_project/train_*.log | tail -20
```

## 快速命令参考

```bash
# 一键部署（从头开始）
cd ~ && \
git clone <your-repo-url> colorization_project && \
cd colorization_project && \
chmod +x setup_gpu.sh run_train.sh && \
bash setup_gpu.sh

# 使用 AutoDL COCO 缓存
mkdir -p ~/autodl-tmp/datasets && \
ln -s ~/autodl-pub/COCO2017 ~/autodl-tmp/datasets/coco

# 开始训练（tmux 后台）
tmux new -s training
cd ~/colorization_project && bash run_train.sh
# Ctrl+B, D 分离

# 监控训练
tmux attach -t training  # 查看训练
nvidia-smi              # 查看 GPU
tail -f train_*.log     # 查看日志
```

## 故障排查

### 问题 1：Git 克隆失败
```bash
# 如果 GitHub 访问慢，使用 Gitee 镜像
# 或者使用 SSH 方式克隆
```

### 问题 2：CUDA out of memory
```bash
# 减小批次大小
python train.py --batch_size 16
```

### 问题 3：依赖安装失败
```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 4：数据集路径错误
```bash
# 确保数据集在正确位置
ls ~/autodl-tmp/datasets/coco/train2017
```

## 预计训练时间

- **RTX 3090 单卡**
- **批次大小 32**
- **50 epochs**
- **预计时间：60-72 小时（2.5-3 天）**
- **预计成本：2.5 元/小时 × 70 小时 = 175 元**

## 训练完成后

```bash
# 下载模型到本地（在 Mac 上执行）
scp root@<autodl-server>:~/checkpoints/best_model.pth ~/Desktop/CV/checkpoints/

# 下载日志
scp -r root@<autodl-server>:~/outputs/ ~/Desktop/CV/

# 关闭 AutoDL 实例（节省费用）
# 在 AutoDL 控制台点击"关机"
```

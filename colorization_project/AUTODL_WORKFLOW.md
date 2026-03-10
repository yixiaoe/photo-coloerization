# AutoDL GPU 训练工作流

## 快速开始（首次部署）

### 在 AutoDL 服务器上

```bash
# 1. 克隆仓库
cd ~
git clone https://github.com/yixiaoe/photo-coloerization.git
cd photo-coloerization/colorization_project

# 2. 准备数据集（解压 COCO2017，约 5-10 分钟）
chmod +x prepare_autodl_data.sh
bash prepare_autodl_data.sh

# 3. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 开始训练（使用 tmux 后台运行）
chmod +x run_train.sh
tmux new -s training
bash run_train.sh
# 按 Ctrl+B 然后按 D 分离会话
```

## 代码更新工作流

### 在本地 Mac 上修改代码

```bash
# 1. 进入项目目录
cd /Users/wangjunran/Desktop/CV/colorization_project

# 2. 修改代码（使用你喜欢的编辑器）

# 3. 提交到 Git
git add .
git commit -m "描述你的修改"
git push origin main
```

### 在 AutoDL 服务器上更新代码

```bash
# 1. 进入项目目录
cd ~/photo-coloerization/colorization_project

# 2. 拉取最新代码
git pull origin main

# 3. 如果需要重启训练
tmux attach -t training
# 按 Ctrl+C 停止当前训练
bash run_train.sh
```

## 目录结构

```
AutoDL 服务器:
~/
├── photo-coloerization/
│   └── colorization_project/     # 项目代码
├── autodl-tmp/
│   └── datasets/
│       └── coco/                  # 解压后的数据集（训练使用）
│           ├── train2017/         # 118287 张图像
│           └── val2017/           # 5000 张图像
├── checkpoints/                   # 模型检查点
└── outputs/
    ├── logs/                      # TensorBoard 日志
    └── results/                   # 上色结果

AutoDL 提供的数据:
~/autodl-pub/COCO2017/             # 原始 zip 文件（不要修改）
├── train2017.zip
├── val2017.zip
└── annotations_trainval2017.zip
```

## 常用命令

### 监控训练

```bash
# 重新连接 tmux 会话
tmux attach -t training

# 查看 GPU 使用
nvidia-smi

# 查看训练日志
tail -f ~/photo-coloerization/colorization_project/train_*.log

# 启动 TensorBoard
tensorboard --logdir ~/outputs/logs --port 6006 --bind_all
```

### Git 操作

```bash
# 查看状态
git status

# 拉取最新代码
git pull origin main

# 查看提交历史
git log --oneline -10
```

## 注意事项

1. **数据盘速度快**：数据集放在 `~/autodl-tmp/` 下，读写速度更快
2. **使用 tmux**：防止 SSH 断开导致训练中断
3. **定期保存**：检查点自动保存在 `~/checkpoints/`
4. **关机前备份**：重要的检查点和日志记得下载到本地

## 下载结果到本地

训练完成后，在 Mac 上执行：

```bash
# 下载模型
scp root@<autodl-server-ip>:~/checkpoints/best_model.pth ~/Desktop/CV/checkpoints/

# 下载日志
scp -r root@<autodl-server-ip>:~/outputs/ ~/Desktop/CV/
```

## 故障排查

### 问题：数据集未找到

```bash
# 检查数据集是否解压
ls ~/autodl-tmp/datasets/coco/

# 如果没有，运行准备脚本
cd ~/photo-coloerization/colorization_project
bash prepare_autodl_data.sh
```

### 问题：CUDA out of memory

```bash
# 修改批次大小
# 编辑 run_train.sh，将 BATCH_SIZE 改为 16
```

### 问题：训练中断

```bash
# 使用 tmux 后台运行
tmux new -s training
bash run_train.sh
# Ctrl+B, D 分离
```

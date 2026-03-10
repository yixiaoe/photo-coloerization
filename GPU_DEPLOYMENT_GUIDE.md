# GPU 平台部署完整指南

## 📋 概述

本指南帮助你将黑白照片上色项目部署到 GPU 平台进行训练。

**重要提示：**
- ✅ 代码已在 Mac 本地准备完成
- ⚠️ Mac 无 NVIDIA GPU，无法直接训练
- 🎯 需要在租用的 GPU 平台上训练

---

## 🚀 快速开始（3 步）

### 步骤 1: 在 Mac 上打包代码

```bash
cd /Users/wangjunran/Desktop/CV
bash pack_project.sh
```

这将生成 `colorization_project_YYYYMMDD_HHMMSS.tar.gz` 文件。

### 步骤 2: 上传到 GPU 平台

**方式 A：使用 SCP（推荐）**
```bash
scp colorization_project_*.tar.gz user@gpu-server:/root/
```

**方式 B：使用 Web 界面**
- 登录 GPU 平台（如 AutoDL）
- 打开 JupyterLab 文件管理器
- 拖拽上传压缩包

**方式 C：使用 Git（最推荐）**
```bash
cd colorization_project
git init
git add .
git commit -m "Initial commit"
git push to your repository

# 在 GPU 服务器上
git clone <your-repo-url>
```

### 步骤 3: 在 GPU 平台运行

```bash
# SSH 登录 GPU 服务器
ssh user@gpu-server

# 解压
tar -xzf colorization_project_*.tar.gz
cd colorization_project

# 一键部署
bash setup_gpu.sh

# 开始训练
bash run_train.sh
```

---

## 📦 详细部署步骤

### 1. Mac 本地准备

#### 1.1 打包项目
```bash
cd /Users/wangjunran/Desktop/CV
bash pack_project.sh
```

**打包内容：**
- ✅ colorization_project/ 所有代码
- ❌ datasets/ (在 GPU 平台下载)
- ❌ checkpoints/ (训练时生成)
- ❌ __pycache__/ (自动排除)

#### 1.2 验证打包
```bash
# 查看文件大小
ls -lh colorization_project_*.tar.gz

# 应该约 50-100 KB（只包含代码）
```

### 2. GPU 平台选择

#### 推荐平台：AutoDL

**优点：**
- 💰 价格便宜：RTX 3090 约 2.5 元/小时
- 🚀 速度快：国内访问快
- 📦 数据集缓存：COCO 可直接挂载
- 🛠️ 预装环境：PyTorch 已安装

**注册和使用：**
1. 访问 https://www.autodl.com
2. 注册账号并充值（建议 200 元）
3. 选择实例：
   - GPU: RTX 3090 或 4090
   - 镜像: PyTorch 2.0 + CUDA 11.8
   - 磁盘: 50GB 以上
4. 启动实例

#### 其他平台

**恒源云** (https://www.gpushare.com)
- 价格: 2-4 元/小时
- 适合长期训练

**Google Colab**
- 免费 T4 GPU
- 适合测试和学习
- 有时间限制

### 3. 上传代码

#### 方法 A：SCP 上传（推荐）

```bash
# 在 Mac 上执行
scp colorization_project_*.tar.gz root@<gpu-server-ip>:/root/

# 如果有密钥
scp -i ~/.ssh/id_rsa colorization_project_*.tar.gz root@<gpu-server-ip>:/root/
```

#### 方法 B：JupyterLab 上传

1. 在 AutoDL 控制台点击"JupyterLab"
2. 打开文件管理器
3. 拖拽 `colorization_project_*.tar.gz` 到 `/root/`
4. 等待上传完成

#### 方法 C：Git 上传（最推荐）

```bash
# 在 Mac 上
cd colorization_project
git init
git add .
git commit -m "Initial commit"

# 推送到 GitHub/Gitee
git remote add origin https://github.com/your-username/colorization.git
git push -u origin main

# 在 GPU 服务器上
cd /root
git clone https://github.com/your-username/colorization.git colorization_project
```

### 4. GPU 平台部署

#### 4.1 SSH 登录

```bash
# AutoDL 提供的 SSH 命令
ssh -p <port> root@<server-ip>

# 输入密码（在 AutoDL 控制台查看）
```

#### 4.2 解压项目

```bash
cd /root
tar -xzf colorization_project_*.tar.gz
cd colorization_project
ls -la  # 验证文件
```

#### 4.3 运行部署脚本

```bash
bash setup_gpu.sh
```

**脚本会自动：**
1. ✅ 检查 Python 和 CUDA
2. ✅ 检查 GPU 状态
3. ✅ 创建必要目录
4. ✅ 安装 PyTorch（如需要）
5. ✅ 安装项目依赖
6. ✅ 验证环境
7. ⚠️ 询问是否下载数据集

**预期输出：**
```
==========================================
黑白照片上色项目 - GPU 平台部署脚本
==========================================
✓ 当前目录正确
✓ NVIDIA GPU 检测成功
✓ PyTorch 已安装且 CUDA 可用
✓ 依赖安装完成
✓ 环境验证成功
==========================================
部署完成！
==========================================
```

### 5. 下载数据集

#### 方法 A：使用脚本下载

```bash
cd colorization_project
python -m data.download_data --dataset coco --data_root ../datasets
```

#### 方法 B：手动下载

```bash
cd ../datasets
mkdir -p coco
cd coco

# 下载训练集（约 18GB）
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# 下载验证集（约 1GB）
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# 清理
rm *.zip
```

#### 方法 C：使用 AutoDL 数据集缓存（最快）

```bash
# AutoDL 提供 COCO 数据集缓存
ln -s /root/autodl-tmp/coco /root/datasets/coco
```

#### 验证数据集

```bash
ls /root/datasets/coco/train2017 | wc -l  # 应该是 118287
ls /root/datasets/coco/val2017 | wc -l    # 应该是 5000
```

### 6. 开始训练

#### 方法 A：使用训练脚本（推荐）

```bash
cd /root/colorization_project
bash run_train.sh
```

**脚本会：**
1. 检查环境和数据集
2. 显示训练配置
3. 询问是否修改参数
4. 开始训练并保存日志

#### 方法 B：直接运行

```bash
python train.py \
    --dataset coco \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 1e-4 \
    --use_amp \
    --checkpoint_dir ../checkpoints \
    --log_dir ../outputs/logs
```

#### 方法 C：使用 tmux 后台运行（推荐）

```bash
# 创建 tmux 会话
tmux new -s training

# 在 tmux 中运行训练
bash run_train.sh

# 分离会话：按 Ctrl+B 然后按 D
# 重新连接：tmux attach -t training
# 查看所有会话：tmux ls
```

#### 方法 D：使用 Jupyter Notebook

1. 在 JupyterLab 中打开 `train_gpu.ipynb`
2. 按顺序运行所有单元格
3. 监控训练进度

### 7. 监控训练

#### 7.1 查看训练日志

```bash
# 实时查看
tail -f train_*.log

# 查看最近的损失
grep "Loss:" train_*.log | tail -20
```

#### 7.2 启动 TensorBoard

```bash
# 在 GPU 服务器上
tensorboard --logdir /root/outputs/logs --port 6006 --bind_all

# 在浏览器访问
http://<gpu-server-ip>:6006
```

#### 7.3 监控 GPU 使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
gpustat -i 1
```

#### 7.4 检查训练进度

```bash
# 查看保存的检查点
ls -lh /root/checkpoints/

# 查看 epoch 进度
grep "Epoch" train_*.log | tail -5
```

### 8. 训练完成后

#### 8.1 下载模型

```bash
# 在 Mac 上执行
scp root@<gpu-server-ip>:/root/checkpoints/best_model.pth ~/Desktop/CV/checkpoints/

# 或使用 rsync
rsync -avz root@<gpu-server-ip>:/root/checkpoints/ ~/Desktop/CV/checkpoints/
```

#### 8.2 下载日志

```bash
scp -r root@<gpu-server-ip>:/root/outputs/logs/ ~/Desktop/CV/outputs/
```

#### 8.3 下载结果

```bash
scp -r root@<gpu-server-ip>:/root/outputs/results/ ~/Desktop/CV/outputs/
```

---

## 🔧 常见问题

### Q1: CUDA out of memory

**症状：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
```bash
# 减小批次大小
python train.py --batch_size 16

# 或减小图像大小
python train.py --image_size 128 --crop_size 112
```

### Q2: SSH 连接断开

**症状：**
训练中断，进程被杀死

**解决方案：**
```bash
# 使用 tmux
tmux new -s training
python train.py ...
# Ctrl+B, D 分离

# 或使用 nohup
nohup python train.py ... > train.log 2>&1 &
```

### Q3: 数据下载慢

**解决方案：**
```bash
# 使用 AutoDL 数据集缓存
ln -s /root/autodl-tmp/coco /root/datasets/coco

# 或使用国内镜像
wget -c http://mirrors.aliyun.com/coco/...
```

### Q4: 依赖安装失败

**解决方案：**
```bash
# 使用国内 pip 源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用 conda
conda install pytorch torchvision -c pytorch
```

### Q5: 训练速度慢

**检查：**
```bash
# 确认使用 GPU
python -c "import torch; print(torch.cuda.is_available())"

# 确认混合精度启用
grep "use_amp" train_*.log
```

**优化：**
```bash
# 启用混合精度
python train.py --use_amp

# 增加数据加载线程
python train.py --num_workers 8
```

---

## 💰 成本估算

### AutoDL RTX 3090

**训练 50 epochs（约 60 小时）：**
- 价格：2.5 元/小时
- 总成本：2.5 × 60 = **150 元**

**建议充值：200 元**（留有余量）

### 节省成本技巧

1. **使用月卡**：长期训练更划算
2. **选择合适时段**：避开高峰期
3. **及时关机**：训练完成立即关闭实例
4. **使用数据集缓存**：节省下载时间

---

## 📝 完整工作流程

### Mac 本地（第 1 天）

```bash
# 1. 打包代码
cd /Users/wangjunran/Desktop/CV
bash pack_project.sh

# 2. 上传到 GPU 平台（选择一种方式）
scp colorization_project_*.tar.gz root@<gpu-server>:/root/
```

### GPU 平台（第 1-2 天）

```bash
# 3. 解压和部署
cd /root
tar -xzf colorization_project_*.tar.gz
cd colorization_project
bash setup_gpu.sh

# 4. 下载数据集
python -m data.download_data --dataset coco --data_root ../datasets

# 5. 开始训练（使用 tmux）
tmux new -s training
bash run_train.sh
# Ctrl+B, D 分离会话
```

### 训练期间（第 2-10 天）

```bash
# 每天检查进度
tmux attach -t training  # 查看训练
nvidia-smi  # 查看 GPU 使用
tail -f train_*.log  # 查看日志
```

### 训练完成（第 11-12 天）

```bash
# 在 Mac 上下载结果
scp root@<gpu-server>:/root/checkpoints/best_model.pth ~/Desktop/CV/checkpoints/
scp -r root@<gpu-server>:/root/outputs/ ~/Desktop/CV/
```

---

## ✅ 检查清单

### 部署前
- [ ] 代码已打包
- [ ] GPU 平台已注册
- [ ] 账户已充值（建议 200 元）

### 部署时
- [ ] 代码已上传
- [ ] setup_gpu.sh 运行成功
- [ ] GPU 可用（nvidia-smi）
- [ ] PyTorch CUDA 可用
- [ ] 数据集已下载

### 训练时
- [ ] 使用 tmux 或 nohup
- [ ] 混合精度已启用
- [ ] TensorBoard 可访问
- [ ] 定期检查进度

### 训练后
- [ ] 模型已下载
- [ ] 日志已下载
- [ ] GPU 实例已关闭

---

## 📞 获取帮助

如果遇到问题：

1. 查看本文档的"常见问题"部分
2. 检查训练日志：`tail -f train_*.log`
3. 查看 GPU 状态：`nvidia-smi`
4. 验证环境：`python ../verify_code.py`

---

**祝训练顺利！** 🎨

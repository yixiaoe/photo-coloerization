# AutoDL 快速部署指南

## 🚀 你现在需要做的事情

### 第 1 步：提交代码到 GitHub

在你的 Mac 上执行：

```bash
cd /Users/wangjunran/Desktop/CV/colorization_project

# 添加所有新文件
git add .

# 提交
git commit -m "添加 AutoDL 支持：数据集准备脚本和工作流文档"

# 推送到 GitHub
git push origin main
```

### 第 2 步：在 AutoDL 服务器上更新代码

在 AutoDL 终端执行：

```bash
# 进入项目目录
cd ~/photo-coloerization/colorization_project

# 拉取最新代码
git pull origin main

# 查看新文件
ls -la
```

你应该看到这些新文件：
- ✅ `prepare_autodl_data.sh` - 数据集准备脚本
- ✅ `AUTODL_WORKFLOW.md` - 工作流文档
- ✅ `.gitignore` - Git 忽略文件
- ✅ `run_train.sh` - 已更新，支持 AutoDL 路径

### 第 3 步：准备数据集

在 AutoDL 终端执行：

```bash
# 给脚本添加执行权限
chmod +x prepare_autodl_data.sh

# 运行数据集准备脚本（约 5-10 分钟）
bash prepare_autodl_data.sh
```

**预期输出：**
```
==========================================
准备 AutoDL COCO 数据集
==========================================
解压训练集（约 18GB，需要几分钟）...
✓ 训练集解压完成
解压验证集（约 1GB）...
✓ 验证集解压完成

==========================================
数据集准备完成！
==========================================
训练集: 118287 张图像
验证集: 5000 张图像
位置: ~/autodl-tmp/datasets/coco/
```

### 第 4 步：开始训练

在 AutoDL 终端执行：

```bash
# 使用 tmux 后台运行（推荐）
tmux new -s training

# 在 tmux 中运行训练
bash run_train.sh

# 选择 y 使用默认配置

# 分离 tmux 会话：按 Ctrl+B 然后按 D
```

### 第 5 步：监控训练

```bash
# 重新连接 tmux 查看训练
tmux attach -t training

# 查看 GPU 使用
nvidia-smi

# 查看训练日志
tail -f train_*.log
```

## 📝 完整命令清单（复制粘贴）

### 在 Mac 上（提交代码）

```bash
cd /Users/wangjunran/Desktop/CV/colorization_project
git add .
git commit -m "添加 AutoDL 支持"
git push origin main
```

### 在 AutoDL 上（部署和训练）

```bash
# 更新代码
cd ~/photo-coloerization/colorization_project
git pull origin main

# 准备数据集
chmod +x prepare_autodl_data.sh
bash prepare_autodl_data.sh

# 开始训练
tmux new -s training
bash run_train.sh
# Ctrl+B, D 分离
```

## ✅ 验证清单

- [ ] Mac 上代码已推送到 GitHub
- [ ] AutoDL 上代码已更新
- [ ] 数据集已解压（118287 + 5000 张图像）
- [ ] 训练已开始，GPU 利用率正常
- [ ] tmux 会话正常运行

## 🎯 预期结果

- **数据集准备**：5-10 分钟
- **训练开始**：立即
- **完整训练**：2-3 天（50 epochs）
- **预计成本**：约 175 元

## 📞 遇到问题？

查看 `AUTODL_WORKFLOW.md` 中的故障排查部分。

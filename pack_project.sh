#!/bin/bash
# Mac 本地打包脚本
# 在 Mac 上运行此脚本打包项目代码

echo "=========================================="
echo "打包项目代码"
echo "=========================================="

cd /Users/wangjunran/Desktop/CV

# 检查目录
if [ ! -d "colorization_project" ]; then
    echo "错误：未找到 colorization_project 目录"
    exit 1
fi

# 创建打包目录
PACKAGE_NAME="colorization_project_$(date +%Y%m%d_%H%M%S)"
echo "打包名称: $PACKAGE_NAME"

# 方法选择
echo ""
echo "选择打包方式："
echo "1) tar.gz (推荐，Linux 服务器)"
echo "2) zip (Windows 兼容)"
read -p "请选择 (1/2): " choice

if [ "$choice" = "2" ]; then
    # 使用 zip
    echo "使用 zip 打包..."
    zip -r "${PACKAGE_NAME}.zip" colorization_project/ \
        -x "colorization_project/__pycache__/*" \
        -x "colorization_project/*/__pycache__/*" \
        -x "colorization_project/*/*/__pycache__/*" \
        -x "*.pyc" \
        -x ".DS_Store"

    PACKAGE_FILE="${PACKAGE_NAME}.zip"
else
    # 使用 tar.gz (默认)
    echo "使用 tar.gz 打包..."
    tar -czf "${PACKAGE_NAME}.tar.gz" \
        --exclude="__pycache__" \
        --exclude="*.pyc" \
        --exclude=".DS_Store" \
        colorization_project/

    PACKAGE_FILE="${PACKAGE_NAME}.tar.gz"
fi

# 显示文件信息
echo ""
echo "=========================================="
echo "打包完成！"
echo "=========================================="
echo "文件: $PACKAGE_FILE"
ls -lh "$PACKAGE_FILE"

# 计算 MD5
if command -v md5 &> /dev/null; then
    echo "MD5: $(md5 -q $PACKAGE_FILE)"
elif command -v md5sum &> /dev/null; then
    echo "MD5: $(md5sum $PACKAGE_FILE | awk '{print $1}')"
fi

echo ""
echo "上传方式："
echo ""
echo "1. 使用 SCP 上传:"
echo "   scp $PACKAGE_FILE user@gpu-server:/root/"
echo ""
echo "2. 使用 rsync 上传 (支持断点续传):"
echo "   rsync -avz --progress $PACKAGE_FILE user@gpu-server:/root/"
echo ""
echo "3. 使用 Web 界面上传:"
echo "   - 打开 GPU 平台的 JupyterLab"
echo "   - 拖拽 $PACKAGE_FILE 到文件管理器"
echo ""
echo "4. 使用 Git (推荐):"
echo "   cd colorization_project"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'Initial commit'"
echo "   git remote add origin <your-repo-url>"
echo "   git push -u origin main"
echo ""
echo "在 GPU 服务器上解压:"
if [ "$choice" = "2" ]; then
    echo "   unzip $PACKAGE_FILE"
else
    echo "   tar -xzf $PACKAGE_FILE"
fi
echo "   cd colorization_project"
echo "   bash setup_gpu.sh"
echo ""

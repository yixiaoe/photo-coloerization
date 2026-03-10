"""
使用示例脚本
演示如何使用训练好的模型进行图像上色
"""

import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.colorize import load_model, colorize_image, colorize_batch


def example_single_image():
    """单张图像上色示例"""
    print("=" * 60)
    print("示例 1: 单张图像上色")
    print("=" * 60)

    # 模型路径
    checkpoint_path = "../checkpoints/best_model.pth"

    # 检查模型是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误：模型文件不存在 {checkpoint_path}")
        print("请先训练模型或下载预训练模型")
        return

    # 加载模型
    print("加载模型...")
    model, device = load_model(checkpoint_path)
    print(f"模型加载成功，设备：{device}")

    # 输入图像路径
    image_path = "../colorization-master/imgs/ansel_adams.jpg"

    if not os.path.exists(image_path):
        print(f"错误：输入图像不存在 {image_path}")
        return

    # 上色
    print(f"处理图像：{image_path}")
    rgb_output = colorize_image(image_path, model, device, target_size=256)

    # 保存结果
    output_dir = "../outputs/examples"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "colorized_single.png")
    Image.fromarray(rgb_output).save(output_path)
    print(f"结果已保存到：{output_path}")

    # 显示结果
    plt.figure(figsize=(12, 5))

    # 原图
    original = Image.open(image_path).convert('L')
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('灰度输入')
    plt.axis('off')

    # 上色结果
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_output)
    plt.title('上色结果')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150)
    print(f"对比图已保存")
    plt.show()


def example_batch_images():
    """批量图像上色示例"""
    print("\n" + "=" * 60)
    print("示例 2: 批量图像上色")
    print("=" * 60)

    checkpoint_path = "../checkpoints/best_model.pth"

    if not os.path.exists(checkpoint_path):
        print(f"错误：模型文件不存在 {checkpoint_path}")
        return

    # 加载模型
    print("加载模型...")
    model, device = load_model(checkpoint_path)

    # 批量图像路径
    image_dir = "../colorization-master/imgs"
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ][:5]  # 只处理前 5 张

    if not image_paths:
        print(f"错误：在 {image_dir} 中没有找到图像")
        return

    print(f"找到 {len(image_paths)} 张图像")

    # 批量上色
    output_dir = "../outputs/examples/batch"
    results = colorize_batch(image_paths, model, device, output_dir=output_dir)

    print(f"\n批量处理完成！结果保存在：{output_dir}")


def example_custom_parameters():
    """自定义参数示例"""
    print("\n" + "=" * 60)
    print("示例 3: 自定义参数")
    print("=" * 60)

    checkpoint_path = "../checkpoints/best_model.pth"

    if not os.path.exists(checkpoint_path):
        print(f"错误：模型文件不存在")
        return

    # 加载模型
    model, device = load_model(checkpoint_path, device='cuda', num_classes=313)

    image_path = "../colorization-master/imgs/ansel_adams.jpg"

    if not os.path.exists(image_path):
        print(f"错误：输入图像不存在")
        return

    # 使用不同的处理大小
    for size in [128, 256, 512]:
        print(f"\n处理大小：{size}x{size}")
        rgb_output = colorize_image(image_path, model, device, target_size=size)

        output_dir = "../outputs/examples/sizes"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"colorized_{size}.png")
        Image.fromarray(rgb_output).save(output_path)
        print(f"  保存到：{output_path}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("黑白照片上色 - 使用示例")
    print("=" * 60 + "\n")

    # 检查 PyTorch 和 CUDA
    print(f"PyTorch 版本：{torch.__version__}")
    print(f"CUDA 可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本：{torch.version.cuda}")
        print(f"GPU 设备：{torch.cuda.get_device_name(0)}")
    print()

    # 运行示例
    try:
        example_single_image()
    except Exception as e:
        print(f"示例 1 出错：{e}")

    try:
        example_batch_images()
    except Exception as e:
        print(f"示例 2 出错：{e}")

    try:
        example_custom_parameters()
    except Exception as e:
        print(f"示例 3 出错：{e}")

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

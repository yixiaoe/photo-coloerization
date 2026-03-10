"""
代码验证脚本
测试各个模块是否可以正常导入和运行
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有模块导入"""
    print("=" * 60)
    print("测试模块导入")
    print("=" * 60)

    try:
        print("✓ 导入 data 模块...")
        from colorization_project.data import dataset, preprocess, download_data
        print("  - dataset.py")
        print("  - preprocess.py")
        print("  - download_data.py")
    except Exception as e:
        print(f"✗ data 模块导入失败: {e}")
        return False

    try:
        print("\n✓ 导入 models 模块...")
        from colorization_project.models import colorization_net, losses, utils
        print("  - colorization_net.py")
        print("  - losses.py")
        print("  - utils.py")
    except Exception as e:
        print(f"✗ models 模块导入失败: {e}")
        return False

    try:
        print("\n✓ 导入 training 模块...")
        from colorization_project.training import config, trainer
        print("  - config.py")
        print("  - trainer.py")
    except Exception as e:
        print(f"✗ training 模块导入失败: {e}")
        return False

    try:
        print("\n✓ 导入 evaluation 模块...")
        from colorization_project.evaluation import metrics, visualize
        print("  - metrics.py")
        print("  - visualize.py")
    except Exception as e:
        print(f"✗ evaluation 模块导入失败: {e}")
        return False

    try:
        print("\n✓ 导入 inference 模块...")
        from colorization_project.inference import colorize
        print("  - colorize.py")
    except Exception as e:
        print(f"✗ inference 模块导入失败: {e}")
        return False

    print("\n" + "=" * 60)
    print("所有模块导入成功！")
    print("=" * 60)
    return True


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("测试模型创建")
    print("=" * 60)

    try:
        import torch
        from colorization_project.models.colorization_net import ColorizationNet

        print("创建模型...")
        model = ColorizationNet(num_classes=313)
        print(f"✓ 模型创建成功")

        # 测试前向传播
        print("\n测试前向传播...")
        dummy_input = torch.randn(2, 1, 224, 224)  # (B, C, H, W)
        output = model(dummy_input)
        print(f"✓ 输入形状: {dummy_input.shape}")
        print(f"✓ 输出形状: {output.shape}")

        # 计算参数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 模型参数量: {num_params:,}")

        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_preprocessing():
    """测试数据预处理"""
    print("\n" + "=" * 60)
    print("测试数据预处理")
    print("=" * 60)

    try:
        import numpy as np
        from colorization_project.data.preprocess import (
            rgb_to_lab, lab_to_rgb, normalize_lab, denormalize_lab
        )

        print("创建测试图像...")
        rgb_image = np.random.rand(224, 224, 3)

        print("✓ RGB -> LAB 转换...")
        lab_image = rgb_to_lab(rgb_image)
        print(f"  LAB 形状: {lab_image.shape}")
        print(f"  L 范围: [{lab_image[:,:,0].min():.2f}, {lab_image[:,:,0].max():.2f}]")
        print(f"  ab 范围: [{lab_image[:,:,1:].min():.2f}, {lab_image[:,:,1:].max():.2f}]")

        print("\n✓ LAB 归一化...")
        lab_normalized = normalize_lab(lab_image)
        print(f"  归一化后 L 范围: [{lab_normalized[:,:,0].min():.2f}, {lab_normalized[:,:,0].max():.2f}]")

        print("\n✓ LAB 反归一化...")
        lab_denorm = denormalize_lab(lab_normalized)

        print("\n✓ LAB -> RGB 转换...")
        rgb_reconstructed = lab_to_rgb(lab_denorm)
        print(f"  重建 RGB 形状: {rgb_reconstructed.shape}")

        return True
    except Exception as e:
        print(f"✗ 数据预处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """测试损失函数"""
    print("\n" + "=" * 60)
    print("测试损失函数")
    print("=" * 60)

    try:
        import torch
        from colorization_project.models.losses import SimplifiedColorLoss

        print("创建损失函数...")
        criterion = SimplifiedColorLoss()

        print("\n测试损失计算...")
        pred_ab = torch.randn(2, 2, 224, 224)
        target_ab = torch.randn(2, 2, 224, 224)
        loss = criterion(pred_ab, target_ab)
        print(f"✓ 损失值: {loss.item():.4f}")

        return True
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """测试配置类"""
    print("\n" + "=" * 60)
    print("测试配置类")
    print("=" * 60)

    try:
        from colorization_project.training.config import TrainingConfig

        print("创建配置...")
        config = TrainingConfig(
            batch_size=16,
            num_epochs=10,
            learning_rate=1e-4
        )
        print(f"✓ 批次大小: {config.batch_size}")
        print(f"✓ 训练轮数: {config.num_epochs}")
        print(f"✓ 学习率: {config.learning_rate}")
        print(f"✓ 设备: {config.device}")

        return True
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """检查依赖包"""
    print("\n" + "=" * 60)
    print("检查依赖包")
    print("=" * 60)

    dependencies = [
        'torch',
        'torchvision',
        'numpy',
        'PIL',
        'skimage',
        'matplotlib',
        'scipy',
        'tqdm'
    ]

    all_installed = True
    for dep in dependencies:
        try:
            if dep == 'PIL':
                __import__('PIL')
            elif dep == 'skimage':
                __import__('skimage')
            else:
                __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} - 未安装")
            all_installed = False

    if not all_installed:
        print("\n请运行以下命令安装缺失的依赖：")
        print("pip install -r colorization_project/requirements.txt")

    return all_installed


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("黑白照片上色项目 - 代码验证")
    print("=" * 60)

    results = []

    # 检查依赖
    results.append(("依赖检查", check_dependencies()))

    # 测试导入
    results.append(("模块导入", test_imports()))

    # 测试模型
    results.append(("模型创建", test_model_creation()))

    # 测试数据预处理
    results.append(("数据预处理", test_data_preprocessing()))

    # 测试损失函数
    results.append(("损失函数", test_loss_functions()))

    # 测试配置
    results.append(("配置类", test_config()))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！项目代码可以正常运行。")
        print("\n下一步：")
        print("1. 下载数据集: python -m colorization_project.data.download_data")
        print("2. 开始训练: python colorization_project/train.py")
    else:
        print("部分测试失败，请检查错误信息并修复。")
    print("=" * 60)


if __name__ == "__main__":
    main()

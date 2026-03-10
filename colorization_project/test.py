"""
测试脚本
用于评估训练好的模型
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from colorization_project.data.dataset import COCOColorizationDataset, ImageNetColorizationDataset
from colorization_project.models.colorization_net import ColorizationNet
from colorization_project.evaluation.metrics import calculate_psnr, calculate_ssim, calculate_l2_distance
from colorization_project.evaluation.visualize import visualize_batch


def main():
    parser = argparse.ArgumentParser(description="测试图像上色模型")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--data_root", type=str, default="../datasets",
                        help="数据集根目录")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=['coco', 'imagenet'],
                        help="数据集类型")
    parser.add_argument("--split", type=str, default="val2017",
                        help="数据集分割（val2017 或 val）")
    parser.add_argument("--image_size", type=int, default=256,
                        help="图像大小")
    parser.add_argument("--crop_size", type=int, default=224,
                        help="裁剪大小")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    parser.add_argument("--output_dir", type=str, default="../outputs/results",
                        help="结果保存目录")
    parser.add_argument("--num_visualize", type=int, default=10,
                        help="可视化样本数量")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    model = ColorizationNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载数据集
    print("加载数据集...")
    if args.dataset == 'coco':
        test_dataset = COCOColorizationDataset(
            coco_root=os.path.join(args.data_root, 'coco'),
            split=args.split,
            image_size=args.image_size,
            crop_size=args.crop_size,
        )
    else:
        test_dataset = ImageNetColorizationDataset(
            imagenet_root=os.path.join(args.data_root, 'imagenet'),
            split='val',
            image_size=args.image_size,
            crop_size=args.crop_size,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"测试集大小: {len(test_dataset)}")

    # 评估
    print("开始评估...")
    total_psnr = 0.0
    total_ssim = 0.0
    total_l2 = 0.0
    num_batches = 0

    vis_count = 0
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (l_channel, ab_channels) in enumerate(tqdm(test_loader)):
            l_channel = l_channel.to(device)
            ab_channels = ab_channels.to(device)

            # 推理
            pred_ab = model(l_channel)

            # 计算指标
            psnr = calculate_psnr(pred_ab, ab_channels)
            ssim = calculate_ssim(pred_ab, ab_channels)
            l2 = calculate_l2_distance(pred_ab, ab_channels)

            total_psnr += psnr
            total_ssim += ssim
            total_l2 += l2
            num_batches += 1

            # 可视化前几个 batch
            if vis_count < args.num_visualize:
                save_dir = os.path.join(args.output_dir, f"batch_{batch_idx}")
                visualize_batch(
                    l_channel, pred_ab, ab_channels,
                    save_dir=save_dir,
                    num_samples=min(4, args.batch_size)
                )
                vis_count += 1

    # 打印结果
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_l2 = total_l2 / num_batches

    print("\n" + "=" * 50)
    print("评估结果:")
    print(f"  平均 PSNR: {avg_psnr:.2f} dB")
    print(f"  平均 SSIM: {avg_ssim:.4f}")
    print(f"  平均 L2 距离: {avg_l2:.2f}")
    print("=" * 50)

    # 保存结果
    result_file = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(result_file, 'w') as f:
        f.write(f"模型: {args.checkpoint}\n")
        f.write(f"数据集: {args.dataset} ({args.split})\n")
        f.write(f"测试样本数: {len(test_dataset)}\n")
        f.write(f"\n评估指标:\n")
        f.write(f"  PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"  SSIM: {avg_ssim:.4f}\n")
        f.write(f"  L2 距离: {avg_l2:.2f}\n")

    print(f"\n结果已保存到: {result_file}")


if __name__ == "__main__":
    main()

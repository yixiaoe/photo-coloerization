"""
数据下载脚本
用于下载 COCO 和 ImageNet 数据集
"""

import os
import argparse
import subprocess


def download_coco(data_root):
    """
    下载 COCO 2017 数据集

    Args:
        data_root: 数据集保存根目录
    """
    coco_dir = os.path.join(data_root, 'coco')
    os.makedirs(coco_dir, exist_ok=True)

    print("开始下载 COCO 2017 数据集...")

    # 训练集
    train_url = "http://images.cocodataset.org/zips/train2017.zip"
    train_zip = os.path.join(coco_dir, "train2017.zip")

    if not os.path.exists(os.path.join(coco_dir, "train2017")):
        print(f"下载训练集: {train_url}")
        subprocess.run(["wget", "-O", train_zip, train_url], check=True)
        subprocess.run(["unzip", train_zip, "-d", coco_dir], check=True)
        os.remove(train_zip)
        print("训练集下载完成")
    else:
        print("训练集已存在，跳过下载")

    # 验证集
    val_url = "http://images.cocodataset.org/zips/val2017.zip"
    val_zip = os.path.join(coco_dir, "val2017.zip")

    if not os.path.exists(os.path.join(coco_dir, "val2017")):
        print(f"下载验证集: {val_url}")
        subprocess.run(["wget", "-O", val_zip, val_url], check=True)
        subprocess.run(["unzip", val_zip, "-d", coco_dir], check=True)
        os.remove(val_zip)
        print("验证集下载完成")
    else:
        print("验证集已存在，跳过下载")

    print("COCO 数据集下载完成！")


def download_imagenet_instructions():
    """
    打印 ImageNet 下载说明
    ImageNet 需要注册账号才能下载
    """
    print("\n" + "=" * 60)
    print("ImageNet 数据集下载说明")
    print("=" * 60)
    print("\nImageNet 数据集需要手动下载：")
    print("1. 访问 https://image-net.org/download.php")
    print("2. 注册账号并登录")
    print("3. 下载 ILSVRC2012 训练集和验证集")
    print("4. 将下载的文件解压到 datasets/imagenet/ 目录")
    print("\n目录结构应为：")
    print("  datasets/imagenet/")
    print("    ├── train/")
    print("    │   ├── n01440764/")
    print("    │   ├── n01443537/")
    print("    │   └── ...")
    print("    └── val/")
    print("        ├── n01440764/")
    print("        ├── n01443537/")
    print("        └── ...")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载训练数据集")
    parser.add_argument("--dataset", type=str, choices=['coco', 'imagenet', 'both'],
                        default='coco', help="要下载的数据集")
    parser.add_argument("--data_root", type=str, default="../datasets",
                        help="数据集保存根目录")

    args = parser.parse_args()

    if args.dataset in ['coco', 'both']:
        download_coco(args.data_root)

    if args.dataset in ['imagenet', 'both']:
        download_imagenet_instructions()

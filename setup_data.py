"""
数据准备脚本
提供多种方式获取动物图片数据集
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

import config


def download_with_progress(url: str, save_path: Path):
    """带进度条的下载"""
    print(f"下载中: {url}")
    response = urllib.request.urlopen(url)
    total_size = int(response.headers.get("content-length", 0))

    block_size = 8192
    with tqdm(total=total_size, unit="B", unit_scale=True, desc=save_path.name) as pbar:
        with open(save_path, "wb") as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"下载完成: {save_path}")


def setup_from_cifar10(target_dir: Path = None):
    """
    方案一：从 CIFAR-10 提取动物类别子集
    CIFAR-10 包含: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    动物类: bird(2), cat(3), deer(4), dog(5), frog(6), horse(7)

    适用场景：快速测试，无需额外下载（torchvision 自动下载）
    """
    target_dir = target_dir or config.DATA_DIR

    print("=" * 60)
    print("方案一：从 CIFAR-10 提取动物类别")
    print("=" * 60)

    # CIFAR-10 动物类别映射
    cifar_classes = {
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
    }

    try:
        from torchvision import datasets
    except ImportError:
        print("错误: 请先安装 torchvision (pip install torchvision)")
        return

    # 下载 CIFAR-10 训练集
    print("\n[1/3] 下载 CIFAR-10 数据集...")
    train_dataset = datasets.CIFAR10(
        root=str(target_dir / "cifar10_download"),
        train=True,
        download=True,
    )
    test_dataset = datasets.CIFAR10(
        root=str(target_dir / "cifar10_download"),
        train=False,
        download=True,
    )

    # 清理并创建目录
    print("\n[2/3] 整理数据集目录...")
    if target_dir.exists():
        shutil.rmtree(target_dir)

    for split in ["train", "val"]:
        for class_name in cifar_classes.values():
            os.makedirs(target_dir / split / class_name, exist_ok=True)

    def extract_images(dataset, split_name, max_per_class=500):
        """从 CIFAR-10 数据集中提取指定类别的图片"""
        class_counts = {name: 0 for name in cifar_classes.values()}

        for img, label in dataset:
            if label in cifar_classes and class_counts[cifar_classes[label]] < max_per_class:
                class_name = cifar_classes[label]
                img_path = target_dir / split_name / class_name / f"{class_counts[class_name]:05d}.png"
                img.save(img_path)
                class_counts[class_name] += 1

        return class_counts

    print("\n[3/3] 提取动物图片...")
    train_counts = extract_images(train_dataset, "train", max_per_class=500)
    val_counts = extract_images(test_dataset, "val", max_per_class=100)

    print("\n数据集创建完成!")
    print("-" * 40)
    print(f"{'类别':<10} {'训练集':<8} {'验证集':<8}")
    print("-" * 40)
    for cls in cifar_classes.values():
        print(f"{cls:<10} {train_counts[cls]:<8} {val_counts[cls]:<8}")
    print("-" * 40)
    print(f"总图片数: {sum(train_counts.values()) + sum(val_counts.values())}")
    print(f"\n数据保存在: {target_dir}")


def create_sample_data(target_dir: Path = None):
    """
    方案二：创建模拟的目录结构（占位），便于用户手动放置图片
    """
    target_dir = target_dir or config.DATA_DIR

    print("=" * 60)
    print("方案二：创建空的数据目录结构")
    print("=" * 60)
    print("请将图片文件放入对应的类别文件夹中。")

    # 创建目录结构
    for split in ["train", "val", "test"]:
        for class_name in config.ANIMAL_CLASSES:
            os.makedirs(target_dir / split / class_name, exist_ok=True)

    print(f"\n目录结构已创建在: {target_dir}")
    print(f"\n支持的 37 个品种: {', '.join(config.ANIMAL_CLASSES)}")
    print("\n组织方式示例:")
    print(f"  {target_dir / 'train' / 'Beagle' / 'beagle_001.jpg'}")
    print(f"  {target_dir / 'train' / 'Siamese' / 'siamese_001.jpg'}")
    print(f"  {target_dir / 'val' / 'Beagle' / 'beagle_val_001.jpg'}")
    print(f"  {target_dir / 'test' / 'Siamese' / 'siamese_test_001.jpg'}")


def setup_from_oxford_pets(target_dir: Path = None):
    """
    方案三：从 Oxford-IIIT Pet Dataset 下载 37 类猫狗品种
    包含 12 种猫 + 25 种狗，共约 7,349 张图片

    适用场景：细粒度猫狗品种分类，torchvision 自动下载
    """
    target_dir = target_dir or config.DATA_DIR

    print("=" * 60)
    print("方案三：Oxford-IIIT Pet Dataset (37 类猫狗品种)")
    print("=" * 60)

    try:
        from torchvision import datasets
    except ImportError:
        print("错误: 请先安装 torchvision (pip install torchvision)")
        return

    try:
        from torchvision.datasets import OxfordIIITPet
    except AttributeError:
        print("错误: 当前 torchvision 版本不支持 OxfordIIITPet，请升级: pip install --upgrade torchvision")
        return

    # 下载训练集 (split='trainval')
    print("\n[1/3] 下载 Oxford Pets 训练集...")
    train_dataset = datasets.OxfordIIITPet(
        root=str(target_dir / "oxford_download"),
        split="trainval",
        download=True,
    )

    # 下载测试集
    print("\n[2/3] 下载 Oxford Pets 测试集...")
    test_dataset = datasets.OxfordIIITPet(
        root=str(target_dir / "oxford_download"),
        split="test",
        download=True,
    )

    # 清理并创建目录
    print("\n[3/3] 按品种整理图片...")
    if target_dir.exists():
        shutil.rmtree(target_dir)

    # OxfordIIITPet 的 classes 属性是 _labels 的映射，我们直接使用 config 中的类别名
    breeds = config.ANIMAL_CLASSES

    for split in ["train", "val"]:
        for breed in breeds:
            os.makedirs(target_dir / split / breed, exist_ok=True)

    def extract_oxford(dataset, split_name, max_per_class=None):
        """从 Oxford Pets 数据集中提取图片到按品种命名的文件夹"""
        class_counts = {name: 0 for name in breeds}

        for img, label in dataset:
            breed_name = breeds[label] if label < len(breeds) else f"unknown_{label}"
            if max_per_class and class_counts[breed_name] >= max_per_class:
                continue
            img_path = target_dir / split_name / breed_name / f"{breed_name}_{class_counts[breed_name]:04d}.jpg"
            img.save(img_path)
            class_counts[breed_name] += 1

        return class_counts

    # 训练集提取全部，测试集全部用于验证
    train_counts = extract_oxford(train_dataset, "train")
    val_counts = extract_oxford(test_dataset, "val")

    print("\n数据集创建完成!")
    print("-" * 50)
    print(f"{'品种':<30} {'训练集':<8} {'验证集':<8}")
    print("-" * 50)
    for breed in breeds:
        tc = train_counts.get(breed, 0)
        vc = val_counts.get(breed, 0)
        if tc + vc > 0:
            print(f"{breed:<30} {tc:<8} {vc:<8}")
    print("-" * 50)
    print(f"总图片数: {sum(train_counts.values()) + sum(val_counts.values())}")
    print(f"\n数据保存在: {target_dir}")


def organize_downloaded_pets(image_dir: str = None):
    """
    方案四：整理已手动下载的 Oxford Pets 图片
    将 images/ 目录下的文件按品种自动分类到 train/val
    """
    target_dir = config.DATA_DIR

    print("=" * 60)
    print("方案四：整理已下载的 Oxford Pets 图片")
    print("=" * 60)

    # 询问图片所在目录
    if image_dir is None:
        image_dir = input("\n请输入图片所在目录路径: ").strip().strip('"').strip("'")

    src = Path(image_dir)
    if not src.exists() and src.parent.exists() and src.parent.name.lower() == "images":
        src = src.parent
    if src.is_dir() and src.name.lower() != "images" and (src / "images").exists():
        src = src / "images"
    if not src.exists():
        print(f"错误: 目录不存在 - {src}")
        return

    breeds = config.ANIMAL_CLASSES
    breed_lookup = {breed.lower(): breed for breed in breeds}

    def normalize_name(name: str) -> str:
        return name.strip().lower().replace("-", "_").replace(" ", "_")

    # 扫描文件，按品种分组
    breed_files = {b: [] for b in breeds}
    unknown_files = []

    print(f"\n[1/3] 扫描图片目录: {src}")
    for f in src.iterdir():
        if not f.name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        # 从文件名提取品种名：去掉末尾的 _数字 部分
        parts = f.stem.rsplit("_", 1)
        if len(parts) < 2:
            unknown_files.append(f.name)
            continue
        breed_candidate = normalize_name(parts[0])
        # 尝试匹配已知品种
        canonical_breed = breed_lookup.get(breed_candidate)
        if canonical_breed:
            breed_files[canonical_breed].append(f)
        else:
            unknown_files.append(f.name)

    found_total = sum(len(v) for v in breed_files.values())
    print(f"  匹配到 {found_total} 张图片 ({len(unknown_files)} 张未匹配)")

    if found_total == 0:
        print("错误: 未找到匹配的图片。请检查目录路径是否正确。")
        return

    # 清理并创建目录
    print("\n[2/3] 创建目标目录结构...")
    if target_dir.exists():
        shutil.rmtree(target_dir)

    for split in ["train", "val"]:
        for breed in breeds:
            os.makedirs(target_dir / split / breed, exist_ok=True)

    # 整理图片（80% 训练，20% 验证，按品种分层采样）
    print("\n[3/3] 正在整理图片...")
    import random
    random.seed(42)

    train_counts = {}
    val_counts = {}

    for breed in breeds:
        files = breed_files[breed]
        if not files:
            train_counts[breed] = 0
            val_counts[breed] = 0
            continue

        random.shuffle(files)
        split_idx = int(len(files) * 0.8)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        for i, f in enumerate(train_files):
            shutil.copy2(f, target_dir / "train" / breed / f"{breed}_{i:04d}.jpg")
        for i, f in enumerate(val_files):
            shutil.copy2(f, target_dir / "val" / breed / f"{breed}_{i:04d}.jpg")

        train_counts[breed] = len(train_files)
        val_counts[breed] = len(val_files)

    # 打印统计
    print("\n整理完成!")
    print("-" * 50)
    print(f"{'品种':<30} {'训练集':<8} {'验证集':<8}")
    print("-" * 50)
    for breed in breeds:
        tc = train_counts.get(breed, 0)
        vc = val_counts.get(breed, 0)
        if tc + vc > 0:
            print(f"{breed:<30} {tc:<8} {vc:<8}")
    print("-" * 50)
    print(f"总图片数: {sum(train_counts.values()) + sum(val_counts.values())}")
    print(f"\n数据保存在: {target_dir}")

    if unknown_files:
        print(f"\n注意: {len(unknown_files)} 张图片未匹配到已知品种")
        print(f"未匹配文件示例: {unknown_files[:5]}")
        print("请检查文件名格式是否为 Oxford Pets 标准格式 (品种名_编号.jpg)")


def main():
    """主入口"""
    print("动物识别项目 - 数据准备工具\n")
    print("请选择数据获取方式:")
    print("  1. 从 CIFAR-10 提取动物图片（快速开始）")
    print("  2. 创建空目录结构（手动放置图片）")
    print("  3. Oxford-IIIT Pet Dataset（自动下载，需 torchvision）")
    print("  4. 整理已手动下载的 Oxford Pets 图片")

    choice = input("\n请选择 (1/2/3/4): ").strip()

    if choice == "1":
        setup_from_cifar10()
    elif choice == "2":
        create_sample_data()
    elif choice == "3":
        setup_from_oxford_pets()
    elif choice == "4":
        organize_downloaded_pets()
    else:
        print("无效选择，退出。")

    print("\n完成! 接下来可以运行 python train.py 开始训练。")


if __name__ == "__main__":
    main()

"""Generate report figures for the animal recognition project."""

import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import config
from data_utils import denormalize, get_dataloaders
from predict import AnimalPredictor


FIGURE_DIR = config.OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def save_figure(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")


def load_history() -> dict:
    history_path = config.OUTPUT_DIR / "training_history.json"
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def imagefolder_samples(root: Path, count: int):
    dataset = datasets.ImageFolder(root=str(root))
    indices = list(range(len(dataset.samples)))
    random.shuffle(indices)
    return [dataset.samples[i] for i in indices[:count]], dataset.classes


def plot_dataset_samples():
    samples, classes = imagefolder_samples(config.TRAIN_DIR, 24)
    fig, axes = plt.subplots(4, 6, figsize=(14, 9))
    fig.suptitle("数据集部分样本截图", fontsize=18, fontweight="bold")

    for ax, (image_path, label_idx) in zip(axes.flat, samples):
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(classes[label_idx], fontsize=9)
        ax.axis("off")

    for ax in axes.flat[len(samples) :]:
        ax.axis("off")

    save_figure(FIGURE_DIR / "dataset_samples.png")


def plot_augmentation_examples():
    samples, classes = imagefolder_samples(config.TRAIN_DIR, 1)
    image_path, label_idx = samples[0]
    image = Image.open(image_path).convert("RGB")
    aug = config.TRAIN_AUGMENTATION
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.IMG_SIZE,
                scale=aug["random_resized_crop_scale"],
                ratio=aug["random_resized_crop_ratio"],
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5 if aug["random_horizontal_flip"] else 0.0),
            transforms.RandomRotation(aug["random_rotation"], fill=128),
            transforms.ColorJitter(*aug["color_jitter"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=aug["normalize_mean"], std=aug["normalize_std"]),
        ]
    )

    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    fig.suptitle(f"数据增强效果示例图: {classes[label_idx]}", fontsize=18, fontweight="bold")

    axes.flat[0].imshow(image)
    axes.flat[0].set_title("原图", fontsize=10)
    axes.flat[0].axis("off")

    for i, ax in enumerate(axes.flat[1:], start=1):
        tensor = transform(image)
        aug_image = denormalize(tensor).permute(1, 2, 0).numpy()
        ax.imshow(aug_image)
        ax.set_title(f"增强 {i}", fontsize=10)
        ax.axis("off")

    save_figure(FIGURE_DIR / "augmentation_examples.png")


def plot_training_curves():
    history = load_history()
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(9, 5.2))
    plt.plot(epochs, history["train_loss"], marker="o", label="训练损失")
    plt.plot(epochs, history["val_loss"], marker="s", label="验证损失")
    plt.title("训练损失与验证损失曲线图", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(FIGURE_DIR / "loss_curve.png")

    plt.figure(figsize=(9, 5.2))
    plt.plot(epochs, history["train_acc"], marker="o", label="训练准确率")
    plt.plot(epochs, history["val_acc"], marker="s", label="验证准确率")
    plt.title("训练准确率与验证准确率曲线图", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(FIGURE_DIR / "accuracy_curve.png")


def compute_confusion_matrix():
    _, val_loader, _ = get_dataloaders(num_workers=0)
    predictor = AnimalPredictor()
    model = predictor.model
    device = predictor.device
    num_classes = len(predictor.classes)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    model.eval()
    with torch.inference_mode():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()
            for true_label, pred_label in zip(labels_np, preds):
                matrix[true_label, pred_label] += 1

    return matrix, predictor.classes


def plot_confusion_matrix():
    matrix, classes = compute_confusion_matrix()
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("混淆矩阵热力图", fontsize=18, fontweight="bold")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("按真实类别归一化比例", rotation=270, labelpad=18)
    save_figure(FIGURE_DIR / "confusion_matrix_heatmap.png")


def plot_prediction_examples():
    source_dir = config.TEST_DIR if config.TEST_DIR.exists() else config.VAL_DIR
    samples, classes = imagefolder_samples(source_dir, 12)
    predictor = AnimalPredictor()

    cols = 4
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    fig.suptitle("部分测试样本预测结果示例图", fontsize=18, fontweight="bold")

    for ax, (image_path, label_idx) in zip(axes.flat, samples):
        image = Image.open(image_path).convert("RGB")
        results, is_out_of_scope = predictor.predict_pil(image, top_k=1)
        pred_name, prob = results[0]
        true_name = classes[label_idx]
        status = "范围外" if is_out_of_scope else pred_name
        color = "tab:green" if pred_name == true_name and not is_out_of_scope else "tab:red"

        ax.imshow(image)
        ax.set_title(
            f"真实: {true_name}\n预测: {status} ({prob:.1%})",
            fontsize=9,
            color=color,
        )
        ax.axis("off")

    for ax in axes.flat[len(samples) :]:
        ax.axis("off")

    save_figure(FIGURE_DIR / "prediction_examples.png")


def main():
    plot_dataset_samples()
    plot_augmentation_examples()
    plot_training_curves()
    plot_confusion_matrix()
    plot_prediction_examples()
    print(f"\nAll figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()

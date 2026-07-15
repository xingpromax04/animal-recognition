"""Training script."""

import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from tqdm import tqdm

import config
from data_utils import get_dataloaders
from model import build_model, get_model_info


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]")
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler.is_enabled()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"})

    return running_loss / total, 100.0 * correct / total


@torch.inference_mode()
def validate(model, dataloader, criterion, scaler, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.EPOCHS} [Val]")
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        with autocast(enabled=scaler.is_enabled()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"})

    return running_loss / total, 100.0 * correct / total


def main():
    print("=" * 60)
    print("Animal classification training")
    print("=" * 60)

    device = torch.device(config.DEVICE)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print(f"Device: {device}")

    print("\n[1/4] Loading data...")
    train_loader, val_loader, _ = get_dataloaders(num_workers=config.NUM_WORKERS)
    print(f"  train size: {len(train_loader.dataset)}")
    print(f"  val size: {len(val_loader.dataset)}")
    print(f"  classes: {len(train_loader.dataset.classes)}")

    class_meta_path = config.OUTPUT_DIR / "class_names.json"
    class_meta = {
        "class_names": train_loader.dataset.classes,
        "num_classes": len(train_loader.dataset.classes),
    }
    with open(class_meta_path, "w", encoding="utf-8") as f:
        json.dump(class_meta, f, ensure_ascii=False, indent=2)
    print(f"  class mapping saved to: {class_meta_path}")

    print("\n[2/4] Building model...")
    model = build_model().to(device)
    info = get_model_info(model)
    print(f"  backbone: {info['backbone']}")
    print(f"  total params: {info['total_params']:,}")
    print(f"  trainable params: {info['trainable_params']:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR)
    scaler = GradScaler(enabled=config.AMP and device.type == "cuda")

    print("\n[3/4] Training...")
    best_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, scaler, device, epoch)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n  Epoch {epoch}/{config.EPOCHS}")
        print(f"    train loss: {train_loss:.4f} | train acc: {train_acc:.2f}%")
        print(f"    val loss: {val_loss:.4f} | val acc: {val_acc:.2f}%")
        print(f"    lr: {current_lr:.6f}")
        print("-" * 50)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  saved best model: {best_acc:.2f}%")

        if epoch % config.SAVE_INTERVAL == 0:
            save_path = config.MODEL_SAVE_DIR / f"epoch_{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  saved checkpoint: {save_path}")

    torch.save(model.state_dict(), config.LAST_MODEL_PATH)

    history_path = config.OUTPUT_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    model_meta_path = config.MODEL_META_PATH
    model_meta = {
        "backbone": info["backbone"],
        "num_classes": len(train_loader.dataset.classes),
        "class_names": train_loader.dataset.classes,
        "img_size": config.IMG_SIZE,
        "best_epoch": best_epoch,
        "best_accuracy": best_acc,
    }
    with open(model_meta_path, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, ensure_ascii=False, indent=2)

    print("\n[4/4] Done.")
    print(f"  best val acc: {best_acc:.2f}% (epoch {best_epoch})")
    print(f"  best model: {config.BEST_MODEL_PATH}")
    print(f"  model meta: {model_meta_path}")
    print(f"  history: {history_path}")
    return history


if __name__ == "__main__":
    main()

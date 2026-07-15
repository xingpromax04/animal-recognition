"""Data loading and preprocessing."""

from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import config


def get_train_transform() -> transforms.Compose:
    aug = config.TRAIN_AUGMENTATION
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.IMG_SIZE,
                scale=aug["random_resized_crop_scale"],
                ratio=aug["random_resized_crop_ratio"],
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5 if aug["random_horizontal_flip"] else 0.0),
            transforms.RandomRotation(aug["random_rotation"]),
            transforms.ColorJitter(*aug["color_jitter"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=aug["normalize_mean"], std=aug["normalize_std"]),
            transforms.RandomErasing(p=aug["random_erasing_p"], value=0),
        ]
    )


def get_val_transform() -> transforms.Compose:
    aug = config.TRAIN_AUGMENTATION
    return transforms.Compose(
        [
            transforms.Resize(
                config.VAL_RESIZE_SIZE,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=aug["normalize_mean"], std=aug["normalize_std"]),
        ]
    )


def _build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": config.PIN_MEMORY,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = config.PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = 2
    if shuffle:
        loader_kwargs["drop_last"] = True
    return DataLoader(dataset, **loader_kwargs)


def get_dataloaders(
    data_dir: Optional[Path] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    data_dir = data_dir or config.DATA_DIR
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = config.NUM_WORKERS if num_workers is None else num_workers

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = datasets.ImageFolder(root=str(data_dir / "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(data_dir / "val"), transform=val_transform)

    train_loader = _build_loader(train_dataset, batch_size, True, num_workers)
    val_loader = _build_loader(val_dataset, batch_size, False, num_workers)

    test_loader = None
    test_dir = data_dir / "test"
    if test_dir.exists():
        test_dataset = datasets.ImageFolder(root=str(test_dir), transform=val_transform)
        test_loader = _build_loader(test_dataset, batch_size, False, num_workers)

    config.ANIMAL_CLASSES = train_dataset.classes
    config.NUM_CLASSES = len(train_dataset.classes)
    return train_loader, val_loader, test_loader


def denormalize(tensor, mean=None, std=None):
    mean = mean or config.TRAIN_AUGMENTATION["normalize_mean"]
    std = std or config.TRAIN_AUGMENTATION["normalize_std"]
    img = tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img.clamp(0, 1)

"""Model definition."""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

import config


def _get_weights(backbone: str, pretrained: bool):
    if not pretrained:
        return None

    backbone = backbone.lower()
    if backbone == "resnet50":
        weights_enum = getattr(models, "ResNet50_Weights", None)
    elif backbone == "efficientnet_b0":
        weights_enum = getattr(models, "EfficientNet_B0_Weights", None)
    else:
        weights_enum = None

    return weights_enum.DEFAULT if weights_enum is not None else None


def _build_resnet50(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    model = models.resnet50(weights=_get_weights("resnet50", pretrained))
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )
    return model


def _build_legacy_resnet50(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    model = models.resnet50(weights=_get_weights("resnet50", pretrained))
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )
    return model


def _build_efficientnet_b0(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    model = models.efficientnet_b0(weights=_get_weights("efficientnet_b0", pretrained))
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_model(
    num_classes: Optional[int] = None,
    pretrained: Optional[bool] = None,
    freeze_layers: Optional[bool] = None,
    dropout: Optional[float] = None,
    backbone: Optional[str] = None,
) -> nn.Module:
    num_classes = num_classes or config.NUM_CLASSES
    pretrained = config.PRETRAINED if pretrained is None else pretrained
    freeze_layers = config.FREEZE_LAYERS if freeze_layers is None else freeze_layers
    dropout = config.DROPOUT if dropout is None else dropout
    backbone = (backbone or config.MODEL_BACKBONE).lower()

    if backbone == "resnet50":
        model = _build_resnet50(num_classes, pretrained, dropout)
    elif backbone == "efficientnet_b0":
        model = _build_efficientnet_b0(num_classes, pretrained, dropout)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
        if backbone == "resnet50":
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.classifier.parameters():
                param.requires_grad = True

    model._codex_backbone = backbone
    return model


def infer_checkpoint_format(state_dict: dict) -> tuple[str, str]:
    keys = set(state_dict.keys())
    if any(key.startswith("features.") for key in keys):
        return "efficientnet_b0", "modern"
    if any(key.startswith("conv1.") for key in keys) or "fc.4.weight" in keys:
        return "resnet50", "legacy"
    return config.MODEL_BACKBONE, "modern"


def build_model_from_state_dict(state_dict: dict, num_classes: Optional[int] = None) -> nn.Module:
    backbone, head_variant = infer_checkpoint_format(state_dict)
    num_classes = num_classes or config.NUM_CLASSES

    if backbone == "resnet50" and head_variant == "legacy":
        model = _build_legacy_resnet50(num_classes, pretrained=False, dropout=config.DROPOUT)
    else:
        model = build_model(num_classes=num_classes, pretrained=False, backbone=backbone)

    return model


def get_model_info(model: nn.Module) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device": next(model.parameters()).device,
        "backbone": getattr(model, "_codex_backbone", "unknown"),
    }


if __name__ == "__main__":
    model = build_model()
    info = get_model_info(model)
    print(model)
    print("\nModel info:")
    print(f"  total params: {info['total_params']:,}")
    print(f"  trainable params: {info['trainable_params']:,}")
    print(f"  backbone: {info['backbone']}")

    dummy_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    output = model(dummy_input)
    print(f"\ninput shape: {dummy_input.shape}")
    print(f"output shape: {output.shape}")

"""Prediction helpers."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

import config
from data_utils import get_val_transform
from model import build_model, build_model_from_state_dict, infer_checkpoint_format


class AnimalPredictor:
    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        self.device = torch.device(device or config.DEVICE)
        self.classes = self._load_class_names()
        self.transform = get_val_transform()
        self.model_meta = self._load_model_meta()
        self.model = self._load_model(model_path or config.BEST_MODEL_PATH).to(self.device)
        self.model.eval()

    @staticmethod
    def _load_class_names() -> List[str]:
        class_meta_path = config.OUTPUT_DIR / "class_names.json"
        if class_meta_path.exists():
            try:
                with open(class_meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                class_names = data.get("class_names")
                if isinstance(class_names, list) and class_names:
                    return class_names
            except Exception:
                pass
        return config.ANIMAL_CLASSES

    @staticmethod
    def _load_model_meta() -> dict:
        if config.MODEL_META_PATH.exists():
            try:
                with open(config.MODEL_META_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    def _candidate_backbones(self) -> List[str]:
        candidates = []
        meta_backbone = self.model_meta.get("backbone")
        if isinstance(meta_backbone, str) and meta_backbone:
            candidates.append(meta_backbone.lower())
        candidates.append(config.LEGACY_BACKBONE)
        candidates.append(config.MODEL_BACKBONE)

        unique = []
        for backbone in candidates:
            if backbone not in unique:
                unique.append(backbone)
        return unique

    def _load_state_dict(self, model_path: Path):
        try:
            return torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(model_path, map_location=self.device)

    def _load_model(self, model_path: Path):
        if not model_path.exists():
            print(f"Warning: model file not found: {model_path}")
            backbone = self.model_meta.get("backbone", config.MODEL_BACKBONE)
            return build_model(backbone=backbone, pretrained=True)

        state_dict = self._load_state_dict(model_path)
        inferred_backbone, head_variant = infer_checkpoint_format(state_dict)
        candidates = [inferred_backbone] + self._candidate_backbones()
        last_error = None
        for backbone in candidates:
            try:
                if backbone == "resnet50" and head_variant == "legacy":
                    model = build_model_from_state_dict(state_dict, num_classes=len(self.classes))
                else:
                    model = build_model(num_classes=len(self.classes), backbone=backbone, pretrained=False)
                model.load_state_dict(state_dict)
                self.model_meta.setdefault("backbone", backbone)
                print(f"Model loaded: {model_path} ({backbone})")
                return model
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Failed to load model weights from {model_path}") from last_error

    @torch.inference_mode()
    def predict_pil(self, image: Image.Image, top_k: int = 3):
        image = image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device, non_blocking=self.device.type == "cuda")
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)

        top_k = max(1, min(top_k, probabilities.shape[1]))
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_index = idx.item()
            class_name = self.classes[class_index] if class_index < len(self.classes) else f"unknown_{class_index}"
            results.append((class_name, prob.item()))
        top1_prob = results[0][1] if results else 0.0
        top2_prob = results[1][1] if len(results) > 1 else 0.0
        is_out_of_scope = (
            top1_prob < config.REJECTION_THRESHOLD
            or (top1_prob - top2_prob) < config.REJECTION_MARGIN
        )
        return results, is_out_of_scope

    def predict(self, image_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
        with Image.open(image_path) as image:
            results, _ = self.predict_pil(image, top_k=top_k)
            return results

    def predict_batch(self, image_paths: List[str], top_k: int = 3) -> List[List[Tuple[str, float]]]:
        return [self.predict(path, top_k) for path in image_paths]

    def predict_with_visualization(self, image_path: str, top_k: int = 3):
        with Image.open(image_path) as image:
            results, is_out_of_scope = self.predict_pil(image, top_k=top_k)
            names = [r[0] for r in results]
            probs = [r[1] for r in results]
            return image.convert("RGB"), names, probs, is_out_of_scope


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Animal classification prediction")
    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model weights")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top results to return")
    args = parser.parse_args()

    predictor = AnimalPredictor(model_path=Path(args.model_path) if args.model_path else None)
    results = predictor.predict(args.image_path, top_k=args.top_k)

    print(f"\nPrediction: {Path(args.image_path).name}")
    print("=" * 40)
    for i, (class_name, prob) in enumerate(results, 1):
        bar = "#" * max(1, int(prob * 30))
        print(f"  {i}. {class_name}: {prob:.2%} {bar}")
    print("=" * 40)


if __name__ == "__main__":
    main()

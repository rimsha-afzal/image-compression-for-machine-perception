"""Load a frozen ImageNet-pretrained ResNet-50 classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


RESNET50_WEIGHTS = models.ResNet50_Weights.IMAGENET1K_V2


@dataclass(frozen=True)
class Prediction:
    predicted_index: int
    predicted_class: str
    top1_confidence: float
    top5: list[dict[str, float | int | str]]
    logits: list[float] | None = None
    probabilities: list[float] | None = None


# Build the standard ResNet-50 ImageNet preprocessing chain.
def build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            # resize, tensor conversion and normalization
            RESNET50_WEIGHTS.transforms(),
        ]
    )


# Select the requested PyTorch device, defaulting to CUDA when available.
def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Infer the active device from the model when the caller did not specify one.
def get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


# Load ResNet-50 with ImageNet weights and freeze it for inference.
def load_resnet50(device: str | torch.device | None = None) -> tuple[nn.Module, list[str]]:
    model = models.resnet50(weights=RESNET50_WEIGHTS)
    model.eval()
    model.requires_grad_(False)

    selected_device = get_device(str(device) if device is not None else None)
    model.to(selected_device)
    return model, list(RESNET50_WEIGHTS.meta["categories"])


# Open an image path or pass through an existing PIL image.
def load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.open(image)


# Run a single-image ResNet-50 prediction and return top-1 plus optional details.
@torch.inference_mode()
def predict_image(
    image: str | Path | Image.Image,
    model: nn.Module,
    class_names: list[str],
    preprocess: transforms.Compose | None = None,
    device: str | torch.device | None = None,
    return_top5: bool = True,
    return_logits: bool = False,
    return_probabilities: bool = False,
) -> Prediction:
    selected_device = (
        get_device(str(device)) if device is not None else get_model_device(model)
    )
    transform = preprocess or build_preprocess()

    if isinstance(image, Image.Image):
        input_tensor = transform(image).unsqueeze(0).to(selected_device)
    else:
        with load_image(image) as pil_image:
            input_tensor = transform(pil_image).unsqueeze(0).to(selected_device)


    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    top1_confidence, top1_index = probabilities[0].max(dim=0)

    top5_predictions: list[dict[str, float | int | str]] = []
    if return_top5:
        top5_confidences, top5_indices = probabilities[0].topk(5)
        top5_predictions = [
            {
                "index": int(index),
                "class": class_names[int(index)],
                "confidence": float(confidence),
            }
            for confidence, index in zip(top5_confidences.cpu(), top5_indices.cpu())
        ]

    return Prediction(
        predicted_index=int(top1_index),
        predicted_class=class_names[int(top1_index)],
        top1_confidence=float(top1_confidence),
        top5=top5_predictions,
        logits=output[0].cpu().tolist() if return_logits else None,
        probabilities=probabilities[0].cpu().tolist() if return_probabilities else None,
    )


# Convert a prediction dataclass into a JSON-serialisable dictionary.
def prediction_to_dict(prediction: Prediction) -> dict[str, Any]:
    return {
        "predicted_index": prediction.predicted_index,
        "predicted_class": prediction.predicted_class,
        "top1_confidence": prediction.top1_confidence,
        "top5": prediction.top5,
        "logits": prediction.logits,
        "probabilities": prediction.probabilities,
    }

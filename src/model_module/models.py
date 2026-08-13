from typing import Callable, Dict

import torch.nn as nn
from torchvision import models

NUM_CLASSES = 3

MODEL_FACTORY: Dict[str, Callable[[], nn.Module]] = {}


def register_model(name: str):
    def decorator(fn: Callable[[], nn.Module]) -> Callable[[], nn.Module]:
        MODEL_FACTORY[name] = fn
        return fn

    return decorator


@register_model("teacher_resnet50")
def get_teacher() -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


@register_model("student_mobilenet_v3_small")
def get_student() -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    return model


def ModelFactory(name: str) -> Callable[[], nn.Module]:
    if name not in MODEL_FACTORY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(MODEL_FACTORY)}")
    return MODEL_FACTORY[name]


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

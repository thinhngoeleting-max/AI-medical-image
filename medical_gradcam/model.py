from pathlib import Path
import warnings

import torch
from torch import nn
from torchvision.models import (ResNet18_Weights, ResNet50_Weights, resnet18, resnet50)


def build_model(architecture, num_classes=2, pretrained=False):
    if architecture == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    elif architecture == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError("architecture must be resnet18 or resnet50")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(path, config, device):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if "model_state" in checkpoint:
        metadata = checkpoint
        classes = checkpoint["classes"]
        architecture = checkpoint["config"]["architecture"]
        image_size = checkpoint["config"]["image_size"]
        state = checkpoint["model_state"]
    else:
        warnings.warn("Legacy notebook weights: preprocessing/class mapping assumed; "
                      "training provenance and patient separation are unverified.")
        classes = ["NORMAL", "PNEUMONIA"]
        architecture, image_size = config.architecture, config.image_size
        state, metadata = checkpoint, {"legacy": True}
    model = build_model(architecture, len(classes))
    model.load_state_dict(state)
    model.to(device).eval()
    return model, classes, image_size, metadata

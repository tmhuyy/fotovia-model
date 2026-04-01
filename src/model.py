import torch.nn as nn
from torchvision import models

from src.config import MODEL_NAME, NUM_CLASSES, PRETRAINED


def build_model():
    # if MODEL_NAME == "resnet18":
    #     weights = models.ResNet18_Weights.DEFAULT if PRETRAINED else None
    #     model = models.resnet18(weights=weights)
    #     in_features = model.fc.in_features
    #     model.fc = nn.Linear(in_features, NUM_CLASSES)
    #     return model

    if MODEL_NAME == "resnext":
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if PRETRAINED else None
        model = models.resnext50_32x4d(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
        return model

    raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")
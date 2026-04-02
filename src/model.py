import torch.nn as nn
from torchvision import models

from src.config import MODEL_NAME, NUM_CLASSES, PRETRAINED


def build_model():
    if MODEL_NAME == "resnet18": # Test end to end flow
        weights = models.ResNet18_Weights.DEFAULT if PRETRAINED else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
        return model

    if MODEL_NAME == "resnext":
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if PRETRAINED else None
        model = models.resnext50_32x4d(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
        return model
    
    if MODEL_NAME == "wide_resnet":
        weights = models.Wide_ResNet50_2_Weights.DEFAULT if PRETRAINED else None
        model = models.wide_resnet50_2(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
        return model
    
    if MODEL_NAME in ("efficientnet", "efficientnet_b0"):
        weights = models.EfficientNet_B0_Weights.DEFAULT if PRETRAINED else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
        return model

    raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")
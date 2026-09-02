import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from pathlib import Path

def create_model(num_classes: int, freeze_base: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model

def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_save_path = target_dir_path/model_name
    print(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def load_model(path: str, num_classes: int) -> nn.Module:
    model = create_model(num_classes, freeze_base=False)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model
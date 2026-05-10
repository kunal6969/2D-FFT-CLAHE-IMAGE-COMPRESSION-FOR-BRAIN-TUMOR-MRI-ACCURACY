import torch
import torch.nn as nn
from torchvision.models import resnet18

def get_trained_model(model_path, device, num_classes=2):
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

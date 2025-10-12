import torch
from torch import nn
import torchvision.models as models
from src import config

class TireClassificationModel(nn.Module):
    def __init__(self, device):
        super().__init__()

        # Load pretrained ResNet50
        self.resnet50_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Hold the in_features of the FC layer of ResNet50
        in_features = self.resnet50_model.fc.in_features

        # Remove the FC layer of ResNet50
        self.resnet50_model.fc = nn.Identity()

        # Freeze all layers (transfer learning)
        for param in self.resnet50_model.parameters():
            param.requires_grad = False

        # Define a model with your own FC layer
        self.fc_model = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 2)
        )

        # Define a model combining ResNet50 and Custom FC Layer
        self.model = nn.Sequential(
            self.resnet50_model,
            self.fc_model
        )

        self.fc_model.apply(self.init_weights)
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.model(x)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def save(self):
        torch.save(self.fc_model.state_dict(), config.MODEL_PATH)
        print(f"• Model saved to {config.MODEL_PATH}")

    def load(self):
        self.fc_model.load_state_dict(torch.load(config.MODEL_PATH, map_location=self.device))
        self.to(self.device)
        print(f"• Model loaded from {config.MODEL_PATH}")
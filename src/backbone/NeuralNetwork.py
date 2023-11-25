
import torch
import torch.nn as nn
import torchvision.models as models

class ClassificationRCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(ClassificationRCNN, self).__init__()
        # Load a pre-trained MobileNetV2 model and use its features
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.backbone = mobilenet.features

        # Add custom layers for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1280, 1024),  # MobileNetV2's last channel is 1280
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



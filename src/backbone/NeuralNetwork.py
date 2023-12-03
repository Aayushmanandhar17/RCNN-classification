import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import collections


class ClassificationRCNN(nn.Module):
    def __init__(self, num_classes=80):
        super(ClassificationRCNN, self).__init__()
        # Load the pretrained Faster R-CNN model
        faster_rcnn = models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
        
        # Extract the ResNet50 backbone
        self.resnet_backbone = faster_rcnn.backbone.body
        
        # Classification head
        self.ffn = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract features using the backbone
        x = self.resnet_backbone(x)
        print(x['0'].shape)
        print(x['1'].shape)
        print(x['2'].shape)
        print(x['3'].shape)

        # Use the output from the last layer
        x = x['3']  # Extracting the tensor with shape [batch_size, 2048, 7, 7]

        # Apply global average pooling to convert feature maps to a vector
        x = F.adaptive_avg_pool2d(x, (1, 1))  # This will change the shape to [batch_size, 2048, 1, 1]
        x = torch.flatten(x, 1)  # Flattening to [batch_size, 2048]

        # Classification head
        x = self.ffn(x)
        return x





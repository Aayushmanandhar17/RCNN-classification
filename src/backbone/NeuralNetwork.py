
import torch
import torch.nn as nn
import torchvision.models as models

class ClassificationRCNN(nn.Module):
    def __init__(self, num_classes=80):
        super(ClassificationRCNN, self).__init__()
        faster_rcnn=models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
        #only extracting the ResNet50 backbone
        self.resnet_backbone=faster_rcnn.backbone.body
        
        self.ffn=nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048,512),
            nn.Relu(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.resnet_backbone(x)
        #average pooling
        x= torch.mean(x, dim=[2,3])
        x=self.ffn(x)
        return x
    
        
        
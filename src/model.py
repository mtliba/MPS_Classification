# src/models.py
import torch
import torch.nn as nn
from torchvision import models
import timm

class ClassificationModel(nn.Module):
    def __init__(self, base_model_name, num_classes, use_sigmoid=False):
        """
        Initialize the classification model.
        
        Args:
            base_model_name (str): Name of the base model to use ('vit_small', 'resnet_small', 'efficientnet_small').
            num_classes (int): Number of output classes for the classification task.
            use_sigmoid (bool): If True, use independent classifiers with sigmoid; otherwise, use softmax.
        """
        super(ClassificationModel, self).__init__()
        self.use_sigmoid = use_sigmoid

        # Initialize the base model
        if base_model_name == 'vit_small':
            self.base_model = timm.create_model('vit_small_patch16_224', pretrained=True)
            self.in_features = self.base_model.head.in_features
            self.base_model.head = nn.Identity()  # Remove the original head
            
        elif base_model_name == 'resnet_small':
            self.base_model = models.resnet18(pretrained=True)
            self.in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove the original head
            
        elif base_model_name == 'efficientnet_small':
            self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
            self.in_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()  # Remove the original head
        
        # Define the classification head
        if use_sigmoid:
            # Independent classifiers with sigmoid
            self.classifier = nn.ModuleList([nn.Sequential(
                nn.Linear(self.in_features, 1),
                nn.Sigmoid()
            ) for _ in range(num_classes)])
        else:
            # Softmax classification
            self.classifier = nn.Linear(self.in_features, num_classes)
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Extract features from the base model
        features = self.base_model(x)

        if self.use_sigmoid:
            # Apply independent classifiers
            outputs = [clf(features) for clf in self.classifier]
            outputs = torch.cat(outputs, dim=1)
        else:
            # Apply softmax classifier
            outputs = self.classifier(features)
            outputs = self.softmax(outputs)

        return outputs
if __name__ == "__main__":
    model = ClassificationModel('vit_small', 5, use_sigmoid=False)
    print(model)

"""
model.py
- LeaflictionCNN class
To define a neural network in PyTorch, we create a class that inherits from nn.Module. 
We define the layers of the network in the __init__ function and specify how data will
pass through the network in the forward function. To accelerate operations in the neural
network, we move it to the accelerator such as CUDA, MPS, MTIA, or XPU. 
If the current accelerator is available, we will use it. Otherwise, we use the CPU.

https://docs.pytorch.org/tutorials/beginner/basics/intro.html
https://www.datacamp.com/tutorial/pytorch-cnn-tutorial
"""

from scripts.utils.logger import get_logger
import torch
from torch import nn

logger = get_logger(__name__)

class LeaflictionCNN(nn.Module):
    """LeaflictionCNN implement a simple CNN for leave disease detection"""
    def __init__(self, num_classes):
        """
        CNN architecture
        Conv block with BN and ReLU
        Params:
        - num_classes: Number of classes to predict.
        """
        super().__init__()
        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        self.features = nn.Sequential(
            CBR(3, 16), nn.MaxPool2d(2),
            CBR(16, 32), nn.MaxPool2d(2),
            CBR(32, 64), nn.MaxPool2d(2),
            CBR(64, 128), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """forward pass"""
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def load(self, path:str, device:str) -> torch.nn.Module:
        """loads the pre-trained model at the specified path"""
        model = self.load_state_dict(torch.load(path, map_location=device))
        return model

    
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

    def __init__(self, num_classes, device):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # (B, 16, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(2),                # (B, 16, 128, 128)

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # (B, 32, 64, 64)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.device = device

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), str(path))

    def load(self, path: str | Path, strict: bool = True) -> None:
        self.load_state_dict(torch.load(str(path), map_location="cpu"), strict=strict)


    def compile(self, loss, optimizer, lr):
        self.loss = loss
        self.lr = lr
        self.optimizer = optimizer

    def fit(self, dataloader):
        size = len(dataloader.dataset)
        self.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self(X)
            loss = self.loss(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def eval(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self(X)
                test_loss += self.loss(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        logger.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    def predict(self, data, classes):
        self.eval()
        x, y = data[0][0], data[0][1]
        with torch.no_grad():
            x = x.to(self.device)
            pred = self(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            logger.info(f'Predicted: "{predicted}", Actual: "{actual}"')



if __name__ == "__main__":
    import sys

    try:
        logger.info("Starting model.py")

        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        logger.info(f"Using {device} device")

        model = LeaflictionCNN().to(device)
        logger.info(model)

    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)

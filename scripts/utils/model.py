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
from pathlib import Path


logger = get_logger(__name__)


class LeaflictionCNN(nn.Module):

    def __init__(self, num_classes, device):
        """
        Building blocks of convolutional neural network.

        Parameters:
            * num_classes: Number of classes to predict.
        """
        logger.info("Initializing LeaflictionCNN")

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # (B, 16, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(2),                # (B, 16, 128, 128)

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # (B, 32, 64, 64)
            nn.AdaptiveAvgPool2d(1),         # -> [B, 32, 1, 1] quel que soit HxW
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    # def save(self, path: str | Path) -> None:
    #     torch.save(self.state_dict(), str(path))

    # def save_model(model, class_names):
    #     """Save model weights + class names"""
    #     import datetime
    #     timestamp = datetime.now()
    #     path ="LeaflictionCNN" + + "_" + timestamp + ".pt"
    #     torch.save(
    #         {
    #             "state_dict": model.state_dict(),
    #             "class_names": list(class_names),  # ex: dataset.classes
    #         },
    #         str(path),
    #     )
    #     logger.info(f"LeaflictionCNN saved at {path}")

    # def load(self, path: str | Path, strict: bool = True) -> None:
    #     logger.info("Loading LeaflictionCNN")
    #     self.load_state_dict(torch.load(str(path), map_location="cpu"), strict=strict)

    # def load_model(path, model_ctor, device=None):
    #     """
    #     Recharge le modèle à partir d'un checkpoint.
    #     - model_ctor: un callable qui construit le modèle, ex: lambda n: TinyCNN(num_classes=n)
    #     - device: "cuda" ou "cpu" (détecté si None)
    #     Retourne: (model, class_names)
    #     """
    #     if device is None:
    #         device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    #     ckpt = torch.load(path, map_location=device)
    #     class_names = ckpt["class_names"]
    #     num_classes = len(class_names)

    #     model = model_ctor(num_classes).to(device)
    #     model.load_state_dict(ckpt["state_dict"], strict=True)
    #     print(f"Loaded model from {path} ({num_classes} classes)")
    #     return model, class_names


def train_one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    device = next(model.parameters()).device

    running_loss = 0.0
    running_acc = 0.0
    seen = 0

    for batch_idx, (X, y) in enumerate(tqdm(dataloader, total=len(dataloader)), start=1):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        # Log progress
        bsz = X.size(0)
        running_loss += loss.item() * bsz
        running_acc  += (logits.argmax(1) == y).float().sum().item()
        seen += bsz

        if batch_idx % 100 == 0 or batch_idx == len(dataloader):
            print(f"[{batch_idx:>4d}/{len(dataloader)}] "
                  f"loss: {running_loss/seen:.4f}  acc: {running_acc/seen:.4f}")



def evaluate(model, dataloader, loss_fn=None):
    """
    Evaluates the model on a dataloader.
    If loss_fn is provided, also returns/prints average loss.
    Returns: (avg_loss_or_None, avg_acc)
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)

            if loss_fn is not None:
                total_loss += loss_fn(logits, y).item() * X.size(0)

            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += X.size(0)

    avg_acc = total_correct / max(1, total_samples)
    avg_loss = total_loss / max(1, total_samples) if loss_fn is not None else None

    if loss_fn is not None:
        print(f"eval: loss={avg_loss:.4f}  acc={avg_acc:.4f}")
    else:
        print(f"eval: acc={avg_acc:.4f}")

    return avg_loss, avg_acc

    # def predict(self, data, classes):
    #     logger.info("Using LeaflictionCNN for prediction")
    #     self.eval()
    #     x, y = data[0][0], data[0][1]
    #     with torch.no_grad():
    #         x = x.to(self.device)
    #         pred = self(x)
    #         predicted, actual = classes[pred[0].argmax(0)], classes[y]
    #         logger.info(f'Predicted: "{predicted}", Actual: "{actual}"')



if __name__ == "__main__":
    import sys

    try:
        logger.info("Executing file 'model.py'")
        import sys
        import os

        sys.path.append(os.path.abspath(".."))

        from scripts.utils.data import LeaflictionData

        batch_size = 32

        data = LeaflictionData(
            original_dir="/Users/a.villa.massone/Code/42/42_Leaffliction/images/Apple",
            force_preproc=False,
            test_split=0.2,
            val_split=0.2,
            batch_size=batch_size,
        )
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        logger.info(f"Using {device} device")

        model = LeaflictionCNN(num_classes=4, device=device).to(device)
        logger.info(model)


        from tqdm import tqdm
        from torch import nn

        learning_rate = 1e-3
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Adam
        batch_size = 32
        epochs = 1

        print("Starting training of LeaflictionCNN")
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}\n-------------------------------")
            train_one_epoch(model=model, dataloader=data.loaders['train'], loss_fn=loss, optimizer=optimizer)
            evaluate(model=model, dataloader=data.loaders['train'], loss_fn=loss)
            evaluate(model=model, dataloader=data.loaders['val'], loss_fn=loss)
        print("Done!")

        # # save weights
        # torch.save({"model": model.state_dict(), "classes": full.classes}, "tinycnn.pt")
        # print("Saved tinycnn.pt")

        # model.save()
        # model.load()

    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)

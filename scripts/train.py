"""
train.py

# Subject
4 class image classification with data augmentation
## Params:
- directory containing dataset

## What it does:
- fetches images in its subdirectories. 
- increase/modify those images
- separate dataset in training and testing sets
- learn the characteristics of the diseases specified in the leaf. 
- save learnings and increased/modified dataset in a .zip

## Constraints:
- Test accuracy must be above 90%
- Test set should have min 100 imgs
- Demonstrated no data leakage or overfitting

## How to use:
$> ./train.[extension] ./Apple/
$> find . -maxdepth 2
./Apple
./Apple/apple_healthy
./Apple/apple_apple_scab
./Apple/apple_black_rot
./Apple/apple_cedar_apple_rust

## Distribution:
cls category	    count	train	test	val	    SUM
0	Grape_Esca	    1382	884	    221	    276	    1381
1	Grape_Black_rot	1178	753	    188	    235	    1176
2	Grape_spot	    1075	688	    172	    215	    1075
3	Grape_healthy   422	    270	    67	    84	    421
    SUM             4057	2595	648	    810

cls category	    count	train	test	val	    SUM
0	Apple_healthy	1640	1049	262	    328	    1639
1	Apple_scab	    629	    402	    100	    125	    627
2	Apple_Black_rot	620	    396	    99	    124	    619
3	Apple_rust	    275	    176	    44	    55	    275
    SUM             3164	2023	505	    632

- load data
    - from subdirectories
    - train test val
- preproc
    - if data exist and argparse force-preproc false, ignore step
    - clean / augment / normalize ?
    - save .zip
- model + compile + fit
- train
    - save .zip
- eval
- monitor experiment w/ wandb
"""

import torch
from torch import nn
import argparse
import sys
from scripts.utils.logger import get_logger
from scripts.utils.data import LeaflictionData
from scripts.utils.model import LeaflictionCNN

logger = get_logger(__name__)

# def fit(dataloader, model, loss_fn, optimizer, device):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if batch % 100 == 0:
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# def eval(dataloader, model, loss_fn, device):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# def predict(model, data, device, classes):
#     model.eval()
#     x, y = data[0][0], data[0][1]
#     with torch.no_grad():
#         x = x.to(device)
#         pred = model(x)
#         predicted, actual = classes[pred[0].argmax(0)], classes[y]
#         print(f'Predicted: "{predicted}", Actual: "{actual}"')



def parse_args() -> argparse.Namespace:
    """Uses `argparse` to handle the arguments : image_path, output_dir"""

    parser = argparse.ArgumentParser(
        description="Leaf Disease Classification - training program"
    )

    parser.add_argument(
        "data_dir", type=str, help="Path to the original image dataset root directory"
    )
    parser.add_argument(
        "--force-preproc",
        action="store_true",
        default=False,
        help="Re-execute preproc even if augmented dataset already exists",
    )

    # for predict - parser.add_argument("--image_path", type=str, help="Path to image for prediction")
    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--img_size", type=int, default=256)
    # parser.add_argument("--model_path", type=str, default="model.pt")

    return parser.parse_args()
# - will use experiment tracking with wandb


def main():
    try:
        logger.info("Starting train.py program")

        args = parse_args()
        print(args)

        batch_size = 32
        data = LeaflictionData(
            original_dir=args.data_dir,
            force_preproc=args.force_preproc,
            test_split=0.2,
            val_split=0.2,
            batch_size=batch_size,
            allowed_dir="images",
        )

        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {device} device")

        model = LeaflictionCNN(num_classes=4, device=device).to(device)
        print(model)


        loss_fn = nn.CrossEntropyLoss()
        learning_rate = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        model.compile(loss=loss_fn, optimizer=optimizer, lr=learning_rate)

        epochs = 1
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            model.fit(data.loaders['train'])
            model.eval(data.loaders['train'])
            model.eval(data.loaders['val'])
        print("Done!")

        model.eval(data.loaders['test'])

        # todo
        # - training
        #   - loss train, loss val, acc train, acc val
        #   - laerning curves
        #   - experiment tracking wandb
        #   - hyperparam optimization
        #   - documentation
        #   - add fit, eval, predict to model class
        # - save zip model and augmented data

        return

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        logger.warning("Program interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

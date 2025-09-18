"""
train.py

# Subject
4 class image classification with data augmentation and transformation

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

## Distribution and metrics
- A folder will be created per experiment with learning curves, metrics reports and cms
- Distribution of the data can be found in the folder : plots/
"""

import argparse
import sys
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from scripts.utils.data import LeaflictionData
from scripts.utils.logger import get_logger
from scripts.utils.model import LeaflictionCNN
from scripts.utils.train_plots import (
    export_classification_reports,
    export_confusion_matrices,
    export_model_architecture,
    plot_learning_curves,
    save_history_csv,
)

logger = get_logger(__name__)


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
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    return parser.parse_args()


def make_train_fn():
    counter = count(1)

    def train_one_epoch(model, dataloader, loss_fn, optimizer):
        """Training loop on one epoch, from dataloader"""
        model.train()
        device = next(model.parameters()).device
        epoch = next(counter)

        tqdm_desc = f"[Epoch {epoch:02d}] train [{len(dataloader):>3} batches | {len(dataloader.dataset):>5} imgs]"

        pbar = tqdm(dataloader, total=len(dataloader), desc=tqdm_desc)
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            # Forward
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            # Backward + update
            loss.backward()
            optimizer.step()

    return train_one_epoch


def evaluate(model, dataloader, split, loss_fn=None) -> dict[str, Any]:
    """
    Model evaluation on dataloader
    Returns dict with accuracy, loss (if provided), y_true, y_pred.
    """
    model.eval()
    device = next(model.parameters()).device
    y_true, y_pred = [], []
    total_loss = 0.0
    correct, total = 0, 0
    tqdm_desc = f"Evaluation {split:>5} [{len(dataloader):>3} batches | {len(dataloader.dataset):>5} imgs]"

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), desc=tqdm_desc)
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = logits.argmax(1)

            y_true.extend(y.tolist())
            y_pred.extend(preds.cpu().tolist())

            if loss_fn is not None:
                total_loss += loss_fn(logits, y).item() * y.size(0)

            correct += (preds == y).sum().item()
            total += y.size(0)
            accuracy = correct / total
            avg_loss = total_loss / total

            postfix = {"accuracy": f"{accuracy:.4f}"}
            if loss_fn is not None:
                postfix["loss"] = f"{avg_loss:.4f}"
            pbar.set_postfix(postfix)
        
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def create_archive_zip(root_dir, exp_dir, augmented_data_dir):
    import zipfile

    zip_path = root_dir / f"{exp_dir.name}_package.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        folder_to_zip = [exp_dir, augmented_data_dir]
        for folder in folder_to_zip:
            for file in folder.rglob("*"):
                zipf.write(file, arcname=file.relative_to(root_dir))

    logger.info(f"Created archive at: \n{zip_path}")



def main():
    try:
        logger.info("Starting train.py program")
        ####### ARGUMENTS #############
        args = parse_args()
        logger.info(args)

        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        epochs = args.epochs
        batch_size = args.batch_size
        patience = args.patience

        original_dir = Path(args.data_dir).resolve()
        if not original_dir.is_dir():
            print(f"Error: '{original_dir}' is not a valid directory.")
            sys.exit(1)

        root_dir = original_dir.parent.parent  # 42Leafliction/images/Apple
        experiment_name = "LeaflictionCNN_" + datetime.now().strftime("%Y-%m-%d_%H-%M")
        exp_dir = root_dir / "experiments" / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Project dir : {root_dir}")
        logger.info(f"Data dir : {original_dir}")
        logger.info(f"Experiment dir : {exp_dir}")

        ####### LOAD DATA #############
        data = LeaflictionData(
            original_dir=original_dir,  # update link
            force_preproc=False,
            test_split=0.2,
            val_split=0.2,
            batch_size=batch_size,
        )

        logger.info(f"Augmented data dir : {data.augmented_dir}")

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f"Using {device} device")

        model = LeaflictionCNN(num_classes=4).to(device)

        ####### TRAINING #############
        logger.info("Starting training")

        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        train_one_epoch = make_train_fn()

        best_acc, best_epoch, bad = 0.0, 0, 0
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            # training and evaluations with logs
            train_one_epoch(
                model=model,
                dataloader=data.loaders["train"],
                loss_fn=loss,
                optimizer=optimizer,
            )
            train_results = evaluate(
                model=model,
                dataloader=data.loaders["train"],
                split="train",
                loss_fn=loss,
            )
            val_results = evaluate(
                model=model, dataloader=data.loaders["val"], split="val", loss_fn=loss
            )

            # record evaluation history
            history["train_loss"].append(train_results["loss"])
            history["train_acc"].append(train_results["accuracy"])
            history["val_loss"].append(val_results["loss"])
            history["val_acc"].append(val_results["accuracy"])

            # early stopping with patience - restaures best weights
            if val_results["accuracy"] > best_acc:
                best_acc = val_results["accuracy"]
                best_epoch = epoch
                torch.save(
                    model.state_dict(), exp_dir / f"best_model_epoch_{best_epoch}.pt"
                )
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    logger.info(
                        f"Early stopping. Restoring state from epoch {best_epoch}"
                    )
                    model.load_state_dict(
                        torch.load(
                            exp_dir / f"best_model_epoch_{best_epoch}.pt",
                            map_location=device,
                        )
                    )
                    break

        ####### EVALUATION #############
        test_results = evaluate(
            model=model, dataloader=data.loaders["test"], split="test"
        )

        ####### LOG & SAVE #############
        class_names = list(data.class_to_idx.keys())

        export_model_architecture(
            model=model, dataloader=data.loaders["train"], out_dir=exp_dir
        )
        plot_learning_curves(
            train_loss=history["train_loss"],
            train_acc=history["train_acc"],
            val_loss=history["val_loss"],
            val_acc=history["val_acc"],
            out_dir=exp_dir,
        )
        save_history_csv(history=history, out_dir=exp_dir)
        export_confusion_matrices(
            train_results=train_results,
            val_results=val_results,
            test_results=test_results,
            out_dir=exp_dir,
            class_names=class_names,
        )
        export_classification_reports(
            train_results=train_results,
            val_results=val_results,
            test_results=test_results,
            out_dir=exp_dir,
            class_names=class_names,
        )

        # LOG experiment data
        experiment = {}
        experiment["out_dir"] = exp_dir
        experiment["accuracy"] = test_results["accuracy"]
        experiment["class_names"] = class_names
        experiment["labels"] = list(data.class_to_idx.values())
        experiment["train_size"] = len(data.loaders["train"].dataset)
        experiment["val_size"] = len(data.loaders["val"].dataset)
        experiment["test_size"] = len(data.loaders["test"].dataset)
        experiment["batch_size"] = data.loaders["train"].batch_size
        experiment["train_batches"] = len(data.loaders["train"])
        experiment["val_batches"] = len(data.loaders["val"])
        experiment["test_batches"] = len(data.loaders["test"])
        experiment["learning_rate"] = learning_rate
        experiment["weight_decay"] = weight_decay
        experiment["epochs"] = epochs
        experiment["batch_size"] = batch_size
        experiment["patience"] = patience
        experiment["device"] = device
        experiment["loss"] = str(loss)
        experiment["optimizer"] = str(optimizer)
        experiment["best_epoch"] = best_epoch

        Path(exp_dir).mkdir(parents=True, exist_ok=True)

        out_file = Path(exp_dir) / "experiment.txt"
        with out_file.open("w") as f:
            f.write("===== Experiment parameters =====\n")
            for k, v in experiment.items():
                f.write(f"{k:<15}:{v}\n")
            f.write("\n\n")
        logger.info(f"Experiment params saved to :\n-{out_file}")

        # rename experiment folder with accuracy
        new_exp_dir = exp_dir.with_name(
            f"{exp_dir.name}_acc_{test_results['accuracy']:.2f}"
        )
        exp_dir.rename(new_exp_dir)

        # create ZIP
        create_archive_zip(
            root_dir=root_dir, 
            exp_dir=exp_dir, 
            augmented_data_dir=data.augmented_dir
            )


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

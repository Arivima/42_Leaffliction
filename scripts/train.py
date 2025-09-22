"""
train.py

# Subject
4 class image classification with data augmentation and transformation

## Params:
- directory containing dataset

## What it does:
- fetches images in its subdirectories.
- increase/modify those images
- separate dataset in training, validation and testing sets
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
import zipfile
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any

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


class LeaflictionExperiment:
    """
    LeaflictionExperiment is class that runs and tracks a training experiment
    It will load a LeaflictionData data class that preprocess and augment the raw data
    It will load and train a LeaflictionCNN model on train/val sets
    It will evaluate the model on the test set and export performance metrics,
      classification reports, confusion matrices and learning curves
    It will create a zip archive with the best weights, the experiment data and the augmented dataset
    It has the following public methods:
        - load_data
        - load_model
        - train
        - evaluate
        - track_experiment
        - create_archive_zip
    """

    def __init__(self):
        """Class initialization with user defined args"""

        def parse_args() -> argparse.Namespace:
            """Uses `argparse` to handle the arguments : image_path, output_dir"""
            parser = argparse.ArgumentParser(
                description="Leaf Disease Classification - training program"
            )
            parser.add_argument(
                "data_dir",
                type=str,
                help="Path to the original image dataset root directory",
            )
            parser.add_argument(
                "--force-preproc",
                action="store_true",
                default=False,
                help="Re-execute preproc even if augmented dataset already exists",
            )
            parser.add_argument("--batch_size", type=int, default=64)
            parser.add_argument("--epochs", type=int, default=50)
            parser.add_argument("--patience", type=int, default=7)
            parser.add_argument("--learning_rate", type=float, default=1e-3)
            parser.add_argument("--weight_decay", type=float, default=1e-4)
            parser.add_argument("--test_split", type=float, default=0.2)
            parser.add_argument("--val_split", type=float, default=0.2)
            return parser.parse_args()

        logger.info("Starting train.py program")

        args = parse_args()
        logger.info(args)

        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.patience = args.patience
        self.force_preproc = args.force_preproc
        self.test_split = args.test_split
        self.val_split = args.val_split

        self.original_dir = Path(args.data_dir).resolve()
        if not self.original_dir.is_dir():
            print(f"Error: '{self.original_dir}' is not a valid directory.")
            sys.exit(1)

        # create Experiment folder
        self.root_dir = self.original_dir.parent.parent  # 42Leafliction/images/Apple
        self.experiment_name = (
            "LeaflictionCNN_"
            + self.original_dir.name
            + "_"
            + datetime.now().strftime("%Y-%m-%d_%H-%M")
        )
        self.exp_dir = self.root_dir / "experiments" / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Project dir : {self.root_dir}")
        logger.info(f"Data dir : {self.original_dir}")
        logger.info(f"Experiment dir : {self.exp_dir}")

    def load_data(self):
        """
        loads a LeaflictionData custom data class
        LeaflictionData includes datasets, dataloaders, path to datasets and class/idx mappings
        """
        self.data = LeaflictionData(
            original_dir=self.original_dir,
            force_preproc=self.force_preproc,
            test_split=self.test_split,
            val_split=self.val_split,
            batch_size=self.batch_size,
        )
        logger.info(
            f"Augmented data dir : {self.data.augmented_dir}"
        )  # 42Leafliction/images/Apple
        logger.info("Dataset distribution and plots exported to 'plots/'")
        logger.info("Dataset augmentation sample exported to 'sample/'")

    def load_model(self):
        """loads a LeaflictionCNN custom class model"""
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f"Using {self.device} device")

        self.model = LeaflictionCNN(num_classes=4).to(self.device)

    def train(self):
        """
        training function
        defines loss and optimizer
        trains model with a dataloader, evaluates training and validation sets
        records history for learning curves
        includes early stopping with user sedfined patience
        """

        def make_train_fn():
            """returns a function with an epoch counter"""
            counter = count(1)

            def train_one_epoch(model, dataloader, loss_fn, optimizer):
                """Training loop on one epoch with dataloader"""
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

        logger.info("Starting training")

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        train_one_epoch = make_train_fn()

        best_acc, self.best_epoch, bad = 0.0, 0, 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(1, self.epochs + 1):
            # training and evaluations with logs
            train_one_epoch(
                model=self.model,
                dataloader=self.data.loaders["train"],
                loss_fn=self.loss,
                optimizer=self.optimizer,
            )
            self.train_results = self._evaluate_model(
                model=self.model,
                dataloader=self.data.loaders["train"],
                split="train",
                loss_fn=self.loss,
            )
            self.val_results = self._evaluate_model(
                model=self.model,
                dataloader=self.data.loaders["val"],
                split="val",
                loss_fn=self.loss,
            )

            # record evaluation history
            self.history["train_loss"].append(self.train_results["loss"])
            self.history["train_acc"].append(self.train_results["accuracy"])
            self.history["val_loss"].append(self.val_results["loss"])
            self.history["val_acc"].append(self.val_results["accuracy"])

            # early stopping with patience - restaures best weights
            if self.val_results["accuracy"] > best_acc:
                best_acc = self.val_results["accuracy"]
                self.best_epoch = epoch
                torch.save(
                    self.model.state_dict(),
                    self.exp_dir / f"best_model_epoch_{self.best_epoch}.pt",
                )
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    logger.info(
                        f"Early stopping. Restoring state from epoch {self.best_epoch}"
                    )
                    self.model.load_state_dict(
                        torch.load(
                            self.exp_dir / f"best_model_epoch_{self.best_epoch}.pt",
                            map_location=self.device,
                        )
                    )
                    break

    @staticmethod
    def _evaluate_model(model, dataloader, split, loss_fn=None) -> dict[str, Any]:
        """
        Model evaluation using dataloader
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

    def evaluate(self):
        self.test_results = self._evaluate_model(
            model=self.model, dataloader=self.data.loaders["test"], split="test"
        )

    def track_experiment(self):
        self.class_names = list(self.data.class_to_idx.keys())

        X, _ = next(iter(self.data.loaders["train"]))
        input_size = tuple(X.shape)

        export_model_architecture(
            model=self.model, input_size=input_size, out_dir=self.exp_dir
        )
        plot_learning_curves(
            train_loss=self.history["train_loss"],
            train_acc=self.history["train_acc"],
            val_loss=self.history["val_loss"],
            val_acc=self.history["val_acc"],
            out_dir=self.exp_dir,
        )
        save_history_csv(history=self.history, out_dir=self.exp_dir)
        export_confusion_matrices(
            train_results=self.train_results,
            val_results=self.val_results,
            test_results=self.test_results,
            out_dir=self.exp_dir,
            class_names=self.class_names,
        )
        export_classification_reports(
            train_results=self.train_results,
            val_results=self.val_results,
            test_results=self.test_results,
            out_dir=self.exp_dir,
            class_names=self.class_names,
        )
        self._log_experiment_data()

        # rename experiment folder with accuracy
        new_exp_dir = self.exp_dir.with_name(
            f"{self.exp_dir.name}_acc_{self.test_results['accuracy']:.2f}"
        )
        self.exp_dir.rename(new_exp_dir)

    def _log_experiment_data(self):
        """exports experiment params and metrics to experiment folder"""
        experiment = {}
        experiment["out_dir"] = self.exp_dir
        experiment["accuracy"] = self.test_results["accuracy"]
        experiment["class_names"] = self.class_names
        experiment["labels"] = list(self.data.class_to_idx.values())
        experiment["train_size"] = len(self.data.loaders["train"].dataset)
        experiment["val_size"] = len(self.data.loaders["val"].dataset)
        experiment["test_size"] = len(self.data.loaders["test"].dataset)
        experiment["batch_size"] = self.data.loaders["train"].batch_size
        experiment["train_batches"] = len(self.data.loaders["train"])
        experiment["val_batches"] = len(self.data.loaders["val"])
        experiment["test_batches"] = len(self.data.loaders["test"])
        experiment["learning_rate"] = self.learning_rate
        experiment["weight_decay"] = self.weight_decay
        experiment["epochs"] = self.epochs
        experiment["batch_size"] = self.batch_size
        experiment["patience"] = self.patience
        experiment["device"] = self.device
        experiment["loss"] = str(self.loss)
        experiment["optimizer"] = str(self.optimizer)
        experiment["best_epoch"] = self.best_epoch

        Path(self.exp_dir).mkdir(parents=True, exist_ok=True)

        out_file = Path(self.exp_dir) / "experiment.txt"
        with out_file.open("w") as f:
            f.write("===== Experiment parameters =====\n")
            for k, v in experiment.items():
                f.write(f"{k:<15}:{v}\n")
            f.write("\n\n")
        logger.info(f"Experiment params saved to :\n-{out_file}")

    def create_archive_zip(self):
        """Creates a zip archive containing model, augmented dataset and experiment data"""
        zip_path = self.root_dir / f"{self.exp_dir.name}_package.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            folder_to_zip = [self.exp_dir, self.data.augmented_dir]
            for folder in folder_to_zip:
                for file in folder.rglob("*"):
                    zipf.write(file, arcname=file.relative_to(self.root_dir))

        logger.info(f"Created archive at: \n{zip_path}")


def main():
    start = datetime.now()
    try:
        experiment = LeaflictionExperiment()
        experiment.load_data()
        experiment.load_model()
        experiment.train()
        experiment.evaluate()
        experiment.track_experiment()
        experiment.create_archive_zip()
        end = datetime.now()
        elapsed = end - start
        minutes, seconds = divmod(elapsed.total_seconds(), 60)
        logger.info(f"Training time = {int(minutes)}m {int(seconds)}s")
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

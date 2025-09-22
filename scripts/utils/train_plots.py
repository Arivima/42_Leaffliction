"""
train_plots.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torchinfo import summary
from scripts.utils.logger import get_logger

logger = get_logger(__name__)


def plot_learning_curves(
    train_loss, val_loss, train_acc, val_acc, out_dir, filename="learning_curves.png"
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss")
    ax1.plot(epochs, val_loss, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Learning Curve - Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Accuracy
    ax2.plot(epochs, train_acc, label="Train Acc")
    ax2.plot(epochs, val_acc, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Learning Curve - Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    # plot
    plt.tight_layout()
    out_path = Path(out_dir) / filename
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info(f"Saved learning curves plots to: \n{out_path}")


def save_history_csv(history, out_dir):
    """save training history as csv"""
    filename = "learning_history.csv"
    out_path = Path(out_dir) / Path(filename)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(history["train_loss"])):
            writer.writerow(
                [
                    i + 1,
                    history["train_loss"][i],
                    history["train_acc"][i],
                    history["val_loss"][i],
                    history["val_acc"][i],
                ]
            )
    logger.info(f"Saved history to: \n{out_path}")


def export_classification_reports(
    out_dir,
    class_names,
    train_results=None,
    val_results=None,
    test_results=None,
    base_filename="classification_report",
):
    """
    Exports classification reports for train val and test
    train_results / val_results / test_results : dict avec 'y_true' et 'y_pred'
    """

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    out_txt = Path(out_dir) / f"{base_filename}.txt"
    out_csv = Path(out_dir) / f"{base_filename}.csv"

    if out_txt.exists():
        out_txt.unlink()
    if out_csv.exists():
        out_csv.unlink()

    def _process_split(results, split):
        """Append exportation for one split"""
        if results is None:
            return
        y_true = np.asarray(results["y_true"])
        y_pred = np.asarray(results["y_pred"])
        labels = np.unique(y_true)

        # text
        cr_txt = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            digits=3,
            zero_division=0,
        )

        if out_csv.exists():
            out_csv.unlink()
        with out_txt.open("a") as f:
            f.write(f"===== {split.upper()} =====\n")
            f.write(cr_txt + "\n\n")

        # csv
        cr_dict = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            digits=3,
            zero_division=0,
            output_dict=True,
        )
        df = pd.DataFrame(cr_dict).T
        idx_map = {str(lbl): name for lbl, name in zip(labels, class_names)}
        df = df.rename(index=idx_map)
        df["split"] = split

        if out_csv.exists():
            df.to_csv(out_csv, mode="a", header=False)
        else:
            df.to_csv(out_csv, index=True)

        logger.info(
            f"Saved classification report for '{split}' to:\n{out_txt}\n{out_csv}"
        )

    for split, results in {
        "train": train_results,
        "val": val_results,
        "test": test_results,
    }.items():
        _process_split(results, split)


def export_confusion_matrices(
    train_results=None,
    val_results=None,
    test_results=None,
    out_dir=".",
    class_names=None,
    filename="confusion_matrices.png",
):
    """Export confusion matrices in a png"""

    def _plot_confusion_matrix(ax, y_true, y_pred, labels, class_names, title):
        """Helper pour tracer une matrice de confusion sur un axe donné."""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax)
        ax.set_title(title)

    available = {
        "Train": train_results,
        "Validation": val_results,
        "Test": test_results,
    }
    splits = {k: v for k, v in available.items() if v is not None}

    if not splits:
        raise ValueError("No split provided")

    labels = range(len(class_names))

    # Plot
    if len(splits) == 3:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        fig.delaxes(axes[-1])

    else:
        fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 5))
        axes = np.atleast_1d(axes)

    mapping = dict(zip(splits.keys(), axes))

    for split, results in splits.items():
        y_true = np.asarray(results["y_true"])
        y_pred = np.asarray(results["y_pred"])
        _plot_confusion_matrix(
            mapping[split], y_true, y_pred, labels, class_names, split
        )
    for ax in mapping.values():
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    fig.suptitle("Confusion Matrices", y=0.98)
    fig.tight_layout()

    # Save
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved confusion matrices to:\n{out_path}")


def export_model_architecture(
    model, input_size, out_dir, filename="model_architecture.txt"
):
    """
    Save model architecture and a detailed summary into a text file.
    """

    device = next(model.parameters()).device

    plain_arch = str(model)
    model_summary = summary(model, input_size=input_size, device=device, verbose=0)

    text = (
        "===== MODEL ARCHITECTURE (str(model)) =====\n"
        f"{plain_arch}\n\n"
        "===== DETAILED SUMMARY (torchinfo) =====\n"
        f"{model_summary}\n"
    )

    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(text)

    logger.info(f"Model architecture + summary saved to :\n-{out_path}")

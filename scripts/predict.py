"""
predict.py

!Requires that a model be saved in "42_Leaffliction/best_model"

- Takes as arg the path to the image on which to run prediction
- Loads the trained model from 42_Leaffliction/best_model

If given a file path:
- displays original image and transformed image side by side
- displays the predicted category

If given a folder path:
- Loads the test dataset in that folder
- generates predictions for the whole test folder
- exports a .csv of predictions with image paths
- exports a classification report and a confusion matrix

Usage:
$> ./predict.py ./images_augmented/Apple_test/apple_healthy/image (9).JPG
$> ./predict.py ./images_augmented/Apple_test
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from scripts.train import LeaflictionExperiment
from scripts.utils.data import LeaflictionData
from scripts.utils.logger import get_logger
from scripts.utils.model import LeaflictionCNN
from scripts.utils.train_plots import (
    export_classification_reports,
    export_confusion_matrices,
)

logger = get_logger(__name__)


def get_model_path(best_model_dir: Path) -> Path:
    """retrieves the path of the best pre-trained model"""
    # get content of best_model directory
    models = list(best_model_dir.glob("*.pt"))
    if not models:
        raise FileNotFoundError(f"No model found in {best_model_dir}")
    # take the most recent if several models
    best_model_path = sorted(models)[-1]
    return best_model_path


def load_model(best_model_path: Path, num_classes: int, device: str) -> torch.nn.Module:
    """Load trained model weights from the best_model directory."""
    # instanciate an empty model
    model = LeaflictionCNN(num_classes=num_classes).to(device)
    # loads the pre-trained model
    model.load(path=best_model_path, device=device)
    return model


def preprocess_image(image_path: Path) -> torch.Tensor:
    """Apply same preprocessing as training."""
    preproc_pipeline = LeaflictionData.get_preproc_pipeline()
    image = Image.open(image_path).convert("RGB")
    return preproc_pipeline(image).unsqueeze(0)  # add batch dimension


def predict(model, device, image_paths, idx_to_class) -> list[dict]:
    """Run inference on a list of images and return predictions."""
    model.eval()
    results = []
    with torch.no_grad():
        for img_path in image_paths:
            tensor = preprocess_image(img_path).to(device)
            logits = model(tensor)
            pred_idx = logits.argmax(1).item()
            pred_class = idx_to_class[pred_idx]
            results.append({"image": str(img_path), "pred_class": pred_class})
            logger.info(f"{img_path.name:30} -> {pred_class}")
    return results


def main():

    # arguments
    parser = argparse.ArgumentParser(description="Leafliction prediction script")
    parser.add_argument(
        "path",  # ./images/Apple"
        type=str,
        help="Path to an image or a folder containing images for prediction",
    )
    args = parser.parse_args()
    print(args)

    # initialization
    image_path = Path(args.path).resolve()
    if not image_path.exists():
        print(f"Image file/folder'{image_path}' not found")
        sys.exit(1)

    project_root = LeaflictionData.find_project_root(p=image_path)
    logger.info(f"project_root {project_root}")

    best_model_dir = project_root / "best_model"
    if not best_model_dir.exists():
        print(f"Best model dir '{best_model_dir}' not found")
        sys.exit(1)

    data = LeaflictionData(original_dir=image_path)
    idx_to_class = data.idx_to_class
    num_classes = len(idx_to_class)
    logger.info(f"num_classes {num_classes}")
    logger.info(f"idx_to_class {idx_to_class}")

    # Device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using {device} device")

    # Load model
    best_model_path = get_model_path(best_model_dir)
    model = load_model(best_model_path, num_classes=num_classes, device=device)
    logger.info(f"Loaded model from {best_model_path}")

    # Predict
    if image_path.is_file():
        logger.info(f"Running predict on a single image {image_path}")

        image_paths = [image_path]
        test_results = predict(model, device, image_paths, idx_to_class)

        # show the diff before after

    elif image_path.is_dir():
        logger.info(f"Running predict on the test set in the directory {image_path}")

        # predict
        test_results = LeaflictionExperiment._evaluate_model(
            model=model, dataloader=data.loaders["test"], split="test"
        )

        # Examples :
        # best_model_dir    /Users/a.villa.massone/Code/42/42_Leaffliction/best_model/
        # best_model_path   /Users/a.villa.massone/Code/42/42_Leaffliction/best_model/best_model_epoch_1.pt
        # csv               /Users/a.villa.massone/Code/42/42_Leaffliction/best_model/best_model_epoch_1_predictions.csv
        # cm                /Users/a.villa.massone/Code/42/42_Leaffliction/best_model/best_model_epoch_1_confusion_matrices.png
        # cr                /Users/a.villa.massone/Code/42/42_Leaffliction/best_model/best_model_epoch_1_classification_report

        out_dir = best_model_dir
        csv_filename = str(best_model_path.with_suffix("").name) + "_predictions.csv"
        cm_filename = (
            str(best_model_path.with_suffix("").name) + "_confusion_matrices.png"
        )
        cr_filename = (
            str(best_model_path.with_suffix("").name) + "_classification_report"
        )

        logger.info(f"best_model_dir {best_model_dir}")
        logger.info(f"best_model_path {best_model_path}")
        logger.info(f"out_dir {out_dir}")
        logger.info(f"csv_filename {csv_filename}")
        logger.info(f"cm_filename {cm_filename}")
        logger.info(f"cr_filename {cr_filename}")

        # Save CSV
        df = pd.DataFrame(test_results)
        df = df[["y_true", "y_pred"]]
        df["image"] = [i[0] for i in data.datasets["test"].imgs]
        csv_path = out_dir / csv_filename
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to :\n{csv_path}")

        # Export performance metrics
        if image_path.is_dir():
            export_confusion_matrices(
                test_results=test_results,
                out_dir=best_model_path.parent,
                class_names=list(idx_to_class.values()),
                filename=cm_filename,
            )
            export_classification_reports(
                test_results=test_results,
                out_dir=best_model_path.parent,
                class_names=list(idx_to_class.values()),
                base_filename=cr_filename,
            )

    else:
        print(f"Invalid input path: {image_path}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Program interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)

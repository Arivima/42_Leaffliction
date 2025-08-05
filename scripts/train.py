"""
In this part, you must write a program named train.[extension] that takes as parameter
a directory and fetches images in its subdirectories. It must then increase/modify those
images in order to learn the characteristics of the diseases specified in the leaf. Those
learnings must be saved and returned in a .zip that also includes your increased/modified
images.
$> ./train.[extension] ./Apple/
$> find . -maxdepth 2
./Apple
./Apple/apple_healthy
./Apple/apple_apple_scab
./Apple/apple_black_rot
./Apple/apple_cedar_apple_rust

You have to separate your data set in two parts, one for Training and one for Validation.
The predictions of your validation set must also have an accuracy above 90% (you
must be able to prove it in an evaluation, with a minimum of 100 images in the validation
set).
Make sure you don't have any knowledge about the validation set or on overfitting
prior to the assessment, so your results don't look suspicious.

classification - 4 classes
        category	    count	train	test	val	    sum
0	Grape_Esca	    1382	884	    221	    276	    1381
1	Grape_Black_rot	1178	753	    188	    235	    1176
2	Grape_spot	    1075	688	    172	    215	    1075
3	Grape_healthy   422	    270	    67	    84	    421
    sum             4057	2595	648	    810

        category	    count	train	test	val	    sum
0	Apple_healthy	1640	1049	262	    328	    1639
1	Apple_scab	    629	    402	    100	    125	    627
2	Apple_Black_rot	620	    396	    99	    124	    619
3	Apple_rust	    275	    176	    44	    55	    275
    sum             3164	2023	505	    632

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

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from utils.logger import get_logger
from Augmentation import (open_image, augment_image, save_images)

logger = get_logger(__name__)

# To include and delete
# AUGMENTED_DIRECTORY = f"{original_dir}_augmented"

# - will use experiment tracking with wandb
# - will use technique for unbalanced dataset

# https://docs.pytorch.org/docs/stable/data.html


    # during transformation
    # splits into val and train 
    # saves into folders


def is_safe_path(base_dir: str, path: str) -> bool:
    real_base = os.path.abspath(base_dir)
    real_path = os.path.abspath(path)
    return real_path.startswith(real_base)



def get_image_folder_dataset(data_dir: str) -> ImageFolder:
    logger.info(f"Processing dataset at '{data_dir}'")

    if not is_safe_path("images", data_dir):
        raise ValueError(
            f"'{data_dir}' is outside allowed directory. Allowed directory : 'images*/'"
        )

    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"'{data_dir}' is not a valid directory.")

    # transforms : used to preprocess and augment data
    # transforms.Compose : chains several transforms into a preproc pipeline
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)), # enforced in case of surprise even though all images are that size
            transforms.ToTensor(), # PIL.Image.Image to torch.Tensor
            # ON PEUT RAJOUTER STANDARDISATION ?
        ]
    )

    # ImageFolder : lazy-loads dataset from folder (specific structure)
    #      ImageFolder.__init__() : lists the content of folder and checks for validity
    #            self.samples = make_dataset(directory, class_to_idx, extensions, is_valid_file)
    #      dataset = ImageFolder("data/images") -> lists files and collects labels
    #      print(dataset.samples[0]) -> access metadata
    # the image is loaded only when you access dataset[i] or iterate over a DataLoader
    #      under the hood : __getitem__ is the actual function that loads the file
    #      img, label = dataset[0] -> image is loaded
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=train_transform,
        # target_transform=,
        # is_valid_file default implementation checks for extension validity
    )
    logger.info("Successfully loaded dataset")
    logger.info(f"{len(dataset)} classes")
    logger.info(f"type {type(dataset)}")
    logger.info(f"sample dataset {dataset.samples[0]}") # metadata (filepath, class_idx)
    logger.info(f"extract dataset {dataset[0]}") # (transformed_image_tensor, class_idx)

    return dataset

# SAUVEGARDER EN DUR COMME PARTIE AUGMENTATION
def train_test_split(
        dataset : ImageFolder,
        test_split: int = 0.2,
        val_split: int = 0.2
        ) -> ImageFolder:
    
    if test_split + val_split >= 0.9:
        raise ValueError("test_split + val_split must be less than 0.9")

    # train test split
    test_size = int(test_split * len(dataset))
    val_size = int(val_split * (len(dataset) - test_size))
    train_size = len(dataset) - test_size - val_size

    logger.info(
        f"test_size {test_size} val_size {val_size} train_size {train_size} "
        f"total {len(dataset)} total {test_size + val_size + train_size}"
    )

    # random_split splits into subsets of dataset
    # issue : not stratified
    train_dataset, test_dataset, val_dataset = random_split(
        dataset, [train_size, test_size, val_size]
    )
    logger.info(
        f"test_size {len(test_dataset)} "
        f"val_size {len(val_dataset)} "
        f"train_size {len(train_dataset)}"
    )
    return {
        'train_dataset':train_dataset, 
        'test_dataset':test_dataset, 
        'val_dataset':val_dataset,
        }


def set_metadata(dataset: ImageFolder):
    # setting mapping, class counts
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    class_count_idx = dict(Counter(dataset.targets))
    class_count_name = {
        idx_to_class[idx]: count for idx, count in class_count_idx.items()
    }

    logger.info(f"class_to_idx {class_to_idx}")
    logger.info(f"idx_to_class {idx_to_class}")
    logger.info(f"class_count_idx {class_count_idx}")
    logger.info(f"class_count_name {class_count_name}")

    return {
        'class_to_idx':class_to_idx,
        'idx_to_class':idx_to_class,
        'class_count_idx':class_count_idx,
        'class_count_name':class_count_name,
    }



def get_data_loader(
    datasets: dict[ImageFolder],
    batch_size:int = 32,
) -> pd.DataFrame:

    # sampler : used during training to apply a specific sampling strategy
    # WeightedRandomSampler handles class imbalance
    # per-sample weights because WeightedRandomSampler samples individual data points, not classes
    train_labels = datasets.train_dataset.targets
    train_class_counts = Counter(train_labels)

    train_weights = 1.0 / torch.tensor(
        [train_class_counts[label] for label in train_labels], dtype=torch.float
    )
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    # DataLoader : dataset iterable loader/collator for in-training batch processing
    train_loader = DataLoader(datasets.train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(datasets.test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(datasets.val_dataset, batch_size=batch_size, shuffle=False)


    return {
        'train_loader':train_loader,
        'test_loader':test_loader,
        'val_loader':val_loader
        }




def augment_data(image_paths: str) -> None:
    """
    generates 6 types of data augmentation per image provided. saved locally.
    Takes a list of image paths
    """
    for image in image_paths:
        image = open_image(image_path=image_path)
        augmented = augment_image(original_image=image)
        save_images(original_image_path=image_path, images=augmented)


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
        help="Re-execute preproc even if augmented dataset already exists",
    )

    # for predict - parser.add_argument("--image_path", type=str, help="Path to image for prediction")
    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--img_size", type=int, default=256)
    # parser.add_argument("--model_path", type=str, default="model.pt")

    return parser.parse_args()


def main():
    try:
        logger.info("Starting train.py program")

        args = parse_args()
        original_data_dir : str = args.data_dir

        # add a condition to do these steps only if the folder does not exist and there is no force preproc flag
        # create a copy of original dataset
        augmented_data_dir : str = original_data_dir + '_augmented'
        os.makedirs(augmented_data_dir, exist_ok=True)

        # load original dataset
        original_dataset : ImageFolder = get_image_folder_dataset(augmented_data_dir)

        # fait le calcul - stratifié
        # créé de nouveaux dir _train _test _val + copie images
        # re-load les datasets et les retourne
        datasets : dict[ImageFolder] = train_test_split(augmented_data_dir, original_dataset)

        # create and save augmented images for training set
        image_paths = [datasets.train_dataset.samples[i][0] for i in range(len(datasets.train_dataset))]
        augment_data(image_paths = image_paths)

        # reload the full augmented training dataset
        train_dir_path : str = original_data_dir + '_augmented' + '_train'
        train_dataset : ImageFolder = get_image_folder_dataset(train_dir_path)
        datasets['train_dataset'] = train_dataset
        

        # refactor in class
        # - model + compile + fit
        # - train
        # - eval
        # - monitor experiment w/ wandb
        # - save zip model and augmented data

        # augmented_images = augment_image(original_image=image)

        # if not args.no_display:
        # display_images(augmented_images)

        # save_images(original_image_path=image_path, images=augmented_images)
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

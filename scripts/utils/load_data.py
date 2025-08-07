import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from ..Augmentation import augment_image, open_image, save_images
from ..Distribution import plot_distribution_combined
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from scripts.utils.logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

# TODO
# clean + docstring
# si debug sur un autre fichier, ne pas imprimer sur le mmien, decommenter augmentation
# faire une fonction sympa analyse descriptive


class LeaflictionData:
    """
    Class to handle data loading and pre-processing
    - Copy original directory
    - Creates splits train-val-test - copy images in-place
    - Apply data augmentation in-place
    - Loads all sets train-val-tests
    - Prepares loaders

    # https://docs.pytorch.org/docs/stable/data.html

    original_dir            # ex : ./images/Apple
    augmented_dir           # ex : ./images_augmented/Apple
    data_dir["train"]   # ex : ./images_augmented/Apple_train
    data_dir["test"]    # ex : ./images_augmented/Apple_test
    data_dir["val"]     # ex : ./images_augmented/Apple_val
    """

    def __init__(
        self,
        original_dir: str,
        force_preproc: bool = False,
        test_split: int = 0.2,
        val_split: int = 0.2,
        batch_size: int = 32,
        allowed_dir: str = "images",
    ):
        logger.info("Initializing LeaflictionData")
        self.original_dir: Path = Path(original_dir).resolve()
        self.force_preproc: bool = force_preproc
        self.test_split: int = test_split
        self.val_split: int = val_split
        self.batch_size: int = batch_size
        self.allowed_dir: str = allowed_dir

        self.augmented_dir: Path = None
        self.data_dirs: dict[str, Path] = {}
        self.original_ds: ImageFolder = None
        self.datasets: dict[str, ImageFolder] = {}
        self.loaders: dict[str, DataLoader] = {}

        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}
        self.class_count_idx: dict[int, int] = {}
        self.class_count_name: dict[str, int] = {}

        self._set_augmented_dir()
        self._set_data_dir()

        self._initializeLeaflictionData()

        logger.debug(self)

        self.data_analysis()

    def __repr__(self):
        class_to_idx = pd.DataFrame.from_dict(self.class_to_idx, orient='index', columns=['idx'])
        class_count_name = pd.DataFrame.from_dict(self.class_count_name, orient='columns')

        return (
            f"LeaflictionData:\n"
            f"  original_dir\t\t={self.original_dir}\n"
            f"  augmented_dir\t\t={self.augmented_dir}\n"
            f"  data_dirs['train']\t={self.data_dirs['train']}\n"
            f"  data_dirs['val']\t={self.data_dirs['val']}\n"
            f"  data_dirs['test']\t={self.data_dirs['test']}\n"
            f"  original_ds_len\t={self.original_ds is not None}\n"
            f"  datasets['train']\t={'train' in self.datasets.keys()}\n"
            f"  datasets['val']\t={'val' in self.datasets.keys()}\n"
            f"  datasets['test']\t={'test' in self.datasets.keys()}\n"
            f"  force_preproc\t\t={self.force_preproc}\n"
            f"  test_split\t\t={self.test_split}\n"
            f"  val_split\t\t={self.val_split}\n"
            f"  batch_size\t\t={self.batch_size}\n"
            f"  allowed_dir\t\t={self.allowed_dir}\n"
            
            f"  class_to_idx\n"
            f"{class_to_idx}\n\n"
            f"  class_count_name\n"
            f"{class_count_name}\n\n"
        )

    def _set_augmented_dir(self):
        current = self.original_dir.resolve()
        parent = current.parent
        grandparent = parent.parent
        new_parent_name = f"{parent.name}_augmented"

        self.augmented_dir = Path(grandparent / new_parent_name / current.name)

    def _set_data_dir(self):
        base_dir = self.augmented_dir.parent
        base_name = self.augmented_dir.name

        self.data_dirs["train"] = Path(base_dir / f"{base_name}_train")
        self.data_dirs["val"] = Path(base_dir / f"{base_name}_val")
        self.data_dirs["test"] = Path(base_dir / f"{base_name}_test")

    def _is_safe_path(self, safe_dir: str, path: str) -> bool:
        allowed_base = os.path.abspath(safe_dir)
        absolute_path = os.path.abspath(path)
        return absolute_path.startswith(allowed_base)

    def _is_preproc_ready(self) -> bool:
        return (
            set(self.data_dirs) == {"train", "test", "val"}
            and all(self.data_dirs[name].exists() for name in ("train", "test", "val"))
            and self.augmented_dir
            and self.augmented_dir.exists()
        )

    def _set_mappings(self):
        """setting class mappings and class counts"""
        self.class_to_idx = self.original_ds.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def _set_class_count(self):
        """setting class mappings and class counts"""
        self.class_count_idx["original"] = dict(Counter(self.original_ds.targets))
        self.class_count_idx["train"] = dict(Counter(self.datasets["train"].targets))
        self.class_count_idx["val"] = dict(Counter(self.datasets["val"].targets))
        self.class_count_idx["test"] = dict(Counter(self.datasets["test"].targets))

        self.class_count_name["original"] = {
            self.idx_to_class[idx]: count
            for idx, count in self.class_count_idx["original"].items()
        }
        self.class_count_name["train"] = {
            self.idx_to_class[idx]: count
            for idx, count in self.class_count_idx["train"].items()
        }
        self.class_count_name["val"] = {
            self.idx_to_class[idx]: count
            for idx, count in self.class_count_idx["val"].items()
        }
        self.class_count_name["test"] = {
            self.idx_to_class[idx]: count
            for idx, count in self.class_count_idx["test"].items()
        }

    def _copy_files(self, files: list[str], destination_dir: Path):
        """Copies files from provided list to the provided destination directory"""
        destination_dir.mkdir(parents=True, exist_ok=True)
        filename = destination_dir.parent.name + '/' + destination_dir.name
        for file in tqdm(files, desc=f"Copying to {filename}", unit="images"):
            shutil.copy(file, destination_dir)


    def _get_image_folder(self, data_dir: Path) -> ImageFolder:
        """
        Loads a dataset and applies data processing (uniform size, cast to tensors)
        Args
            data_dir (Path) : Path to the dataset to load
        Returns
            (ImageFolder) : Loaded dataset
        """
        if not self._is_safe_path(self.allowed_dir, data_dir):
            raise ValueError(
                f"'{data_dir}' is outside allowed directory. Allowed directory : 'images*/'"
            )

        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"'{data_dir}' is not a valid directory.")

        # transforms : used to preprocess and augment data
        # transforms.Compose : chains several transforms into a preproc pipeline
        train_transform = transforms.Compose(
            [
                transforms.Resize(
                    (256, 256)
                ),  # enforced in case of surprise even though all images are that size
                transforms.ToTensor(),  # PIL.Image.Image to torch.Tensor
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
        logger.info(f"Successfully loaded '{data_dir}'")

        return dataset

    def _initializeLeaflictionData(self):
        """
        - Executes preprocessing if not done already, or force_preproc flag is True:
            - Creates train-val-test sets
            - Augment training data
        # and also, and we will refactor this into separate functions:
            - Loads all train-val-test sets
            - Loads all train-val-test loaders
        """
        if self.force_preproc or not self._is_preproc_ready():
            logger.info('Going through preprocessing')

            # _create_copy_dataset -> Copy and loads a copy of the dataset
            self._create_copy_dataset()

            # _make_train_test_split -> splits, creates and loads 3 datasets : train, val, test
            self._make_train_test_split()

            # _data_augmentation -> create and save augmented images for training set
            train_image_paths = [
                self.datasets["train"].samples[i][0]
                for i in range(len(self.datasets["train"]))
            ]
            self._data_augmentation(image_paths=train_image_paths)

            # reload the full augmented training dataset
            self.datasets["train"] = self._get_image_folder(self.data_dirs["train"])
            self._set_class_count()

        else:
            logger.info('Datasets are ready. Skipping preprocessing')
            self.original_ds = self._get_image_folder(self.augmented_dir)
            self._set_mappings()
            self.datasets["train"] = self._get_image_folder(self.data_dirs["train"])
            self.datasets["val"] = self._get_image_folder(self.data_dirs["val"])
            self.datasets["test"] = self._get_image_folder(self.data_dirs["test"])
            self._set_class_count()

        # generates data loaders for training
        self._get_data_loaders()

    def _create_copy_dataset(self):
        """
        Creates and loads a copy of original dataset
        """
        original_filename = self.original_dir.parent.name + '/' + self.original_dir.name
        destination_filename = self.augmented_dir.parent.name + '/' + self.augmented_dir.name
        logger.info(f"Copying {original_filename} to {destination_filename}")
        # create a copy of original dataset
        shutil.copytree(self.original_dir, self.augmented_dir, dirs_exist_ok=True)

        # load original dataset
        self.original_ds = self._get_image_folder(self.augmented_dir)
        self._set_mappings()

    def _make_train_test_split(self) -> ImageFolder:
        """
        Computes sizes of train, test, val sets
        Copies files from original dataset into train, test, val sets
        Respects the original set structure
        Stratifies classes : will proportionally copy from each class into the new set
        """

        if self.test_split + self.val_split >= 0.9:
            raise ValueError("test_split + val_split must be less than 0.9")

        # create the directories
        if not self.augmented_dir.exists():
            raise NotADirectoryError(f"{self.augmented_dir} does not exist.")

        for dir_path in self.data_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Group files by class index into a dict
        class_to_files = defaultdict(list)
        for filepath, class_idx in self.original_ds.samples:
            class_to_files[class_idx].append(Path(filepath))

        # Copy files, stratified
        for class_idx, files in class_to_files.items():
            class_name = self.idx_to_class[class_idx]
            random.shuffle(files)
            total = len(files)
            test_len = int(self.test_split * total)
            val_len = int(self.val_split * (total - test_len))
            train_len = total - test_len - val_len

            train_files = files[:train_len]
            val_files = files[train_len : train_len + val_len]
            test_files = files[train_len + val_len :]

            # Copy
            self._copy_files(train_files, self.data_dirs["train"] / class_name)
            self._copy_files(val_files, self.data_dirs["val"] / class_name)
            self._copy_files(test_files, self.data_dirs["test"] / class_name)

        # get all split datasets
        self.datasets["train"] = self._get_image_folder(self.data_dirs["train"])
        self.datasets["val"] = self._get_image_folder(self.data_dirs["val"])
        self.datasets["test"] = self._get_image_folder(self.data_dirs["test"])

        return {
            "train": self.datasets["train"],
            "val": self.datasets["val"],
            "test": self.datasets["test"],
        }

    def _data_augmentation(self, image_paths: list[str]) -> None:
        """
        Generates 6 types of data augmentation per image provided.
        Saved in-place, type of augmentation appended to filename.
        Args:
            image_paths (list[str]) a list of image paths strings
        """
        for image_path in tqdm(image_paths, desc="Augmenting training data", unit="image"):
            image = open_image(image_path=image_path)
            augmented_images = augment_image(original_image=image)
            save_images(original_image_path=image_path, images=augmented_images)

    def _get_data_loaders(
        self,
    ) -> pd.DataFrame:
        # sampler : used during training to apply a specific sampling strategy
        # WeightedRandomSampler handles class imbalance
        # per-sample weights because WeightedRandomSampler samples individual data points, not classes
        train_labels = self.datasets["train"].targets
        train_class_counts = Counter(train_labels)

        train_weights = 1.0 / torch.tensor(
            [train_class_counts[label] for label in train_labels], dtype=torch.float
        )
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        # DataLoader : dataset iterable loader/collator for in-training batch processing
        self.loaders["train_loader"] = DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            sampler=train_sampler,
        )
        self.loaders["test_loader"] = DataLoader(
            self.datasets["test"], batch_size=self.batch_size, shuffle=False
        )
        self.loaders["val_loader"] = DataLoader(
            self.datasets["val"], batch_size=self.batch_size, shuffle=False
        )
        logger.info(f"Data loader available : {self.loaders.keys()}")

    def data_analysis(self):
        class_count_name = pd.DataFrame.from_dict(self.class_count_name, orient='columns')
        original_df = class_count_name[['original']].reset_index(names='category')
        original_df.rename(columns={'original': 'count'}, inplace=True)

        train_df = class_count_name[['train']].reset_index(names='category')
        train_df.rename(columns={'train': 'count'}, inplace=True)

        val_df = class_count_name[['val']].reset_index(names='category')
        val_df.rename(columns={'val': 'count'}, inplace=True)

        test_df = class_count_name[['test']].reset_index(names='category')
        test_df.rename(columns={'test': 'count'}, inplace=True)

        for key, df in {"original_df":original_df, "train_df":train_df, "val_df":val_df, "test_df":test_df}.items():
            print(key)
            print(df)
            plot_distribution_combined(df=df, title=key)

if __name__ == "__main__":
    data = LeaflictionData(
        original_dir="./images/Apple",
        force_preproc=True,
        test_split=0.2,
        val_split=0.2,
        batch_size=32,
        allowed_dir="images",
    )

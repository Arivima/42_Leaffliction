"""
load_data.py
- LeaflictionData class
"""

import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from scripts.utils.logger import get_logger

from ..Augmentation import augment_image, display_images, open_image, save_images
from ..Distribution import plot_multiple_distributions

logger = get_logger(__name__)


class LeaflictionData:
    """
    A class to handle data loading and pre-processing
    - Creates a back up for the original directory
    - Splits dataset into subsets for train-val-test (save in-place)
    - Apply data augmentation (save in-place) (see samples/)
    - Apply preprocessing : size harmonization, normalization, conversion to tensors
    - Loads all sets train-val-tests
    - Prepares training loaders
    - Provides descriptive analysis (standout + see plot/)

    - If data has already been processed once, set force_preproc to True to force processing
    - If force processing is activated, existing processed data will be deleted, user confirmation is required
    - Dataset type : torchvision.datasets.ImageFolder
        https://docs.pytorch.org/docs/stable/data.html

    Class arguments:
    - original_dir: str             (Mandatory) : original image dataset    # ex : ./images/Apple
    - force_preproc: bool = False   (Optional)  : ensures pre-processing execution even if data is ready
    - test_split: int = 0.2         (Optional)  : ratio of test set vs train set
    - val_split: int = 0.2          (Optional)  : ratio of validation set vs train set (train - test)
    - batch_size: int = 32          (Optional)  : how many images per training batch
    - allowed_dir: str = "images"   (Optional)  : where to process images   # ex : ./images

    Attributes:
    - augmented_dir: Path                       : path of processed image dataset
    - data_dirs: dict[str, Path]                : path of processed training, validation and test sets
    - data_dir_train_raw: Path                  : path of raw training set
    - original_ds: ImageFolder                  : original dataset
    - datasets: dict[str, ImageFolder]          : processed training, validation and test sets
    - loaders: dict[str, DataLoader]            : torch DataLoader for training

    - class_to_idx: dict[str, int]              : mapping from class (target) to index
    - idx_to_class: dict[int, str]              : mapping from index to class (target)
    - class_count_idx: dict[int, int]           : image count per class by index
    - class_count_name: dict[str, int]          : image count per class by name

    the class will create the following directories:
    - augmented_dir         # ex : ./images_augmented/Apple             (processed image dataset)
    - data_dir["train"]     # ex : ./images_augmented/Apple_train       (processed training set)
    - data_dir["test"]      # ex : ./images_augmented/Apple_test        (processed test set)
    - data_dir["val"]       # ex : ./images_augmented/Apple_val         (processed validation set)
    - data_dir_train_raw    # ex : ./images_augmented/Apple_train_raw   (raw training set)
    - samples               # ex : ./samples                            (sample of augmented images)
    - plots                 # ex : ./plots                              (datasets descriptive plots)

    P
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
        """Gets datasets ready for training"""
        logger.info("Initializing LeaflictionData")

        self._init_attributes()

        self.original_dir: Path = Path(original_dir).resolve()
        self.force_preproc: bool = force_preproc
        self.test_split: int = test_split
        self.val_split: int = val_split
        self.batch_size: int = batch_size
        self.allowed_dir: str = allowed_dir

        self._init_paths()

        if self.force_preproc and self.augmented_dir.exists():
            self._clean_directories()

        if not self._is_preproc_ready():
            logger.info("No existing data. Going through preprocessing")
            self._preprocess()
        else:
            logger.info("Datasets are ready. Skipping preprocessing")
            self._load()

        self._create_data_loaders()

        logger.info("Running descriptive analysis")
        logger.debug(self)
        self._run_data_analysis()

    def _init_attributes(self):
        """initializes all attributes - full list of class attributes"""
        self.original_dir: Path = None
        self.force_preproc: bool = None
        self.test_split: int = None
        self.val_split: int = None
        self.batch_size: int = None
        self.allowed_dir: str = None

        self.augmented_dir: Path = None
        self.data_dir_train_raw: Path = None
        self.data_dirs: dict[str, Path] = {}
        self.original_ds: ImageFolder = None
        self.datasets: dict[str, ImageFolder] = {}
        self.loaders: dict[str, DataLoader] = {}

        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}
        self.class_count_idx: dict[int, int] = {}
        self.class_count_name: dict[str, int] = {}

    def _init_paths(self):
        """
        Initializes paths. Uses self.original_dir to set:
        - self.augmented_dir
        - self.data_dirs
        """
        # Set augmented directory
        current = self.original_dir.resolve()
        parent = current.parent
        grandparent = parent.parent
        new_parent_name = f"{parent.name}_augmented"

        self.augmented_dir = Path(grandparent / new_parent_name / current.name)

        # Set data split directories
        base_dir = self.augmented_dir.parent
        base_name = self.augmented_dir.name

        self.data_dirs["train"] = Path(base_dir / f"{base_name}_train")
        self.data_dirs["val"] = Path(base_dir / f"{base_name}_val")
        self.data_dirs["test"] = Path(base_dir / f"{base_name}_test")

        # Set train raw directory
        self.data_dir_train_raw = Path(base_dir / f"{base_name}_train_raw")

    def _clean_directories(self):
        """removes existing directories at the provided path"""
        logger.warning("Force preprocessing enabled : cleaning all augmented folders")
        subdirs = [
            subdir
            for subdir in self.augmented_dir.parent.iterdir()
            if subdir.is_dir() and subdir.name.startswith(self.augmented_dir.name)
        ]
        print("Deleting directories:")
        for subdir in subdirs:
            print(f"{subdir}")
        answer = input("Yes / No ? ")
        if answer.lower() in ["yes", "y"]:
            for subdir in subdirs:
                if subdir.exists():
                    shutil.rmtree(subdir)
                    logger.info(f"Deleted {subdir}")
        else:
            logger.info("Ignoring force_preproc flag. Resuming.")


    def _load(self):
        """
        - loads:
            - original dataset
            - train-val-test sets
        - sets class mappings and class counts metadata
        """
        # load copy of original dataset
        self.original_ds = self._load_image_folder(self.augmented_dir)
        self._set_mappings()

        # load split datasets
        self.datasets["train"] = self._load_image_folder(self.data_dirs["train"])
        self.datasets["val"] = self._load_image_folder(self.data_dirs["val"])
        self.datasets["test"] = self._load_image_folder(self.data_dirs["test"])
        self._set_class_count()

    def _preprocess(self):
        """
        - Executes preprocessing:
            - Makes a copy of the original dataset
            - Creates train-val-test sets in-place
            - Augment training data
            - loads all datasets
        """
        # Make a copy of the dataset
        self._copy_dataset(self.original_dir, self.augmented_dir)

        # load copy of original dataset
        self.original_ds = self._load_image_folder(self.augmented_dir)
        self._set_mappings()

        # Splits dataset into 3 subsets : train, val, test - copy them in-place
        self._split_dataset()

        # load split datasets
        self.datasets["train"] = self._load_image_folder(self.data_dirs["train"])
        self.datasets["val"] = self._load_image_folder(self.data_dirs["val"])
        self.datasets["test"] = self._load_image_folder(self.data_dirs["test"])

        # Create augmented images for training set - save them in-place
        train_image_paths = [
            self.datasets["train"].samples[i][0]
            for i in range(len(self.datasets["train"]))
        ]
        self._run_data_augmentation(image_paths=train_image_paths)

        # reload the fully augmented training dataset
        self.datasets["train"] = self._load_image_folder(self.data_dirs["train"])
        self._set_class_count()

    def __repr__(self):
        """overrides default implementation for debugging purposes"""
        class_to_idx = pd.DataFrame.from_dict(
            self.class_to_idx, orient="index", columns=["idx"]
        )

        return (
            f"LeaflictionData:\n"
            f"  original_dir\t\t={self.original_dir}\n"
            f"  augmented_dir\t\t={self.augmented_dir}\n"
            f"  data_dirs['train']\t={self.data_dirs['train']}\n"
            f"  data_dirs['val']\t={self.data_dirs['val']}\n"
            f"  data_dirs['test']\t={self.data_dirs['test']}\n"
            f"  original_ds\t\t={self.original_ds is not None}\n"
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
        )

    def _is_safe_path(self, safe_dir: str, path: str) -> bool:
        """Return True is user given path is within allowed directory"""
        allowed_base = os.path.abspath(safe_dir)
        absolute_path = os.path.abspath(path)
        return absolute_path.startswith(allowed_base)

    def _is_preproc_ready(self) -> bool:
        """returns True if all directories are created and exist"""
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
        train_raw_ds = self._load_image_folder(self.data_dir_train_raw)

        self.class_count_idx["original"] = dict(Counter(self.original_ds.targets))
        self.class_count_idx["train_raw"] = dict(Counter(train_raw_ds.targets))
        self.class_count_idx["train"] = dict(Counter(self.datasets["train"].targets))
        self.class_count_idx["val"] = dict(Counter(self.datasets["val"].targets))
        self.class_count_idx["test"] = dict(Counter(self.datasets["test"].targets))

        self.class_count_name["original"] = {
            self.idx_to_class[idx]: count
            for idx, count in self.class_count_idx["original"].items()
        }
        self.class_count_name["train_raw"] = {
            self.idx_to_class[idx]: count
            for idx, count in self.class_count_idx["train_raw"].items()
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

    def _copy_dataset(self, source_dir, destination_dir):
        """
        Creates a copy of a source dataset into destination
        ex : images/ to images_augmented/
        """
        source_dirname = source_dir.parent.name + "/" + source_dir.name
        destination_dirname = destination_dir.parent.name + "/" + destination_dir.name
        logger.info(f"Copying {source_dirname} to {destination_dirname}")

        shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)

    def _copy_files(self, files: list[str], destination_dir: Path):
        """
        Copies files from a list to a destination directory
        - checks if files already exist before copying
        function can be improved with concurrent.futures.ThreadPoolExecutor
        """
        # creates the directory
        destination_dir.mkdir(parents=True, exist_ok=True)

        # copies files to destination
        filename = destination_dir.parent.name + "/" + destination_dir.name
        for file in tqdm(files, desc=f"Copying to {filename:<35}", unit="images"):
            dst_file = destination_dir / Path(file).name
            if not dst_file.exists():
                shutil.copy(file, destination_dir)

    def _load_image_folder(self, data_dir: Path) -> ImageFolder:
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
                # transforms.Normalize(), # implement later if training curves don't converge
            ]
        )

        # ImageFolder : lazy-loads dataset from folder (specific structure)
        #      ImageFolder.__init__() : lists the content of folder and checks for validity
        #            self.samples = make_dataset(directory, class_to_idx, extensions, is_valid_file)
        #      dataset = ImageFolder"data/images") -> lists files and collects labels
        #      print(dataset.samples[0]) -> access metadata
        # the image is loaded only when you access dataset[i] or iterate over a DataLoader
        #      under the hood : __getitem__ is the actual function that loads the file
        dataset = datasets.ImageFolder(
            root=data_dir,
            transform=train_transform,
            # target_transform=,
            # is_valid_file default implementation checks for extension validity
        )
        logger.info(f"Successfully loaded '{data_dir}'")

        return dataset

    def _split_dataset(self):
        """
        Computes sizes of train, test, val sets
        Copies files from original dataset into train, test, val sets
        Respects the original set structure
        Stratifies classes : will proportionally copy from each class into the new set
        Uses:
        - test/train splits ratio : self.test_split, self.val_split
        - directory paths : self.augmented_dir, self.data_dirs
        - images path : self.original_ds.sample
        - class mapping : self.idx_to_class
        - self._copy_files
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

        # Stratified copy - shuffled pick per class
        for class_idx, files in class_to_files.items():
            # shuffle
            random.shuffle(files)

            # split
            total = len(files)
            test_size = int(self.test_split * total)
            val_size = int(self.val_split * (total - test_size))
            train_size = total - test_size - val_size

            train_files: list[str] = files[:train_size]
            val_files: list[str] = files[train_size : train_size + val_size]
            test_files: list[str] = files[train_size + val_size :]

            # Copy
            class_name = self.idx_to_class[class_idx]
            self._copy_files(train_files, self.data_dirs["train"] / class_name)
            self._copy_files(val_files, self.data_dirs["val"] / class_name)
            self._copy_files(test_files, self.data_dirs["test"] / class_name)

    def _run_data_augmentation(self, image_paths: list[str]):
        """
        - Generates 6 types of data augmentation per image provided.
        - Saved in-place, type of augmentation appended to filename.
        - show sample of augmentation side by side, saved in samples/
        Args:
            image_paths (list[str]) a list of image paths strings
        """
        self._copy_dataset(self.data_dirs["train"], self.data_dir_train_raw)

        counter = 0
        for image_path in tqdm(
            image_paths, desc="Augmenting training data", unit="image"
        ):
            image = open_image(image_path=image_path)
            augmented_images = augment_image(original_image=image)
            save_images(original_image_path=image_path, images=augmented_images)

            file = Path(image_path)

            parts = [
                part
                for name in [file.parent.parent.name, file.parent.name, file.name]
                for part in name.split("_")
            ]
            deduplicated = list(dict.fromkeys(parts))
            filename = "_".join(deduplicated)
            if counter < 10:
                display_images(
                    images=augmented_images, save=True, display=False, filename=filename
                )
                counter += 1

    def _create_data_loaders(self):
        """
        - Sets data loaders for train, test, val sets
        - Uses WeightedRandomSampler for the training set based on target class distribution
        """
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
        logger.info(f"Data loaders available : {self.loaders.keys()}")

    def _run_data_analysis(self):
        """
        - displays class counts per dataset split, before and after augmentation
        - creates descriptive graphs, saved in plots/
        """

        def get_data():
            data = {
                dataset: self.class_count_name.get(dataset, {})
                for dataset in self.class_count_name.keys()
            }
            return data

        def get_df(data):
            df = pd.DataFrame(data).fillna(0).astype(int)
            df["total"] = df[["train", "val", "test"]].sum(axis=1)
            df.loc["total"] = df.sum(axis=0)
            return df

        def display_df(df):
            formatters = {col: "{:,}".format for col in df.columns}
            print(df.to_string(formatters=formatters))

        def get_count_df(
            class_count_name: dict[str, dict[str, int]], col_name: str
        ) -> pd.DataFrame:
            df = pd.DataFrame.from_dict(class_count_name, orient="columns")
            if col_name not in df.columns:
                raise ValueError(f"'{col_name}' not found in class_count_name")
            df_split = df[[col_name]].reset_index(names="category")
            df_split.rename(columns={col_name: "count"}, inplace=True)
            return df_split

        data = get_data()
        df = get_df(data)
        display_df(df)

        distributions = []
        for dataset in data.keys():
            df = get_count_df(self.class_count_name, dataset)
            distributions.append((df, dataset))

        title = self.original_dir.resolve().name
        plot_multiple_distributions(plot_title=title, distributions=distributions)


if __name__ == "__main__":
    import sys

    try:
        logger.info("Starting load_data.py")

        data = LeaflictionData(
            original_dir="./images/Apple",
            force_preproc=True,
            test_split=0.2,
            val_split=0.2,
            batch_size=32,
            allowed_dir="images",
        )

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

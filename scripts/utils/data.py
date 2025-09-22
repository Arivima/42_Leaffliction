"""
data.py
- LeaflictionData class
"""

import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torchvision.utils import save_image

from scripts.utils.logger import get_logger

from scripts.Augmentation import augment_image, display_images, open_image, save_images
from scripts.Distribution import plot_multiple_distributions


import numpy as np
from plantcv import plantcv as pcv      
from PIL import Image

logger = get_logger(__name__)



class CustomTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        img = np.array(img)
        b = pcv.rgb2gray_lab(rgb_img=img, channel='b')
        b_thresh = pcv.threshold.otsu(gray_img=b, object_type='light')
        mask = pcv.fill_holes(bin_img=b_thresh)
        apply_mask = pcv.apply_mask(img=img, mask=mask, mask_color='white')
        maskPil = Image.fromarray(apply_mask)
        return maskPil


class LeaflictionData:
    """
    A class to handle data loading and pre-processing
    - Creates a copy of the original directory
    - Splits dataset into subsets for train-val-test (save in-place)
    - Apply data augmentation (save in-place) (see samples/)
    - Apply preprocessing : size harmonization, normalization, conversion to tensors
    - Loads all sets train-val-tests
    - Prepares training loaders
    - Provides descriptive analysis (standout + see plot/)

    - If data has already been processed once, set force_preproc to True to force processing
    - If force processing is activated, existing processed data will be deleted, user confirmation is required
    - PyTorch has two primitives to work with data: torch.utils.data.DataLoader and torch.utils.data.Dataset. 
        Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.
        DataLoader supports automatic batching, sampling, shuffling and multiprocess data loading
        We are using Dataset type : torchvision.datasets.ImageFolder
        https://docs.pytorch.org/docs/stable/data.html

    Class arguments:
    - original_dir: str | Path      (Mandatory) : original image dataset    # ex : ./images/Apple
    - force_preproc: bool = False   (Optional)  : ensures pre-processing execution even if data is ready
    - test_split: int = 0.2         (Optional)  : ratio of test set vs train set
    - val_split: int = 0.2          (Optional)  : ratio of validation set vs train set (train - test)
    - batch_size: int = 32          (Optional)  : how many images per training batch

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

    Raises:
    - ValueError if original_dir is invalid
    """

    def __init__(
        self,
        original_dir: str | Path,
        force_preproc: bool = False,
        test_split: int = 0.2,
        val_split: int = 0.2,
        batch_size: int = 64,
    ):
        """Gets datasets ready for training"""
        logger.info("Initializing LeaflictionData")

        # initialize attributes and paths
        self._init_attributes()

        try:
            p = Path(original_dir).expanduser()
            self.original_dir: Path = p if p.is_absolute() else p.resolve(strict=True)
        except Exception as e:
            raise ValueError("Invalid attribute 'original_dir'", e)
        
        self.force_preproc: bool = force_preproc
        self.test_split: int = test_split
        self.val_split: int = val_split
        self.batch_size: int = batch_size

        self._init_paths()

        # if asked by user, reset 'augmented_directory'
        do_clean = self.force_preproc and self.augmented_dir.exists()
        if do_clean:
            self._clean_directories()

        # preprocess & load data
        do_preproc = not self._is_preproc_ready()
        if do_preproc:
            logger.info("No existing data. Going through preprocessing")
            self._preprocess()
        else:
            logger.info("Datasets are ready. Skipping preprocessing")
            self._load()

        # set-up data-loaders for training
        self._create_data_loaders()
        logger.info(f"Data loaders available : {self.loaders.keys()}")

        # save a sample of the transformed dataset used for training
        # self.save_dataset_transfo()

        # provides a descriptive dataset analysis
        if do_preproc:
            logger.info("Running descriptive analysis on processed data")
            self._run_data_analysis()



    def _init_attributes(self):
        """initializes all attributes - full list of class attributes"""
        self.original_dir: str | Path = None
        self.force_preproc: bool = None
        self.test_split: int = None
        self.val_split: int = None
        self.batch_size: int = None

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

        self.mean = [0.7165, 0.7616, 0.6708]    # written in hard from previous computation
        self.std = [0.3109, 0.2628, 0.3583]     # written in hard from previous computation
        self.preproc_pipeline = None

    def _init_paths(self):
        """
        Initializes paths. Uses self.original_dir to set:
        - self.augmented_dir
        - self.data_dirs
        """
        # Set augmented directory
        current = self.original_dir                     # ex '42_Leaffliction/images/Apple"
        parent = current.parent                         # '42_Leaffliction/images'
        grandparent = parent.parent                     # '42_Leaffliction'
        new_parent_name = f"{parent.name}_augmented"    # '42_Leaffliction/images_augmented'

        if parent.name != 'images' or grandparent.name != '42_Leaffliction':
            raise ValueError(f"Invalid attribute 'original_dir' : {current}. Dataset should be located in the 'images/' directory.")

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


    def _format_metadata(self):
        """Creates a json with metadata for debugging purposes"""
        def _s(x):
            return str(x) if x is not None else None

        payload = {
            "LeaflictionData": {
                "paths": {
                    "original_dir": _s(self.original_dir),
                    "augmented_dir": _s(self.augmented_dir),
                    "data_dirs": {k: _s(v) for k, v in self.data_dirs.items()},
                    "data_dir_train_raw": _s(self.data_dir_train_raw),
                },
                "flags": {
                    "force_preproc": bool(self.force_preproc),
                    "test_split": float(self.test_split) if self.test_split is not None else None,
                    "val_split": float(self.val_split) if self.val_split is not None else None,
                    "batch_size": int(self.batch_size) if self.batch_size is not None else None,
                },
                "loaded": {
                    "data_processed": self._is_preproc_ready(),
                    "original_ds": self.original_ds is not None,
                    "datasets": {k: (k in self.datasets) for k in ("train", "val", "test")},
                    "loaders": {k: (k in self.loaders) for k in ("train", "val", "test")},
                },
                "classes": {
                    "num_classes": len(self.idx_to_class) if self.idx_to_class else 0,
                    "idx_to_class": self.idx_to_class,
                },
            }
        }
        return payload
        
    def __repr__(self):
        """overrides default implementation for debugging purposes"""
        payload = self._format_metadata()
        return json.dumps(payload, indent=4, sort_keys=False, ensure_ascii=False)


    # def _is_safe_path(self, path: str) -> bool:
    #     """
    #     Ensure `path` is inside `<.../42_Leaffliction/images/>`.
    #     Return True is user given path is within allowed directory
    #     """

    #     project="42_Leaffliction"
    #     subdir="images"
    #     p = Path(path).resolve()

    #     # Find the nearest ancestor named `project`
    #     for a in (p, *p.parents):
    #         if a.name == project:
    #             safe_root = a / subdir
    #             if p.is_relative_to(safe_root):
    #                 return True
    #     return False


    @staticmethod
    def find_project_root(p: Path, name="42_Leaffliction") -> Path:
        p = p.resolve()
        for a in (p, *p.parents):
            if a.name == name:
                return a
        raise ValueError(f"Not inside '{name}': {p}")


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
        def _compute_image_normalization():
            dataset = datasets.ImageFolder(
                root=self.data_dirs['train'],
                transform=transforms.Compose([
                    CustomTransform(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
            )
            logger.info(f"Loaded '{self.data_dirs['train']}' to compute mean and std for normalization")

            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

            # Compute mean and std
            mean = 0.
            std = 0.
            total_images = 0

            for X, _ in tqdm(loader, desc="Computing ..."):
                # X shape: (batch_size, channels, height, width)
                batch_samples = X.size(0)
                # Flatten each image: (batch_size, channels, height*width)
                X = X.view(batch_samples, X.size(1), -1)
                # Mean across spatial dims → shape (batch_size, channels)
                batch_mean = X.mean(2)  
                # Std across spatial dims → shape (batch_size, channels)
                batch_std = X.std(2)    
                # Sum over the batch → shape (channels,)
                mean += batch_mean.sum(0)
                std += batch_std.sum(0)
                total_images += batch_samples

            mean /= total_images
            std /= total_images

            logger.info(f"Train dataset mean: {mean}")
            logger.info(f"Train dataset std: {std}")

            #!TODO WRITE TO JSON

            return mean, std

        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"'{data_dir}' is not a valid directory.")

        if self.mean is None or self.std is None:
            self.mean, self.std = _compute_image_normalization()

        if self.preproc_pipeline is None:
            self.preproc_pipeline = self.get_preproc_pipeline()

        # ImageFolder : lazy-loads dataset from folder (specific structure)
        #      ImageFolder.__init__() : lists the content of folder and checks for validity
        #            self.samples = make_dataset(directory, class_to_idx, extensions, is_valid_file)
        #      dataset = ImageFolder"data/images") -> lists files and collects labels
        #      print(dataset.samples[0]) -> access metadata
        # the image is loaded only when you access dataset[i] or iterate over a DataLoader
        #      under the hood : __getitem__ is the actual function that loads the file
        dataset = datasets.ImageFolder(
            root=data_dir,
            transform=self.preproc_pipeline,
            # target_transform=,
            # is_valid_file default implementation checks for extension validity
        )
        logger.info(f"Successfully loaded '{data_dir}'")

        return dataset

    @staticmethod
    def get_preproc_pipeline():
        # transforms : used to preprocess and augment data
        # transforms.Compose : chains several transforms into a preproc pipeline
        return transforms.Compose(
            [
                CustomTransform(),
                transforms.Resize((256, 256)),  # enforced in case of surprise even though all images are that size
                transforms.ToTensor(),  # PIL.Image.Image to torch.Tensor
                # transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist())
                transforms.Normalize([0.7165, 0.7616, 0.6708], [0.3109, 0.2628, 0.3583]) # computed on dataset
            ]
        )


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
            # augment image - save in-place
            image = open_image(image_path=image_path)
            augmented_images = augment_image(original_image=image)
            save_images(original_image_path=image_path, images=augmented_images)

            # export a sample side-by-side for the first 10 images
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
        - Uses a sampler (WeightedRandomSampler) for the training set based on target class distribution
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
        self.loaders["train"] = DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            sampler=train_sampler,
        )
        self.loaders["test"] = DataLoader(
            self.datasets["test"], 
            batch_size=self.batch_size, 
            shuffle=False
        )
        self.loaders["val"] = DataLoader(
            self.datasets["val"], 
            batch_size=self.batch_size, 
            shuffle=False
        )

    def _run_data_analysis(self):
        """
        - displays class counts per dataset split, before and after augmentation
        - creates descriptive graphs, saved in plots/
        """

        def get_data():
            """Return class counts for each dataset split."""
            data = {
                dataset: self.class_count_name.get(dataset, {})
                for dataset in self.class_count_name.keys()
            }
            return data

        def get_df(data):
            """Return a DataFrame with class counts and totals."""
            df = pd.DataFrame(data).fillna(0).astype(int)
            df["total"] = df[["train", "val", "test"]].sum(axis=1)
            df.loc["total"] = df.sum(axis=0)
            df.index.name = 'classes'
            return df

        def export_df(df, path : str | Path):
            """Export a DataFrame to a Markdown table file."""
            df_formatted = df.map(lambda x: f"{x:,}" if isinstance(x, (int, float)) else x)
            md_str = df_formatted.to_markdown(index=True)
            path.write_text(md_str)
            
        def export_metadata(self, path : str | Path):
            """Export metadata to a JSON file"""
            payload = json.loads(self.__repr__())
            with open(path, 'w', encoding='utf-8',) as f:
                json.dump(payload, f, ensure_ascii=False, indent=4)
            
        def get_count_df(
            class_count_name: dict[str, dict[str, int]], col_name: str
        ) -> pd.DataFrame:
            """Return a DataFrame with counts for a single dataset split."""
            df = pd.DataFrame.from_dict(class_count_name, orient="columns")
            if col_name not in df.columns:
                raise ValueError(f"'{col_name}' not found in class_count_name")
            df_split = df[[col_name]].reset_index(names="category")
            df_split.rename(columns={col_name: "count"}, inplace=True)
            return df_split


        # export object metadata as a json
        metadata_path = Path(f"plots/_{self.original_dir.name}_metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        export_metadata(self, path=metadata_path)
        logger.info(f"Exported preproc metadata at : {metadata_path}")

        # export dataset distribution
        data = get_data()
        df = get_df(data)
        df_path=Path(f"plots/{self.original_dir.name}_distribution.md")
        export_df(df, path=df_path)
        logger.info(f"Exported data distribution at : {df_path}")

        # export distribution plots
        distributions = []
        for dataset in data.keys():
            df = get_count_df(self.class_count_name, dataset)
            distributions.append((df, dataset))

        logger.info("Exporting plots ...")
        title = self.original_dir.resolve().name
        plot_multiple_distributions(plot_title=title, distributions=distributions)

    def save_dataset_transfo(self):
        """Saves a sample of the transformed dataset for reference"""
        project_root = self.augmented_dir.parent.parent                         # 42_Leafliction
        dataset_name = self.augmented_dir.name                                  # Apple
        outdir = Path(project_root / "samples" / "transformed" / dataset_name)  # 42_Leafliction/samples/transformed/Apple
        os.makedirs(outdir, exist_ok=True)

        for i, (X, y) in enumerate(self.loaders["train"]):
            if i < 5:
                for j in range(X.size(0)):
                    if j < 5:

                        save_image(X[j], f"{outdir}/{i*self.loaders['train'].batch_size + j}_class{y[j].item()}.png")




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
        )

        for X, y in data.loaders['train']:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

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

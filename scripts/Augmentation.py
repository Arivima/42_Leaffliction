"""
Augmentation.py
A program that generate 6 types of data augmentation for each image given as input.
Displays to screen and saves to a local directory (prefixed with original file name).
Types of augmentation: Flip, Rotate, Skew, Shear, Crop, Distortion
Use :
`$> uv run scripts/Augmentation.py images_augmented/Apple/Apple_healthy/image\ \(1\).JPG`
https://www.geeksforgeeks.org/python/python-opencv-cheat-sheet/
https://github.com/Jacob12138xieyuan/opencv-python-cheat-sheet/blob/master/opencv-python%20cheetsheet.ipynb
https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
https://www.comet.com/site/blog/opencv-python-cheat-sheet-from-importing-images-to-face-detection/
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils.logger import get_logger

logger = get_logger(__name__)

# AUGMENTED_OUTPUT_DIR = "images_augmented"


def open_image(image_path: str) -> np.ndarray:
    """Loads the image at `image_path` to a np.ndarray"""
    if not os.path.isfile(image_path):
        raise NotADirectoryError(f"'{image_path}' is not at a valid path.")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image file {image_path} could not be read.")

    logger.info(f"image.shape: {image.shape}")
    logger.info(f"image.size: {image.size}")
    logger.info(f"type(image): {type(image)}")
    logger.info(f"image.dtype: {image.dtype}")

    return image


# def flip_image(image: np.ndarray) -> np.ndarray:
#     """returns a flipped image - horizontal axis"""
#     logger.debug("flip_image")
#     return cv2.flip(image, 1)


def blur_image(image: np.ndarray) -> np.ndarray:
    """
    returns a blurred image
    https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    """
    logger.debug("blur_image")
    return cv2.blur(image, (5, 5))


def rotate_image(image: np.ndarray) -> np.ndarray:
    """
    returns a rotated image - 45°
    https://www.geeksforgeeks.org/python/python-opencv-cheat-sheet/
    """
    logger.debug("rotate_image")
    h, w = image.shape[:2]
    c = (w // 2, h // 2)
    a = 45
    s = 1.0  # no scaling
    rotation_matrix = cv2.getRotationMatrix2D(center=c, angle=a, scale=s)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h), borderValue=(255, 255, 255)
    )
    return rotated


def skew_image(image: np.ndarray) -> np.ndarray:
    """
    returns a skewed image
    https://www.geeksforgeeks.org/python/python-opencv-cheat-sheet/
    """
    logger.debug("skew_image")
    h, w = image.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    delta = random.uniform(-0.3, 0.3) * w
    pts2 = np.float32([[delta, 0], [w + delta, 0], [0, h]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, M, (w, h))


def shear_image(image: np.ndarray) -> np.ndarray:
    """
    returns a sheared image
    https://www.geeksforgeeks.org/python/python-opencv-cheat-sheet/
    """
    logger.debug("shear_image")
    h, w = image.shape[:2]
    shear_x = random.uniform(-0.3, 0.3)
    shear_y = random.uniform(-0.3, 0.3)
    M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    return cv2.warpAffine(image, M, (w, h))


def crop_image(image: np.ndarray) -> np.ndarray:
    """returns a cropped image, resized to be ingested in training pipeline"""
    logger.debug("crop_image")
    h, w = image.shape[:2]
    ratio = 0.15
    top = int(h * ratio)
    bottom = int(h * (1 - ratio))
    left = int(w * ratio)
    right = int(w * (1 - ratio))
    cropped = image[top:bottom, left:right]
    return cv2.resize(cropped, (w, h))


def distord_image(image: np.ndarray) -> np.ndarray:
    """returns a distorted image"""
    logger.debug("distord_image")
    h, w = image.shape[:2]
    mapy, mapx = np.indices((h, w), dtype=np.float32)
    mapx = mapx + 20 * np.sin(mapy / 20.0)
    mapy = mapy + 20 * np.cos(mapx / 20.0)
    return cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR)


def contrast_image(image: np.ndarray, alpha: float = 2) -> np.ndarray:
    """
    returns a constrasted image
    https://www.tutorialspoint.com/how-to-change-the-contrast-and-brightness-of-an-image-using-opencv-in-python
    """
    logger.debug("contrast_image")
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


# def lighten_image(image: np.ndarray, beta: int = 70) -> np.ndarray:
#     """
#     returns a lightened image
#     https://www.tutorialspoint.com/how-to-change-the-contrast-and-brightness-of-an-image-using-opencv-in-python
#     """
#     logger.debug("lighten_image")
#     return cv2.convertScaleAbs(image, alpha=1, beta=beta)


def augment_image(original_image: np.ndarray) -> dict:
    """
    Takes an image and returns a dict with that image augmented in 6 different
    ways (key: augmentation type, value: augmented image)
    """

    AUGMENTATIONS = {
        # "flip": flip_image,
        "rotate": rotate_image,
        "skew": skew_image,
        "shear": shear_image,
        "crop": crop_image,
        "distortion": distord_image,
        "blur": blur_image,
        "contrast": contrast_image,
        # "lighten": lighten_image,
    }

    augmented_images = {}
    augmented_images["original_image"] = original_image
    for type, function in AUGMENTATIONS.items():
        augmented_images[type] = function(original_image)

    display_images(augmented_images)

    return augmented_images


def save_images(original_image_path: str, images: dict):
    """Saves images to original image directory"""
    # aug_img.save(os.path.join(base_dir, f"{base_name}_{name}.JPG"))

    parts = os.path.split(original_image_path)
    dir_path = parts[:-1]
    base_name = parts[-1]
    ext = os.path.splitext(original_image_path)

    logger.debug(f"parts : {parts}")
    logger.debug(f"dir_path : {dir_path}")
    logger.debug(f"base_name : {base_name}")
    logger.debug(f"ext : {ext}")

    os.makedirs(dir_path, exist_ok=True)

    for type, image in images.items():
        filename = f"{base_name}_{type}.{ext}"
        logger.debug(f"filename : {filename}")
        image_path = os.path.join(dir_path, filename)

        logger.info(f"Saving image to {image_path}")
        success = cv2.imwrite(image_path, image)
        if not success:
            logger.error(f"Error while saving the image : {filename}")


def display_images(images: dict):
    """Displays all augmented images side by side in one row"""
    n = len(images)
    if n == 0:
        return

    max_width = 12
    max_height = 4
    width = min(2 * n, max_width)
    height = min(2, max_height)

    _, axes = plt.subplots(1, n, figsize=(width, height))
    if n == 1:
        axes = [axes]

    for ax, (title, image) in zip(axes, images.items()):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    """Uses `argparse` to handle the arguments : image_path, output_dir"""
    parser = argparse.ArgumentParser(
        description="A program that generate 6 types of data augmentation for each image given as input."
    )
    parser.add_argument("image_path", help="Path to the image for input")
    # parser.add_argument(
    #     "output_dir",
    #     help="Path to the augmented image destination for output",
    #     default=AUGMENTED_OUTPUT_DIR,
    #     required=False, ## CHECK IF BEST PRACTICE
    # )
    return parser.parse_args()


## FOR NOW PROGRAM WILL BE EXECUTED WITH ONE SINGLE IMAGE
## WILL HAVE ANOTHER/ADAPTATION FOR ALL IMAGES
def main():
    try:
        logger.info("Starting Augmentation.py program")
        args = parse_args()
        image_path = args.image_path  # should be a valid path
        # output_dir = args.output_dir # if not a directory, will be created
        # logger.debug(f"Arguments : output_dir {output_dir}")

        ## CONTINUE PROTECTION AND TESTING INVALID ARGS

        logger.info(f"Processing image at path '{image_path}'")
        image = open_image(image_path=image_path)

        augmented_images = augment_image(original_image=image)
        return

        ## DISPLAY AUGMENTED IMAGE 1 screen 7 cols, one row per image
        display_images(augmented_images)

        ## SAVE TO LOCAL DIR
        ## RE_READ SUBJECT TO UNDERSTAND HOW TO STRUCTURE DIRECTORY
        save_images(original_image_path=image_path, images=augmented_images)

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

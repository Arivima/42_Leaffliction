"""
Augmentation.py

A program that generate 6 types of data augmentation for each image given as input.
Displays to screen and saves to a local directory (prefixed with original file name).
Optionnally deactivate display and only process images.

Use :
`$> uv run scripts/Augmentation.py images_augmented/Apple/Apple_healthy/image\ \(1\).JPG`
`$> uv run scripts/Augmentation.py --no-display images_augmented/Apple/Apple_healthy/image\ \(1\).JPG`

Ressources:
- https://thepythoncode.com/article/image-transformations-using-opencv-in-python
- https://www.geeksforgeeks.org/python/python-opencv-cheat-sheet/
- https://github.com/Jacob12138xieyuan/opencv-python-cheat-sheet/blob/master/opencv-python%20cheetsheet.ipynb
- https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
- https://www.comet.com/site/blog/opencv-python-cheat-sheet-from-importing-images-to-face-detection/

Types of augmentation included:
- Translation : linear transformation - shifts image towards a direction
- *Reflection : linear transformation - flips image along an axis
- Rotation : linear transformation - rotates around the center
- Shearing : linear transformation - skews the image by displacing pixels in one direction based on their position in another (creates a slanted effect).
- *Scaling : Linear transformation - enlarges (zoom) or shrinks the image by multiplying pixel positions by a scalar. Does not affect dimensions.
- *Distort : Non-linear warping — modifies the geometry of the image using a non-linear function, often mimicking lens distortion effects (e.g. barrel, fisheye).
- Cropping : Operation that cuts out image borders or a region; reduces size.
- *Resizing : Changes image dimensions via interpolation; affects resolution but preserves content.
- Blurring : Convolution operation that smooths the image by averaging nearby pixels.
- Constrast : Enhances the difference between light and dark areas.
- *Lighten : Increases overall brightness by adding intensity to pixels.

* Need to be activated manually
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from scripts.utils.logger import get_logger

logger = get_logger(__name__)


def is_valid_image_path(path: str) -> bool:
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    ext = os.path.splitext(path)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def is_safe_path(base_dir: str, path: str) -> bool:
    real_base = os.path.abspath(base_dir)
    real_path = os.path.abspath(path)
    return real_path.startswith(real_base)


def open_image(image_path: str) -> np.ndarray:
    """Loads the image at `image_path` to a np.ndarray"""
    # logger.info(f"Processing image at path '{image_path}'")
    if not is_valid_image_path(path=image_path):
        raise ValueError("Invalid file extension.")

    if not is_safe_path("images", image_path):
        raise ValueError(f"Image path '{image_path}' is outside allowed directory.")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image '{image_path}' not found.")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image file {image_path} could not be read.")

    # logger.info(f"image.shape: {image.shape}")
    # logger.info(f"type(image): {type(image)}")
    # logger.info(f"image.dtype: {image.dtype}")

    return image


def translate_image(image: np.ndarray, tx: int = 50, ty: int = 50) -> np.ndarray:
    """
    Image translation is the rectilinear shift of an image from one location to another, so the shifting of an object is called translation.
    https://thepythoncode.com/article/image-transformations-using-opencv-in-python
        # alternative
        matrix = np.float32([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
            ])
        return cv2.warpPerspective(image, matrix, (w, h), borderValue=(255, 255, 255))
    tx, ty : (int) translation factor for x, y
    Returns: (np.ndarray) a translated image.
    """
    h, w = image.shape[:2]
    matrix = np.float32(
        [
            [1, 0, tx],
            [0, 1, ty],
        ]
    )
    translated = cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))
    # logger.info(f'translate_image {translated.shape}')
    return translated


def flip_image(image: np.ndarray) -> np.ndarray:
    """
    Image reflection (or mirroring) is useful for flipping an image,
    it can flip the image vertically as well as horizontally.
    Here we flip the image along the horizontal axis.
    https://thepythoncode.com/article/image-transformations-using-opencv-in-python#Image_Reflection
        # alternative
        1 . Equilvalent to image = cv2.flip(image, 1)
        2. x-axis reflection - built-in : cv2.flip(image, 0)
        matrix = np.float32([[1,  0, 0],
                            [0, -1, h],
                            [0,  0, 1]])
        3. y-axis reflection - built-in : cv2.flip(image, 1)
        M = np.float32([[-1, 0, w],
                        [ 0, 1, 0],
                        [ 0, 0, 1]])
        return cv2.warpPerspective(image, matrix,(int(w),int(h)))
    Returns: (np.ndarray) a flipped image along the horizontal axis.
    """
    h, w = image.shape[:2]
    matrix = np.float32(
        [
            [1, 0, 0],
            [0, -1, h],
        ]
    )
    flipped = cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))
    # logger.info(f'flip_image {flipped.shape}')
    return flipped


def rotate_image(image: np.ndarray, angle: int = 215) -> np.ndarray:
    """
    Image Rotation : returns a rotated image by specified angle
    rotation : motion of a certain space that preserves at least one point
    https://thepythoncode.com/article/image-transformations-using-opencv-in-python#Image_Rotation
    https://www.geeksforgeeks.org/python/python-opencv-cheat-sheet/
        # alternative
        h, w = image.shape[:2]
        c = (w // 2, h // 2)
        s = 1.0  # no scaling
        matrix = cv2.getRotationMatrix2D(center=c, angle=angle, scale=s)
        return cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))
    Returns: (np.ndarray) a rotated image by `angle` degrees.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    theta = np.radians(-angle)
    cos, sin = np.cos(theta), np.sin(theta)
    matrix = np.array(
        [[cos, -sin, (1 - cos) * cx + sin * cy], [sin, cos, (1 - cos) * cy - sin * cx]],
        dtype=np.float32,
    )
    rotated = cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))
    # logger.info(f'rotate_image {rotated.shape}')
    return rotated


def shear_image(image: np.ndarray, sx: int = 0.5, sy: int = 0.0) -> np.ndarray:
    """
    Shear mapping is a linear map that displaces each point in a fixed direction,
    it substitutes every point horizontally or vertically by a specific value in
    proportion to its x or y coordinates, there are two types of shearing effects.
    Default values shear in the x direction.
    https://thepythoncode.com/article/image-transformations-using-opencv-in-python#Image_Shearing
    https://www.geeksforgeeks.org/python/python-opencv-cheat-sheet/
    sx, sy : (float) shearing factor for x, y
    Returns: (np.ndarray) a sheared image.
    """
    h, w = image.shape[:2]
    matrix = np.array([[1, sx, 0], [sy, 1, 0]], dtype=np.float32)
    sheared = cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))
    # logger.info(f'shear_image {sheared.shape}')
    return sheared


def scale_image(image: np.ndarray, sx: int = 0.5, sy: int = 0.5) -> np.ndarray:
    """
    Image scaling is a process used to resize a image.
    built-in function cv2.resize()
    sx, sy : scaling factors for x, y (>1 zooming effect, < 1 shrinking effect)
    Returns: (np.ndarray) a scaled image.
    """
    h, w = image.shape[:2]
    matrix = np.float32([[sx, 0, 0], [0, sy, 0]])
    scaled = cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))
    # logger.info(f'scale_image {scaled.shape}')
    return scaled


def crop_image(image: np.ndarray, ratio: float = 0.2) -> np.ndarray:
    """
    Image Cropping : returns a cropped image
    ratio : (float) cropping ratio (0-1) to be removed
    ! Needs to be resized to be ingested in training pipeline
    Returns: (np.ndarray) a cropped image.
    """
    if not (0 <= ratio <= 1):
        raise ValueError(
            f"Invalid ratio : {ratio}. Should be between 0 and 1 excluded."
        )
    h, w = image.shape[:2]
    ratio = ratio / 2
    top = int(h * ratio)
    bottom = int(h * (1 - ratio))
    left = int(w * ratio)
    right = int(w * (1 - ratio))
    cropped = image[top:bottom, left:right]
    # logger.info(f'crop_image {cropped.shape}')
    return cropped


def resize_image(image: np.ndarray, sx: int = 0.5, sy: int = 0.5) -> np.ndarray:
    """
    Image Resize is a process used to resize a image.
    built-in function cv2.resize()
    sx, sy : scaling factors for x, y (>1 zooming effect, < 1 shrinking effect)
    Returns: (np.ndarray) a scaled image.
    """
    h, w = image.shape[:2]
    new_w = int(w * sx)
    new_h = int(h * sy)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # logger.info(f'resize_image {resized.shape}')
    return resized


def distort_image(image: np.ndarray, k: float = 0.5) -> np.ndarray:
    """
    Apply barrel distortion using distortion map and cv2.remap.
    Image distortion : non linear transformation
    k : (float) >0 -> barrel distortion, <0 -> pincushion distortion
    Returns: (np.ndarray) a barrel distorted image.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    y, x = np.indices((h, w), dtype=np.float32)
    dx = (x - cx) / w
    dy = (y - cy) / h
    r2 = dx**2 + dy**2
    factor = 1 + k * r2
    map_x = cx + dx * w * factor
    map_y = cy + dy * h * factor

    distorted = cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )
    # logger.info(f'distort_image {distorted.shape}')
    return distorted


def blur_image(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Apply a blurring kernel to the image.
    A blurring kernel is a small matrix used in convolution to compute
    the new value of each pixel by averaging its neighborhood.
    Replacing the center pixel with the average of itself and its 8 neighbors.
    Kernel values are : 1 / (ksize * ksize), the sum of the kernel is 1
    https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    Equivalent to cv2.blur(image, (ksize, ksize))
    ksize : (int) blurring factor
    Returns: (np.ndarray) a blurred image.
    """
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    blurred = cv2.filter2D(image, -1, kernel)
    # logger.info(f'blur_image {blurred.shape}')
    return blurred


def contrast_image(image: np.ndarray, alpha: float = 2) -> np.ndarray:
    """
    Apply a formula on the pixel content to increase image constrast
    Formula : new_pixel = clip(alpha * old_pixel + beta, 0, 255)
    we cast to float32 to handle overflow before recasting to uint8
    https://www.tutorialspoint.com/how-to-change-the-contrast-and-brightness-of-an-image-using-opencv-in-python
    Equivalent to cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    alpha : (float) Contrast factor - >1 increases, <1 reduces
    beta : (int) Brightness offset - >0 lightens, <0 darkens
    Returns: (np.ndarray) a contrasted image.
    """
    beta = 0
    contrasted = image.astype(np.float32) * alpha + beta
    contrasted = np.clip(contrasted, 0, 255)
    # logger.info(f'contrast_image {contrasted.shape}')
    return contrasted.astype(np.uint8)


def lighten_image(image: np.ndarray, beta: int = 70) -> np.ndarray:
    """
    Apply a formula on the pixel content to lighten the image
    Formula : new_pixel = clip(alpha * old_pixel + beta, 0, 255)
    we cast to float32 to handle overflow before recasting to uint8
    https://www.tutorialspoint.com/how-to-change-the-contrast-and-brightness-of-an-image-using-opencv-in-python
    Equivalent to cv2.convertScaleAbs(image, alpha=1, beta=beta)
    alpha : (float) Contrast factor - >1 increases, <1 reduces
    beta : (int) Brightness offset - >0 lightens, <0 darkens
    Returns: (np.ndarray) a lightened image.
    """
    alpha = 1
    lightened = image.astype(np.float32) * alpha + beta
    lightened = np.clip(lightened, 0, 255)
    # logger.info(f'contrast_image {lightened.shape}')
    return lightened.astype(np.uint8)


def augment_image(original_image: np.ndarray) -> dict:
    """
    Takes an image and returns a dict with that image augmented in 6 different
    ways (key: augmentation type, value: augmented image)
    """
    # logger.info("Augmenting image")

    AUGMENTATIONS = {
        "Translate": translate_image,
        # "Flip": flip_image,
        "Rotate": rotate_image,
        "Shear": shear_image,
        # "Scale": scale_image,
        "Crop": crop_image,
        # "Resize": resize_image,
        # "Distortion": distort_image,
        "Blur": blur_image,
        "Contrast": contrast_image,
        # "Lighten": lighten_image,
    }

    augmented_images = {}
    augmented_images["original_image"] = original_image
    for key, function in AUGMENTATIONS.items():
        augmented_images[key] = function(original_image)

    return augmented_images


def display_images(
    images: dict, save: bool = False, display: bool = True, filename: str = "sample.jpg"
):
    """Displays all augmented images side by side in one row"""
    # logger.info("Displaying augmented images")
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

    if save:
        def find_project_root(p: Path, name="42_Leaffliction") -> Path:
            p = p.resolve()
            for a in (p, *p.parents):
                if a.name == name:
                    return a
            raise ValueError(f"Not inside '{name}': {p}")

        project_root = find_project_root(Path(os.getcwd()))  # 42_Leafliction
        if "Apple" in filename:
            dataset_name = "Apple"
        elif "Grape" in filename:
            dataset_name = "Grape"

        output_dir = Path(
            project_root / "samples" / "augmented" / dataset_name
        )  # 42_Leafliction/samples/augmented/Apple

        os.makedirs(output_dir, exist_ok=True)
        save_path = output_dir / filename
        plt.savefig(save_path)
        # logger.info(f"Saved figure to: {save_path}")

    if display:
        plt.show()
    else:
        plt.close()


def save_images(original_image_path: str, images: dict):
    """Saves images to original image directory"""
    first_split = os.path.split(original_image_path)
    second_split = os.path.splitext(first_split[-1])
    dir_path = first_split[0]
    base_name = second_split[0]
    ext = second_split[-1]
    # logger.info(f"Saving augmented images to '{dir_path}'")

    os.makedirs(dir_path, exist_ok=True)

    for key, image in images.items():
        if key != "original_image":
            filename = f"{base_name}_{key}{ext}"
            image_path = os.path.join(dir_path, filename)

            success = cv2.imwrite(image_path, image)
            if not success:
                logger.error(f"Could not save image '{filename}'")
                raise IOError(f"cv2.imwrite failed to save '{image_path}'")
            # else:
            # logger.info(f"Saved '{filename}'")


def parse_args() -> argparse.Namespace:
    """Uses `argparse` to handle the arguments : image_path, output_dir"""
    parser = argparse.ArgumentParser(
        description="A program that generate 6 types of data augmentation for each image given as input."
    )
    parser.add_argument("image_path", help="Path to the image for input")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    return parser.parse_args()


def main():
    try:
        # logger.info("Starting Augmentation.py program")

        args = parse_args()
        image_path = args.image_path

        image = open_image(image_path=image_path)

        augmented_images = augment_image(original_image=image)

        if not args.no_display:
            display_images(augmented_images)

        save_images(original_image_path=image_path, images=augmented_images)
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

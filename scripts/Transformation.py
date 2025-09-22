"""
Transformation.py

A program a program that generates 6 image transformations and a color analysis.
If a file is given the transformations will just be displayed.
Otherwise if an input document and an output folder are given the transformations
will be saved in the output folder. Generating image transformations may take some time

Use:
`$> uv run scripts/Transformation.py --help`
`$> uv run scripts/Transformation.py -src="images" -dst="Transformation"`
`$> uv run scripts/Transformation.py images/Apple/Apple_healthy/image\ \(1\).JPG`
`$> uv run scripts/Transformation.py images/Apple/Apple_healthy/image\ \(1\).JPG -roi -mask`

Ressources:
- https://plantcv.readthedocs.io/en/stable/

"""

import os
import re
import shutil
import sys

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv
from utils.logger import get_logger

logger = get_logger(__name__)


class ImageTransformation:
    def __init__(self, imgSrc, outDir=".", save=False):
        self.imgSrc = imgSrc
        self.img, self.path, filename = pcv.readimage(imgSrc, mode="rgba")
        self.filename = re.sub(r"\..*", "", filename)
        self.b = pcv.rgb2gray_lab(rgb_img=self.img, channel="b")
        self.b_thresh = pcv.threshold.otsu(gray_img=self.b, object_type="light")
        self.save = save
        self.outDir = outDir
        pcv.params.debug_outdir = outDir
        pcv.params.sample_label = "plant"
        if save:
            pcv.params.debug = "print"
        else:
            pcv.params.debug = "plot"

    def showOrigine(self):
        pcv.readimage(self.imgSrc, mode="rgba")
        if self.save:
            os.rename(
                self.outDir + "/input_image.png",
                self.outDir + "/" + self.filename + "Origine.png",
            )

    def showGaussian(self):
        pcv.params.device = 0
        pcv.gaussian_blur(self.b_thresh, ksize=(31, 31))
        if self.save:
            os.rename(
                self.outDir + "/0_gaussian_blur.png",
                self.outDir + "/" + self.filename + "Gaussian.png",
            )

    def showMask(self):
        pcv.params.device = 0
        pcv.apply_mask(img=self.img, mask=self.b_thresh, mask_color="white")
        if self.save:
            os.rename(
                self.outDir + "/0_masked.png",
                self.outDir + "/" + self.filename + "Mask.png",
            )

    def showRoi(self):
        top, left, bottom, right = self.rectangleRoiGeneration()
        pcv.params.device = 0
        pcv.roi.rectangle(img=self.img, x=left, y=top, h=bottom - top, w=right - left)
        if self.save:
            os.rename(
                self.outDir + "/0_roi.png",
                self.outDir + "/" + self.filename + "Roi.png",
            )

    def showAnalyse(self):
        pcv.params.device = 0
        pcv.analyze.size(img=self.img, labeled_mask=self.b_thresh, n_labels=1)
        if self.save:
            os.rename(
                self.outDir + "/1_shapes.png",
                self.outDir + "/" + self.filename + "Analyse.png",
            )

    def showPseudolandmarks(self):
        pcv.params.device = 0
        pcv.homology.x_axis_pseudolandmarks(
            img=self.img, mask=self.b_thresh, label="test"
        )
        if self.save:
            os.rename(
                self.outDir + "/0_x_axis_pseudolandmarks.png",
                self.outDir + "/" + self.filename + "Pseudolandmarks.png",
            )

    def showColorAnalyse(self):
        pcv.params.device = 0
        _, hist_data = pcv.visualize.histogram(
            img=self.img, mask=self.b_thresh, hist_data=True
        )
        if self.save:
            os.rename(
                self.outDir + "/3_hist.png",
                self.outDir + "/" + self.filename + "ColorAnalyse.png",
            )
        else:
            plt.plot(
                hist_data[hist_data["color channel"] == "red"]["pixel intensity"],
                hist_data[hist_data["color channel"] == "red"][
                    "proportion of pixels (%)"
                ],
                label="red",
                color="red",
            )
            plt.plot(
                hist_data[hist_data["color channel"] == "blue"]["pixel intensity"],
                hist_data[hist_data["color channel"] == "blue"][
                    "proportion of pixels (%)"
                ],
                label="blue",
                color="blue",
            )
            plt.plot(
                hist_data[hist_data["color channel"] == "green"]["pixel intensity"],
                hist_data[hist_data["color channel"] == "green"][
                    "proportion of pixels (%)"
                ],
                label="green",
                color="green",
            )
            plt.xlabel("pixel intensity")
            plt.ylabel("proportion of pixels (%)")
            plt.legend(title="color channel")
            plt.show()

    def rectangleRoiGeneration(self):
        rows, cols = np.where(self.b_thresh)
        top = rows.min()
        left = cols.min()
        bottom = rows.max()
        right = cols.max()
        return top, left, bottom, right


def showTransformation(
    tr, origine, gaussian, mask, roi, analyze, pseudolandmarks, color
):
    if origine:
        tr.showOrigine()
    if gaussian:
        tr.showGaussian()
    if mask:
        tr.showMask()
    if roi:
        tr.showRoi()
    if analyze:
        tr.showAnalyse()
    if pseudolandmarks:
        tr.showPseudolandmarks()
    if color:
        tr.showColorAnalyse()


@click.command()
@click.option(
    "-src",
    type=click.Path(exists=True, file_okay=False, writable=True),
    default=None,
    help="Folder source path",
)
@click.option("-dst", type=click.Path(), default=None, help="Destination directory")
@click.option("-origine", is_flag=True, help="Generate the original image")
@click.option("-mask", is_flag=True, help="Generate mask")
@click.option("-gaussian", is_flag=True, help="Generate gaussian blur")
@click.option("-roi", is_flag=True, help="Generate region of interest")
@click.option("-analyze", is_flag=True, help="Generate analyse object")
@click.option("-pseudolandmarks", is_flag=True, help="Generate pseudolandmarks")
@click.option("-color", is_flag=True, help="Generate color histogram")
@click.argument(
    "directsrc",
    type=click.Path(exists=True, dir_okay=False, writable=True),
    required=False,
)
def main(
    src,
    dst,
    origine,
    mask,
    gaussian,
    roi,
    analyze,
    pseudolandmarks,
    color,
    directsrc=None,
):
    if (
        origine or mask or gaussian or roi or analyze or pseudolandmarks or color
    ) == False:
        origine, mask, gaussian, roi, analyze, pseudolandmarks, color = (
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        )
    try:
        if directsrc:
            os.makedirs(f"tranceformImage/", exist_ok=True)
            tr = ImageTransformation(directsrc, outDir="tranceformImage/", save=True)
            showTransformation(
                tr, origine, mask, gaussian, roi, analyze, pseudolandmarks, color
            )
        elif src and dst:
            os.makedirs(dst, exist_ok=True)
            for pathRoot, folders, files in os.walk(src):
                middle = os.path.relpath(pathRoot, src)
                pathRootDst = os.path.join(dst, middle)

                if len(folders) > 0:
                    for folder in folders:
                        path = os.path.join(pathRootDst, folder)
                        os.makedirs(path, exist_ok=True)
                for file in files:
                    pathFile = os.path.join(pathRoot, file)
                    pcv.params.debug = None
                    tr = ImageTransformation(pathFile, outDir=pathRootDst, save=True)
                    showTransformation(
                        tr,
                        origine,
                        mask,
                        gaussian,
                        roi,
                        analyze,
                        pseudolandmarks,
                        color,
                    )
            shutil.make_archive("transformation", format="zip", root_dir=dst)

    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

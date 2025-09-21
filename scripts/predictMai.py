"""
predict.py

A program that generates a prediction based on a given model on an image.
Or checks the accuracy on an image folder

Use:
`$> uv run scripts/Transformation.py --help`
`$> uv run scripts/predict.py -base_dataset zip/images_augmented/Apple -model model.pt images/Apple -validation`
`$> uv run uv run scripts/predict.py -base_dataset zip/images_augmented/Apple -model model.pt images/Apple/Apple_Black_rot/image\ \(2\).JPG`

"""
import os
import sys

import click
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
from utils.logger import get_logger
from utils.model import LeaflictionCNN
import torch.nn.functional as F
import torchvision
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

train_transform = transforms.Compose(
    [
        transforms.Resize(
            (256, 256)
        ),
        transforms.ToTensor(),
    ]
)

logger = get_logger(__name__)

def get_classes(dir):
    dataset = torchvision.datasets.ImageFolder(dir)
    return dataset.classes

def get_mask_transform(img):
    b = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    b_thresh = pcv.threshold.otsu(gray_img=b, object_type='light')
    mask = pcv.fill_holes(bin_img=b_thresh)
    apply_mask = pcv.apply_mask(img=img, mask=mask, mask_color='white')
    return apply_mask

def predictionValidation(srcDir, CNNModel, baseDataset):
    classes = get_classes(baseDataset)
    validation_data = torchvision.datasets.ImageFolder(root=srcDir, transform=train_transform)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_data)
    model = LeaflictionCNN(len(classes))
    model.load_state_dict(torch.load(CNNModel))
    model.eval()
    total=0
    correct=0
    for images, labels in validation_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on validation images: %.6f%%' % (100.0*correct/total))
    pass

def figureShow(img, apply_mask, res):
    fig = plt.figure(figsize=(8, 4))
    fig.suptitle(f'Class predicted : {res}', fontsize=16)
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    fig.add_subplot(1, 2, 2)
    plt.imshow(apply_mask)
    plt.show()

def prediciton(srcImage, CNNModel, baseDataset):
    classes = get_classes(baseDataset)
    print(classes)
    model = LeaflictionCNN(len(classes))
    model.load_state_dict(torch.load(CNNModel))
    model.eval()

    img, _, _ = pcv.readimage(srcImage, mode="rgba")
    apply_mask = get_mask_transform(img=img)

    image = Image.open(srcImage)
    input_tensor = train_transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = F.softmax(output, dim=1)
    print(probabilities)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    res = classes[predicted_class]

    figureShow(img, apply_mask, res)

@click.command()
@click.option('-validation', is_flag=True, help='check prediction')
@click.option('-base_dataset', type=click.Path(exists=True, dir_okay=True, writable=True), required=True)
@click.option('-model', type=click.Path(exists=True, dir_okay=False, writable=True), required=True)
@click.argument('image_to_predict', type=(click.Path(exists=True, writable=True)), required=True)
def main(validation, base_dataset, model, image_to_predict):
    try :
        if validation and os.path.isdir(image_to_predict):
            predictionValidation(image_to_predict, model, base_dataset)
        elif (not validation) and os.path.isfile(image_to_predict):
            prediciton(image_to_predict, model, base_dataset)
        else:
            click.echo("Usage: predict.py -base_dataset DIR_BASE_DATA_SET -model PATH_FILE_MODEL IMAGE_TO_PREDICT")

    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
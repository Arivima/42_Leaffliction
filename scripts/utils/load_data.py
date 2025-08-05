import torch
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import Counter


def get_loaders(
    data_dir, batch_size, img_size, train_transform, val_transform, use_sampler=False
):
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    if use_sampler:
        labels = [full_dataset[i][1] for i in range(len(full_dataset))]
        class_counts = Counter(labels)
        weights = 1.0 / torch.tensor(
            [class_counts[label] for label in labels], dtype=torch.float
        )
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, class_names

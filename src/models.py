import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = [0.0967, 0.1195, 0.1397]
STD = [0.1158, 0.1421, 0.1677]


class CustomImageDataset(Dataset):
    def __init__(
        self,
        mapping,
        root_dir=os.path.join("..", "data"),
        files_dir=None,
        transform=None,
    ):
        self.root_dir = root_dir
        if files_dir is None:
            files_dir = os.path.join(root_dir, "processed")
        self.files_dir = files_dir

        self.mapping = pd.read_csv(os.path.join(root_dir, mapping))
        self.transform = transform

        self.image_paths = [
            os.path.join(self.files_dir, f"{img_id}.jpg")
            for img_id in self.mapping["asset_id"]
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        image = transforms.ToTensor()(image)
        if self.transform:
            image = self.transform(image)

        return image


def compute_mean_std(dataloader):
    mean = 0.0
    std = 0.0
    for images in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)
    return mean, std


def get_transform(size=None, train=False):
    transformers = [
        transforms.Normalize(mean=MEAN, std=STD),
    ]

    if train:
        transformers.extend(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomRotation(degrees=(-180, 180)),
            ]
        )

    if size is not None:
        transformers.append(transforms.Resize((size, size)))

    return transforms.Compose(transformers)


def reverse_transform(tensor):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.numpy()[0]

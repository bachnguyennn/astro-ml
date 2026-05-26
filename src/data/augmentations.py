"""Augmentation pipelines.

Key design choice: the night sky has *no canonical orientation*, so we
use full 180° random rotation (and flips). Doing this with naive
ImageNet augs (which only flip horizontally) would leak orientation
priors that don't exist in the data.
"""

from __future__ import annotations

from torchvision import transforms


# Night-sky-tuned normalization. Computed from a representative batch of
# rendered images; refresh if you change the renderer.
NIGHT_SKY_MEAN = (0.10, 0.10, 0.15)
NIGHT_SKY_STD = (0.15, 0.15, 0.20)


def build_train_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NIGHT_SKY_MEAN, std=NIGHT_SKY_STD),
    ])


def build_eval_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NIGHT_SKY_MEAN, std=NIGHT_SKY_STD),
    ])

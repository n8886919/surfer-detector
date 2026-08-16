from __future__ import annotations

import io
import random

import torch
from PIL import Image
from torchvision import transforms


class RandomJpegCompression:
    def __init__(self, probability: float = 0.25, quality: tuple[int, int] = (35, 90)):
        self.probability = probability
        self.quality = quality

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(*self.quality))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


class RandomGaussianNoise:
    def __init__(self, probability: float = 0.15, sigma: tuple[float, float] = (0.005, 0.04)):
        self.probability = probability
        self.sigma = sigma

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return image
        sigma = random.uniform(*self.sigma)
        return (image + torch.randn_like(image) * sigma).clamp(0, 1)


def action_transform(*, training: bool, image_size: int = 160):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if not training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=12,
                translate=(0.10, 0.10),
                scale=(0.82, 1.18),
                shear=(-4, 4, -2, 2),
            )
        ], p=0.7),
        transforms.RandomPerspective(distortion_scale=0.12, p=0.12),
        transforms.ColorJitter(brightness=0.45, contrast=0.40, saturation=0.35, hue=0.035),
        transforms.RandomGrayscale(p=0.06),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.2, 1.8))], p=0.18),
        transforms.RandomAdjustSharpness(sharpness_factor=0.3, p=0.10),
        RandomJpegCompression(),
        transforms.ToTensor(),
        RandomGaussianNoise(),
        transforms.RandomErasing(p=0.12, scale=(0.02, 0.10), ratio=(0.4, 2.5), value="random"),
        normalize,
    ])


DETECTOR_AUGMENTATION = {
    "hsv_h": 0.025,
    "hsv_s": 0.55,
    "hsv_v": 0.45,
    "degrees": 12.0,
    "translate": 0.15,
    "scale": 0.65,
    "shear": 4.0,
    "perspective": 0.0008,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 0.8,
    "mixup": 0.08,
    "cutmix": 0.05,
    "close_mosaic": 5,
}

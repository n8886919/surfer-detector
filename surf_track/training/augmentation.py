from __future__ import annotations

import io
import math
import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2


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


class RandomResolution:
    """Downsample the crop so the final Resize has to upscale it back.

    Capture distance is uncontrolled — drone, shore, any lens — but 96.8% of the crops in
    the dataset are *downsampled* to 160 and none is smaller than 64 px, so the model has
    never seen a distant surfer. Every other degradation here runs at the final 160 px,
    which does not reproduce the signal of a genuinely low-resolution capture.

    The 48 px floor is where a 30 px-tall body lands after the 1.55 padding; log-uniform
    keeps the median near 88 px rather than piling up at the aggressive end.
    """

    def __init__(self, probability: float = 0.5, target: tuple[int, int] = (48, 160)):
        self.probability = probability
        self.target = target

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        low, high = self.target
        side = round(math.exp(random.uniform(math.log(low), math.log(high))))
        width, height = image.size
        scale = side / max(width, height)
        if scale >= 1:
            return image
        # Kernel varies because deployment may scale on the VIC or on the GPU.
        kernel = random.choice((Image.BILINEAR, Image.BICUBIC, Image.BOX))
        return image.resize((max(8, round(width * scale)), max(8, round(height * scale))), kernel)


class RandomGaussianNoise:
    def __init__(self, probability: float = 0.15, sigma: tuple[float, float] = (0.005, 0.04)):
        self.probability = probability
        self.sigma = sigma

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return image
        sigma = random.uniform(*self.sigma)
        return (image + torch.randn_like(image) * sigma).clamp(0, 1)


def action_transform(*, training: bool, image_size: int = 160, degrade_to: int | None = None):
    """`degrade_to` forces a fixed downsample first, for the low-resolution eval tier."""
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if not training:
        steps = []
        if degrade_to:
            steps.append(RandomResolution(probability=1.0, target=(degrade_to, degrade_to)))
        steps += [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
        return transforms.Compose(steps)

    return transforms.Compose([
        # Margin jitter: at deployment the crop comes from a detector box, so the 1.55
        # padding will not be exact. This owns the scale/translate axes now.
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=12,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                shear=(-4, 4, -2, 2),
            )
        ], p=0.7),
        transforms.RandomPerspective(distortion_scale=0.12, p=0.12),
        transforms.ColorJitter(brightness=0.45, contrast=0.40, saturation=0.35, hue=0.035),
        transforms.RandomGrayscale(p=0.06),
        # Physical capture order: optics blur, then sensor sampling, then encoding. JPEG
        # must come after the downsample so its 8x8 blocks land on the source grid.
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.2, 1.8))], p=0.18),
        transforms.RandomAdjustSharpness(sharpness_factor=0.3, p=0.10),
        RandomResolution(),
        RandomJpegCompression(probability=0.30, quality=(30, 90)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        RandomGaussianNoise(),
        transforms.RandomErasing(p=0.12, scale=(0.02, 0.10), ratio=(0.4, 2.5), value="random"),
        normalize,
    ])


def detector_transform(*, training: bool):
    """Box-aware whole-frame augmentation for the detector, on `transforms.v2`.

    `SanitizeBoundingBoxes` is not hygiene, it is mandatory: every geometric step clamps
    boxes to the canvas, a clamped box can come out with zero width, and FCOS asserts on
    that inside the loss. It runs in the eval pipeline too, because a hand-drawn box of zero
    height would break evaluation the same way.

    Scale jitter has to change box size *relative to the canvas*. The detector's own
    `GeneralizedRCNNTransform` rescales every frame to 896x512, so anything that resizes the
    whole frame (`ScaleJitter`, `RandomShortestSize`) is undone before the backbone sees it —
    only zoom-out padding and affine scaling survive. That axis matters more than the rest
    here: the camera zooms out to survey the wave and in to film the ride, so the same surfer
    arrives at very different pixel sizes.

    Translation stays small on purpose. A surfer shifted half out of frame keeps recognisable
    pixels while their box gets clamped to a sliver, which teaches exactly the false negative
    the tracker cannot afford.
    """
    if not training:
        return v2.Compose([
            v2.ToImage(),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),
            v2.ToPureTensor(),
        ])

    return v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomZoomOut(fill=0, side_range=(1.0, 1.7), p=0.3),
        v2.RandomAffine(degrees=5, translate=(0.04, 0.04), scale=(0.85, 1.15)),
        v2.RandomPhotometricDistort(p=0.5),
        v2.RandomGrayscale(p=0.05),
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.2, 1.5))], p=0.15),
        # Last of the pixel steps, so the 8x8 blocks land on the grid the model actually
        # sees rather than on a grid a later resize would smear.
        v2.JPEG(quality=(45, 95)),
        v2.SanitizeBoundingBoxes(),
        # Detection models want float [0, 1] and normalise internally; pure tensors keep the
        # tv_tensor subclasses out of the model's own transform.
        v2.ToDtype(torch.float32, scale=True),
        v2.ToPureTensor(),
    ])

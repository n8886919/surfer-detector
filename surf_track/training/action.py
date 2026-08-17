from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small

from surf_track.training.augmentation import action_transform


ACTION_NAMES = ("chasing_wave", "takeoff", "surfing")


def select_device() -> torch.device:
    """CUDA when available; `SURF_TRACK_DEVICE` overrides for debugging or CPU-only hosts."""
    override = os.environ.get("SURF_TRACK_DEVICE")
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crop_for_sample(image: Image.Image, sample: dict[str, object]) -> Image.Image:
    """The padded body crop an action model sees for one annotation."""
    width, height = image.size
    box_width = float(sample["width"]) * width
    box_height = float(sample["height"]) * height
    center_x = (float(sample["x"]) + float(sample["width"]) / 2) * width
    center_y = (float(sample["y"]) + float(sample["height"]) / 2) * height
    side = max(box_width, box_height) * 1.55
    left = max(0, center_x - side / 2)
    top = max(0, center_y - side / 2)
    right = min(width, center_x + side / 2)
    bottom = min(height, center_y + side / 2)
    return image.crop((left, top, right, bottom))


class ActionCropDataset(Dataset):
    def __init__(self, samples: list[dict[str, object]], *, training: bool):
        self.samples = samples
        self.transform = action_transform(training=training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(Path(sample["path"])) as source:
            crop = crop_for_sample(source.convert("RGB"), sample)
        return self.transform(crop), torch.tensor(sample["labels"], dtype=torch.float32)


def f1_scores(targets: torch.Tensor, logits: torch.Tensor) -> tuple[list[float], list[int]]:
    """Per-class F1 plus how many positives each class actually had in this split.

    A class with no positives scores 0.0, which reads as total failure but really means
    "not measurable". Callers must report the counts so the two can be told apart. Never
    return a non-finite value: these end up in metrics_json, and `json.dumps` would emit a
    bare `NaN` that breaks `JSON.parse` in the Train page.
    """
    predictions = logits.sigmoid() >= 0.5
    truth = targets >= 0.5
    scores: list[float] = []
    positives: list[int] = []
    for index in range(len(ACTION_NAMES)):
        predicted = predictions[:, index]
        actual = truth[:, index]
        true_positive = int((predicted & actual).sum())
        false_positive = int((predicted & ~actual).sum())
        false_negative = int((~predicted & actual).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
        positives.append(int(actual.sum()))
    return scores, positives


def macro_f1(scores: list[float], positives: list[int]) -> float:
    """Average only over classes that had positives, so an absent class cannot drag it down."""
    measured = [score for score, count in zip(scores, positives) if count]
    return sum(measured) / len(measured) if measured else 0.0


def predict_action_samples(samples: list[dict[str, object]], model_path: Path) -> list[list[float]]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(ACTION_NAMES))
    model.load_state_dict(checkpoint["model"])
    device = select_device()
    model.to(device)
    model.eval()
    loader = DataLoader(ActionCropDataset(samples, training=False), batch_size=32, num_workers=0)
    probabilities: list[list[float]] = []
    with torch.inference_mode():
        for images, _targets in loader:
            probabilities.extend(model(images.to(device)).sigmoid().tolist())
    return probabilities


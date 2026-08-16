from __future__ import annotations

import copy
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from surf_track.training.augmentation import action_transform


ACTION_NAMES = ("chasing_wave", "takeoff", "surfing")


class ActionCropDataset(Dataset):
    def __init__(self, samples: list[dict[str, object]], *, training: bool):
        self.samples = samples
        self.transform = action_transform(training=training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(Path(sample["path"])) as source:
            image = source.convert("RGB")
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
            crop = image.crop((left, top, right, bottom))
        return self.transform(crop), torch.tensor(sample["labels"], dtype=torch.float32)


def _f1_scores(targets: torch.Tensor, logits: torch.Tensor) -> list[float]:
    predictions = logits.sigmoid() >= 0.5
    truth = targets >= 0.5
    scores: list[float] = []
    for index in range(len(ACTION_NAMES)):
        predicted = predictions[:, index]
        actual = truth[:, index]
        true_positive = int((predicted & actual).sum())
        false_positive = int((predicted & ~actual).sum())
        false_negative = int((~predicted & actual).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return scores


def predict_action_samples(samples: list[dict[str, object]], model_path: Path) -> list[list[float]]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(ACTION_NAMES))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(ActionCropDataset(samples, training=False), batch_size=32, num_workers=0)
    probabilities: list[list[float]] = []
    with torch.inference_mode():
        for images, _targets in loader:
            probabilities.extend(model(images).sigmoid().tolist())
    return probabilities


def train_action_model(
    samples: list[dict[str, object]],
    output_path: Path,
    *,
    epochs: int,
    on_epoch,
) -> dict[str, float]:
    random.seed(42)
    torch.manual_seed(42)
    grouped = {split: [sample for sample in samples if sample["split"] == split] for split in ("train", "valid", "test")}
    if len(grouped["train"]) < 12 or len(grouped["valid"]) < 3:
        raise ValueError("至少需要 12 個 train 框和 3 個 valid 框才能開始動作訓練。")

    train_loader = DataLoader(ActionCropDataset(grouped["train"], training=True), batch_size=16, shuffle=True, num_workers=0)
    valid_loader = DataLoader(ActionCropDataset(grouped["valid"], training=False), batch_size=32, num_workers=0)

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for block in model.features[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(ACTION_NAMES))

    positives = torch.tensor([
        sum(int(sample["labels"][index]) for sample in grouped["train"])
        for index in range(len(ACTION_NAMES))
    ], dtype=torch.float32)
    negatives = len(grouped["train"]) - positives
    loss_function = nn.BCEWithLogitsLoss(pos_weight=(negatives / positives.clamp_min(1)).clamp(max=12))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=8e-4, weight_decay=1e-4)

    best_score = -1.0
    best_state = None
    best_metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(images)

        model.eval()
        all_targets: list[torch.Tensor] = []
        all_logits: list[torch.Tensor] = []
        with torch.inference_mode():
            for images, targets in valid_loader:
                all_targets.append(targets)
                all_logits.append(model(images))
        scores = _f1_scores(torch.cat(all_targets), torch.cat(all_logits))
        metrics = {
            "train_loss": loss_total / len(grouped["train"]),
            "action_macro_f1": sum(scores) / len(scores),
            **{f"{name}_f1": score for name, score in zip(ACTION_NAMES, scores)},
            "train_boxes": float(len(grouped["train"])),
            "valid_boxes": float(len(grouped["valid"])),
        }
        if metrics["action_macro_f1"] >= best_score:
            best_score = metrics["action_macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
        on_epoch(epoch, metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best_state,
        "architecture": "mobilenet_v3_small",
        "actions": ACTION_NAMES,
        "image_size": 160,
        "metrics": best_metrics,
    }, output_path)
    return best_metrics

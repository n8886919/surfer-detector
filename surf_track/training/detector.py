"""Single-class surfer detector: a MobileNetV3-Large FPN backbone with an FCOS head.

Sized for the Jetson Orin Nano rather than for COCO leaderboards. Measured at 896x512:
14.79 GMAC, 5.0M params, 9548 candidate locations, and a plugin-free ONNX export once the
postprocessing is stripped. See README's "Orin 部署契約" for why 896x512 and not 640 or
1024 — strides must divide evenly on both axes or the DeepStream parser mis-decodes.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.fcos import FCOS, FCOSHead
from torchvision.ops import FrozenBatchNorm2d, box_iou
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

INPUT_WIDTH = 896
INPUT_HEIGHT = 512
FPN_CHANNELS = 128
HEAD_CONVS = 4
# FCOS treats the label as a channel index into [N, HWA, num_classes] and marks background
# as -1, so the single foreground class must be 0. A 1 would index out of bounds.
SURFER_LABEL = 0


def build_detector(*, trainable_stages: int = 2, weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2):
    """FCOS over strides 8/16/32/64/128.

    `trainable_stages` counts backbone stages left unfrozen, from the deepest: 0 freezes the
    whole backbone, 5 trains all of it. With ~1200 boxes, freezing pays — start at 0 for the
    first epochs to settle the head, then continue at 2. Never train the stem.

    `norm_layer=FrozenBatchNorm2d` is not optional: with live BatchNorm the 6GB training GPU
    forces a micro-batch small enough to corrupt the running statistics.
    `_mobilenet_extractor`'s defaults are deliberately avoided — its finest stride is 32,
    which leaves only ~540 locations for boxes whose median height is 87 px.
    """
    features = mobilenet_v3_large(weights=weights_backbone, norm_layer=FrozenBatchNorm2d).features
    # Stage boundaries, the same way torchvision finds them internally.
    boundaries = [0] + [i for i, block in enumerate(features) if getattr(block, "_is_cn", False)]
    boundaries.append(len(features) - 1)
    picked = [boundaries[index] for index in (-4, -3, -1)]

    stage_starts = boundaries[:-1]
    if not 0 <= trainable_stages <= len(stage_starts):
        raise ValueError(f"trainable_stages must be 0..{len(stage_starts)}")
    frozen_until = (
        len(features)
        if trainable_stages == 0
        else stage_starts[len(stage_starts) - trainable_stages]
    )
    for block in features[:frozen_until]:
        for parameter in block.parameters():
            parameter.requires_grad_(False)

    backbone = BackboneWithFPN(
        features,
        {str(index): str(level) for level, index in enumerate(picked)},
        [features[index].out_channels for index in picked],
        FPN_CHANNELS,
        extra_blocks=LastLevelP6P7(FPN_CHANNELS, FPN_CHANNELS),
    )
    model = FCOS(
        backbone,
        num_classes=1,
        min_size=INPUT_HEIGHT,
        max_size=INPUT_WIDTH,
        score_thresh=0.05,
        nms_thresh=0.6,
        detections_per_img=20,
    )
    # Default head assumes 256 channels; rebuild it at the FPN width actually used.
    model.head = FCOSHead(
        FPN_CHANNELS,
        model.anchor_generator.num_anchors_per_location()[0],
        1,
        num_convs=HEAD_CONVS,
    )
    return model


def load_detector(model_path):
    """Rebuild the trained graph from the checkpoint's own record of how it was built."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = build_detector(
        trainable_stages=int(checkpoint["trainable_stages"]), weights_backbone=None,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def predict_detector_images(
    paths: list, model_path, *, score_threshold: float = 0.5, batch_size: int = 4,
) -> list[list[dict[str, float]]]:
    """Detect surfers in whole frames, as normalised xywh boxes per frame.

    Frames are downscaled the same way the training feeder downscales them — `draft` inside
    the JPEG decoder, then an exact `thumbnail` — so the preview sees the pixels the model
    was trained on rather than a 4K frame the model's own transform would rescale differently.
    """
    from PIL import Image
    from torchvision.transforms.v2.functional import to_dtype, to_image

    from surf_track.training.action import select_device

    model, _ = load_detector(model_path)
    device = select_device()
    model.to(device)

    results: list[list[dict[str, float]]] = []
    for start in range(0, len(paths), batch_size):
        frames = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as source:
                source.draft("RGB", (INPUT_WIDTH, INPUT_HEIGHT))
                frame = source.convert("RGB")
                frame.thumbnail((INPUT_WIDTH, INPUT_HEIGHT))
            frames.append(frame)
        batch = [to_dtype(to_image(frame), torch.float32, scale=True).to(device) for frame in frames]
        # no_grad, not inference_mode: the boxes are post-processed on the way out and
        # inference tensors forbid the in-place writes that costs.
        with torch.no_grad():
            detections = model(batch)
        for frame, found in zip(frames, detections):
            width, height = frame.size
            keep = found["scores"] >= score_threshold
            results.append([
                {
                    "x": float(box[0]) / width,
                    "y": float(box[1]) / height,
                    "width": float(box[2] - box[0]) / width,
                    "height": float(box[3] - box[1]) / height,
                    "score": float(score),
                }
                for box, score in zip(found["boxes"][keep].cpu(), found["scores"][keep].cpu())
            ])
    return results


class RawDetector(nn.Module):
    """backbone + head only — what a DeepStream PGIE engine should contain.

    torchvision bakes score filtering and NMS into forward, which puts If/NonZero/TopK in
    the exported graph and TensorRT cannot build that statically. Training and local
    evaluation keep using the full model; only the export is truncated.
    """

    def __init__(self, model: FCOS):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, images: torch.Tensor):
        features = self.backbone(images)
        if isinstance(features, dict):
            features = list(features.values())
        outputs = self.head(features)
        return tuple(outputs.values()) if isinstance(outputs, dict) else outputs


# --- metrics -------------------------------------------------------------------------
# Single class, so pycocotools and torchmetrics would both be new dependencies for work
# that fits in a screenful. torchvision.ops.box_iou is already available.

def _greedy_matches(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_threshold: float,
    score_threshold: float,
) -> tuple[list[float], list[float], int]:
    """Score-ordered hit/miss flags, plus the ground-truth count.

    Each ground-truth box may be claimed once, by the highest-scoring detection that
    reaches the IoU threshold — the usual VOC/COCO rule.
    """
    scored: list[tuple[float, int, torch.Tensor]] = []
    for index, prediction in enumerate(predictions):
        boxes = prediction["boxes"]
        for position, score in enumerate(prediction["scores"].tolist()):
            if score >= score_threshold:
                scored.append((score, index, boxes[position]))
    scored.sort(key=lambda item: -item[0])

    claimed = [torch.zeros(len(target["boxes"]), dtype=torch.bool) for target in targets]
    hits: list[float] = []
    scores: list[float] = []
    for score, index, box in scored:
        truth = targets[index]["boxes"]
        scores.append(score)
        if len(truth) == 0:
            hits.append(0.0)
            continue
        overlaps = box_iou(box.unsqueeze(0).to(truth.dtype), truth)[0].clone()
        overlaps[claimed[index]] = -1.0
        best = int(overlaps.argmax())
        if float(overlaps[best]) >= iou_threshold:
            claimed[index][best] = True
            hits.append(1.0)
        else:
            hits.append(0.0)
    return hits, scores, sum(len(target["boxes"]) for target in targets)


def ap50(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    *,
    iou_threshold: float = 0.5,
) -> float:
    """All-point-interpolation average precision at one IoU threshold."""
    hits, _, total = _greedy_matches(predictions, targets, iou_threshold, 0.0)
    if total == 0 or not hits:
        return 0.0
    flags = torch.tensor(hits)
    true_positives = flags.cumsum(0)
    false_positives = (1 - flags).cumsum(0)
    recall = true_positives / total
    precision = true_positives / (true_positives + false_positives)
    # Precision envelope, then integrate over recall steps.
    envelope = precision.flip(0).cummax(0).values.flip(0)
    recall = torch.cat([torch.zeros(1), recall])
    return float(((recall[1:] - recall[:-1]) * envelope).sum())


def recall_at(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    *,
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> float:
    """Recall at the confidence the pipeline will actually deploy at.

    This is the number the tracker cares about: a miss is a gap in a track's identity.
    """
    hits, _, total = _greedy_matches(predictions, targets, iou_threshold, score_threshold)
    return float(sum(hits) / total) if total else 0.0


def false_positives_per_image(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    *,
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> float:
    """Spurious detections per frame — these become tracker ID churn and wasted SGIE crops."""
    hits, _, _ = _greedy_matches(predictions, targets, iou_threshold, score_threshold)
    return float(len(hits) - sum(hits)) / len(predictions) if predictions else 0.0


def small_box_recall(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    *,
    score_threshold: float,
    max_height: float = 32.0,
    iou_threshold: float = 0.5,
) -> float:
    """Recall restricted to short boxes, where the input-resolution choice pays off.

    Only ground truth is filtered; predictions still compete freely, so a detection that
    lands on a tall box simply fails to match and is ignored rather than counted against.
    """
    slim = [
        {"boxes": target["boxes"][(target["boxes"][:, 3] - target["boxes"][:, 1]) < max_height]}
        for target in targets
    ]
    return recall_at(
        predictions, slim, score_threshold=score_threshold, iou_threshold=iou_threshold
    )


def evaluate(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    *,
    score_threshold: float = 0.5,
) -> dict[str, float]:
    """The four numbers worth reporting, keyed for training_runs.metrics_json.

    Counts ship alongside the rates for the same reason f1_scores reports positives: a
    recall of 0.0 over zero eligible boxes means "not measurable", not "failed".
    """
    small = sum(
        int(((target["boxes"][:, 3] - target["boxes"][:, 1]) < 32.0).sum()) for target in targets
    )
    return {
        "detector_map50": ap50(predictions, targets),
        "detector_recall": recall_at(predictions, targets, score_threshold=score_threshold),
        "detector_fp_per_image": false_positives_per_image(
            predictions, targets, score_threshold=score_threshold
        ),
        "detector_small_recall": small_box_recall(
            predictions, targets, score_threshold=score_threshold
        ),
        "detector_small_boxes": float(small),
        "detector_score_threshold": score_threshold,
        "detector_boxes": float(sum(len(target["boxes"]) for target in targets)),
        "detector_images": float(len(targets)),
    }

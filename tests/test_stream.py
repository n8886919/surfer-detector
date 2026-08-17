from __future__ import annotations

import pytest

from surf_track.training.stream import StreamError, validate_host


@pytest.mark.parametrize("host", [
    "nolan@192.168.31.128",
    "user@host",
    "a.b+c@example.com",
    "_svc@10.0.0.1",
])
def test_valid_hosts_are_accepted(host: str) -> None:
    assert validate_host(host) == host


def test_surrounding_whitespace_is_trimmed() -> None:
    assert validate_host("  nolan@1.2.3.4  ") == "nolan@1.2.3.4"


@pytest.mark.parametrize("host", [
    # ssh has no "--" terminator, so a leading dash would be parsed as an option and
    # -oProxyCommand=... would run an arbitrary command on this machine.
    "-oProxyCommand=touch /tmp/pwned@host",
    "-o@host",
    "-nolan@1.2.3.4",
    "host-without-user",
    "user@",
    "@host",
    "",
    "   ",
    "user@host with space",
    "user@host;touch /tmp/pwned",
    None,
    123,
])
def test_unsafe_or_malformed_hosts_are_rejected(host: object) -> None:
    with pytest.raises(StreamError):
        validate_host(host)


def test_training_transform_always_returns_160_after_resolution_jitter() -> None:
    pytest.importorskip("torch")
    from PIL import Image

    from surf_track.training.augmentation import action_transform

    transform = action_transform(training=True)
    source = Image.effect_noise((160, 160), 60).convert("RGB")
    # RandomResolution fires at p=0.5, so this exercises both branches.
    for _ in range(30):
        assert tuple(transform(source).shape) == (3, 160, 160)


def test_lowres_eval_tier_degrades_detail_but_keeps_the_shape() -> None:
    pytest.importorskip("torch")
    from PIL import Image

    from surf_track.training.augmentation import action_transform

    source = Image.effect_noise((160, 160), 60).convert("RGB")
    clean = action_transform(training=False)(source)
    degraded = action_transform(training=False, degrade_to=64)(source)
    assert tuple(degraded.shape) == tuple(clean.shape) == (3, 160, 160)

    def high_frequency_energy(tensor) -> float:
        return float((tensor[:, :, 1:] - tensor[:, :, :-1]).abs().mean())

    assert high_frequency_energy(degraded) < high_frequency_energy(clean)


def test_f1_scores_report_positives_and_never_serialise_as_nan() -> None:
    torch = pytest.importorskip("torch")
    import json
    import math

    from surf_track.training.action import f1_scores, macro_f1

    # chasing_wave has no positives here, so its F1 is not measurable rather than zero.
    targets = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    logits = torch.tensor([[-2.0, 2.0, 2.0], [-2.0, -2.0, 2.0]])
    scores, positives = f1_scores(targets, logits)

    assert positives == [0, 1, 2]
    assert all(math.isfinite(score) for score in scores)
    # A bare NaN in metrics_json would break JSON.parse on the Train page.
    assert "NaN" not in json.dumps(scores)
    # The unmeasurable class must not drag the macro down with a fake 0.0.
    assert macro_f1(scores, positives) == pytest.approx(1.0)
    assert macro_f1([0.0, 0.0, 0.0], [0, 0, 0]) == 0.0


def _detection(boxes, scores):
    torch = pytest.importorskip("torch")
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.zeros(len(scores), dtype=torch.int64),
    }


def _truth(boxes):
    torch = pytest.importorskip("torch")
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)}


def test_detector_metrics_on_a_hand_checkable_case() -> None:
    pytest.importorskip("torch")
    from surf_track.training.detector import (
        ap50, evaluate, false_positives_per_image, recall_at,
    )

    # Frame 1: one ground truth, hit at 0.9. Frame 2: one missed, plus a 0.8 false positive.
    predictions = [_detection([10, 10, 50, 90], [0.9]), _detection([200, 200, 240, 280], [0.8])]
    targets = [_truth([10, 10, 50, 90]), _truth([600, 300, 640, 380])]

    assert ap50(predictions, targets) == pytest.approx(0.5)
    assert recall_at(predictions, targets, score_threshold=0.5) == pytest.approx(0.5)
    assert false_positives_per_image(predictions, targets, score_threshold=0.5) == pytest.approx(0.5)
    # Raising the threshold drops the false positive without losing the hit.
    assert recall_at(predictions, targets, score_threshold=0.85) == pytest.approx(0.5)
    assert false_positives_per_image(predictions, targets, score_threshold=0.85) == pytest.approx(0.0)

    assert ap50(predictions[:1], targets[:1]) == pytest.approx(1.0)
    assert ap50([_detection([], [])], targets[:1]) == pytest.approx(0.0)
    assert evaluate(predictions, targets)["detector_boxes"] == 2.0


def test_small_box_recall_only_counts_short_ground_truth() -> None:
    pytest.importorskip("torch")
    from surf_track.training.detector import evaluate

    predictions = [_detection([10, 10, 50, 36], [0.9]), _detection([], [])]
    targets = [_truth([10, 10, 50, 35]), _truth([200, 100, 240, 180])]  # 25 px and 80 px tall
    metrics = evaluate(predictions, targets)

    assert metrics["detector_recall"] == pytest.approx(0.5)
    assert metrics["detector_small_recall"] == pytest.approx(1.0)
    # The count travels with the rate so a 0.0 over no eligible boxes reads as "not measurable".
    assert metrics["detector_small_boxes"] == 1.0


def test_detector_trains_with_single_class_labels_and_empty_frames() -> None:
    torch = pytest.importorskip("torch")
    from surf_track.training.detector import (
        INPUT_HEIGHT, INPUT_WIDTH, SURFER_LABEL, build_detector,
    )

    model = build_detector(weights_backbone=None)
    model.train()
    images = [torch.rand(3, INPUT_HEIGHT, INPUT_WIDTH), torch.rand(3, INPUT_HEIGHT, INPUT_WIDTH)]
    targets = [
        {"boxes": torch.tensor([[100.0, 80.0, 240.0, 260.0]]),
         "labels": torch.full((1,), SURFER_LABEL, dtype=torch.int64)},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
    ]
    losses = model(images, targets)
    assert set(losses) == {"classification", "bbox_regression", "bbox_ctrness"}
    sum(losses.values()).backward()

    # FCOS indexes the class channel by label, so a 1 would be out of bounds for one class.
    with pytest.raises(IndexError):
        model(images, [
            {"boxes": torch.tensor([[100.0, 80.0, 240.0, 260.0]]),
             "labels": torch.ones(1, dtype=torch.int64)},
            targets[1],
        ])


def test_backbone_freezing_is_bounded_and_actually_freezes() -> None:
    pytest.importorskip("torch")
    from surf_track.training.detector import build_detector

    def trainable_fraction(stages):
        body = build_detector(trainable_stages=stages, weights_backbone=None).backbone.body
        total = sum(p.numel() for p in body.parameters())
        return sum(p.numel() for p in body.parameters() if p.requires_grad) / total

    assert trainable_fraction(0) == 0.0
    assert trainable_fraction(5) == 1.0
    assert trainable_fraction(1) < trainable_fraction(2) < trainable_fraction(5)
    with pytest.raises(ValueError):
        build_detector(trainable_stages=9, weights_backbone=None)

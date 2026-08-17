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

from __future__ import annotations

import random
from collections.abc import Iterable


def assign_group_splits(
    group_ids: Iterable[str],
    *,
    seed: int = 42,
    train_ratio: int = 80,
    valid_ratio: int = 10,
    test_ratio: int = 10,
) -> dict[str, str]:
    """Randomly split source groups while keeping adjacent frames together."""
    if train_ratio + valid_ratio + test_ratio != 100:
        raise ValueError("train/valid/test ratios must total 100")
    if min(train_ratio, valid_ratio, test_ratio) < 0:
        raise ValueError("split ratios cannot be negative")

    unique_groups = sorted({group_id.strip() for group_id in group_ids if group_id.strip()})
    random.Random(seed).shuffle(unique_groups)
    count = len(unique_groups)
    train_count = round(count * train_ratio / 100)
    valid_count = round(count * valid_ratio / 100)
    if train_count + valid_count > count:
        valid_count = max(0, count - train_count)

    assignments: dict[str, str] = {}
    for index, group_id in enumerate(unique_groups):
        if index < train_count:
            split = "train"
        elif index < train_count + valid_count:
            split = "valid"
        else:
            split = "test"
        assignments[group_id] = split
    return assignments

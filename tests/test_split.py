from surf_track.dataset import assign_group_splits


def test_group_split_is_deterministic_and_keeps_groups_together() -> None:
    groups = [f"video-{index:02d}" for index in range(20)]
    first = assign_group_splits(groups, seed=42)
    second = assign_group_splits(reversed(groups), seed=42)

    assert first == second
    assert list(first.values()).count("train") == 16
    assert list(first.values()).count("valid") == 2
    assert list(first.values()).count("test") == 2


def test_duplicate_group_ids_receive_one_split() -> None:
    result = assign_group_splits(["clip-a", "clip-a", "clip-b"], seed=7)
    assert set(result) == {"clip-a", "clip-b"}

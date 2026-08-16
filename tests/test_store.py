from pathlib import Path

import pytest

from surf_track.store import StoreError, SurfTrackStore


def create_test_dataset(store: SurfTrackStore, name: str = "pilot") -> dict[str, object]:
    return store.create_dataset(
        name,
        source_provider="google_drive",
        source_url="https://drive.google.com/drive/folders/1AbCdEfGhijKLMnOP",
        source_title="Shared surf clips",
        sharer_name="測試分享者",
        source_video_count=48,
    )


def test_partial_annotations_are_not_detector_ready(tmp_path: Path) -> None:
    store = SurfTrackStore(tmp_path / "var")
    dataset = create_test_dataset(store)
    image_path = store.media_dir / "frame.jpg"
    image_path.write_bytes(b"not-a-real-jpeg")
    image = store.add_image(dataset["id"], image_path, source_group="video-1", split="train")

    saved = store.save_annotations(
        image["id"],
        [
            {
                "x": 0.1,
                "y": 0.2,
                "width": 0.3,
                "height": 0.4,
                "chasing_wave": True,
            }
        ],
        complete_for_detection=False,
        crop_region={"x": 0.05, "y": 0.1, "width": 0.5, "height": 0.6},
    )

    assert saved["annotation_count"] == 1
    assert saved["complete_for_detection"] is False
    assert saved["crop_region"] == {"x": 0.05, "y": 0.1, "width": 0.5, "height": 0.6}
    assert store.get_annotations(str(image["id"]))["crop_region"] == saved["crop_region"]
    snapshot = store.workspace_snapshot()
    assert snapshot["datasets"][0]["labeled_image_count"] == 1
    assert snapshot["datasets"][0]["detector_ready_count"] == 0
    assert snapshot["datasets"][0]["sharer_name"] == "測試分享者"
    assert snapshot["datasets"][0]["source_video_count"] == 48


def test_runtime_paths_are_private_and_sharer_is_required(tmp_path: Path) -> None:
    store = SurfTrackStore(tmp_path / "var")
    assert store.data_dir.stat().st_mode & 0o777 == 0o700
    assert store.media_dir.stat().st_mode & 0o777 == 0o700
    assert store.secrets_dir.stat().st_mode & 0o777 == 0o700
    assert store.database_path.stat().st_mode & 0o777 == 0o600

    with pytest.raises(StoreError, match="分享者"):
        store.create_dataset(
            "missing attribution",
            source_provider="google_drive",
            source_url="https://drive.google.com/drive/folders/1AbCdEfGhijKLMnOP",
            sharer_name="",
        )


def test_running_job_is_paused_for_resume_after_restart(tmp_path: Path) -> None:
    store = SurfTrackStore(tmp_path / "var")
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO jobs(
                id, job_type, state, step, completed_items, total_items,
                checkpoint_json, created_at, updated_at
            ) VALUES ('job-1', 'ingest', 'running', 'extract', 12, 100, '{"video": 2}', 'now', 'now')
            """
        )

    reopened = SurfTrackStore(tmp_path / "var")
    job = reopened.workspace_snapshot()["jobs"][0]
    assert job["state"] == "paused"
    assert job["completed_items"] == 12
    assert job["checkpoint"] == {"video": 2}


def test_ingest_job_is_created_once_and_can_resume(tmp_path: Path) -> None:
    store = SurfTrackStore(tmp_path / "var")
    dataset = create_test_dataset(store)

    created = store.start_ingest_job(str(dataset["id"]))
    assert created["dataset_id"] == dataset["id"]
    assert created["state"] == "queued"
    assert created["total_items"] == 48
    assert created["checkpoint"] == {
        "completed_video_ids": [],
        "splits": {},
        "frame_counts": {},
    }

    store.update_job(
        str(created["id"]),
        state="failed",
        step="download",
        completed_items=3,
        total_items=48,
        message="temporary failure",
        checkpoint={"completed_video_ids": ["video-1"], "splits": {}, "frame_counts": {}},
    )
    resumed = store.start_ingest_job(str(dataset["id"]))
    assert resumed["id"] == created["id"]
    assert resumed["state"] == "queued"
    assert resumed["completed_items"] == 3
    assert resumed["checkpoint"]["completed_video_ids"] == ["video-1"]


def test_add_images_is_idempotent(tmp_path: Path) -> None:
    store = SurfTrackStore(tmp_path / "var")
    dataset = create_test_dataset(store)
    frame_dir = store.media_dir / str(dataset["id"]) / "frames" / "train" / "video-1"
    frame_dir.mkdir(parents=True)
    frames = [frame_dir / "frame-00000001.jpg", frame_dir / "frame-00000002.jpg"]
    for frame in frames:
        frame.write_bytes(b"jpeg")

    assert store.add_images(str(dataset["id"]), frames, source_group="video-1", split="train") == 2
    assert store.add_images(str(dataset["id"]), frames, source_group="video-1", split="train") == 0
    assert len(store.list_label_images(str(dataset["id"]))) == 2


def test_action_samples_and_training_run_are_persisted(tmp_path: Path) -> None:
    store = SurfTrackStore(tmp_path / "var")
    dataset = create_test_dataset(store)
    image_path = store.media_dir / "frame.jpg"
    image_path.write_bytes(b"jpeg")
    image = store.add_image(dataset["id"], image_path, source_group="video-1", split="train")
    store.save_annotations(
        image["id"],
        [{"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4, "surfing": True}],
        complete_for_detection=False,
    )

    samples = store.list_action_samples(str(dataset["id"]))
    assert samples[0]["labels"] == [0, 0, 1]
    assert samples[0]["path"] == image_path

    run = store.create_training_run(str(dataset["id"]), total_epochs=8)
    updated = store.update_training_run(str(run["id"]), state="running", epoch=1, metrics={"train_loss": 0.5})
    assert updated["metrics"] == {"train_loss": 0.5}
    with pytest.raises(StoreError, match="已經在訓練"):
        store.create_training_run(str(dataset["id"]), total_epochs=8)


def test_existing_annotated_images_can_be_marked_detection_ready(tmp_path: Path) -> None:
    store = SurfTrackStore(tmp_path / "var")
    dataset = create_test_dataset(store)
    image_path = store.media_dir / "frame.jpg"
    image_path.write_bytes(b"jpeg")
    image = store.add_image(dataset["id"], image_path, source_group="video-1", split="train")
    store.save_annotations(
        image["id"],
        [{"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4, "surfing": True}],
        complete_for_detection=False,
    )

    assert store.mark_annotated_images_detection_ready(str(dataset["id"])) == 1
    assert store.get_annotations(str(image["id"]))["complete_for_detection"] is True

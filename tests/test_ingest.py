from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from surf_track.config import Settings
from surf_track.ingest import IngestManager
from surf_track.store import SurfTrackStore


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_ingest_extracts_registers_and_completes_without_network(tmp_path: Path, monkeypatch) -> None:
    source_video = tmp_path / "source.mp4"
    generated = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=64x48:d=2",
            "-pix_fmt", "yuv420p", str(source_video),
        ],
        capture_output=True,
        check=False,
    )
    assert generated.returncode == 0

    store = SurfTrackStore(tmp_path / "var")
    dataset = store.create_dataset(
        "pilot",
        source_provider="google_drive",
        source_url="https://drive.google.com/drive/folders/1AbCdEfGhijKLMnOP",
        source_title="Shared clips",
        sharer_name="source owner",
        source_video_count=1,
    )
    job = store.start_ingest_job(str(dataset["id"]))
    manager = IngestManager(store, Settings(data_dir=tmp_path / "var"))

    monkeypatch.setattr(
        "surf_track.ingest.GDownDriveClient.scan",
        lambda _self, _link: {
            "videos": [{"id": "1AbCdEfGhijKLMnOP", "name": "clip.mp4"}],
        },
    )

    def fake_download(*, output: str, progress, **_kwargs) -> str:
        shutil.copyfile(source_video, output)
        progress(source_video.stat().st_size, source_video.stat().st_size)
        return output

    monkeypatch.setattr("surf_track.ingest.gdown.download", fake_download)
    monkeypatch.setattr(manager, "_run_rclone", lambda _arguments: None)

    manager._process(str(dataset["id"]), str(job["id"]))

    completed = store.get_job(str(job["id"]))
    images = store.list_label_images(str(dataset["id"]))
    assert completed["state"] == "completed"
    assert completed["completed_items"] == 1
    assert len(images) == 2
    assert {image["split"] for image in images} == {"train"}
    assert (store.media_dir / str(dataset["id"]) / "manifest.json").is_file()
    assert not list((store.media_dir / str(dataset["id"]) / "videos").glob("*.mp4"))

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any

import gdown

from surf_track.config import Settings
from surf_track.dataset.split import assign_group_splits
from surf_track.drive import DriveApiError, GDownDriveClient, parse_drive_link
from surf_track.store import StoreError, SurfTrackStore


VIDEO_EXTENSIONS = {".3gp", ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
DRIVE_ID = re.compile(r"^[A-Za-z0-9_-]{10,}$")


class IngestError(RuntimeError):
    """Safe error that can be shown in the local UI."""


class IngestManager:
    """Run one resumable ingest worker per Dataset."""

    def __init__(self, store: SurfTrackStore, settings: Settings):
        self.store = store
        self.settings = settings
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    def start(self, dataset_id: str) -> dict[str, object]:
        if not self.settings.private_drive_connected():
            raise IngestError("私人 Google Drive 尚未連接。")
        job = self.store.start_ingest_job(dataset_id)
        if job["state"] == "completed":
            return job
        with self._lock:
            running = self._threads.get(dataset_id)
            if running is not None and running.is_alive():
                return job
            thread = threading.Thread(
                target=self._run,
                args=(dataset_id, str(job["id"])),
                name=f"ingest-{dataset_id}",
                daemon=True,
            )
            self._threads[dataset_id] = thread
            thread.start()
        return job

    def _run(self, dataset_id: str, job_id: str) -> None:
        try:
            self._process(dataset_id, job_id)
        except Exception as exc:  # keep the local server alive if a worker fails
            try:
                job = self.store.get_job(job_id)
                message = str(exc) if isinstance(exc, (IngestError, StoreError)) else "處理失敗，請重試。"
                self.store.update_job(
                    job_id,
                    state="failed",
                    step=str(job["step"]),
                    completed_items=int(job["completed_items"]),
                    total_items=int(job["total_items"]),
                    message=message,
                    checkpoint=dict(job["checkpoint"]),
                )
            except Exception:
                pass
        finally:
            with self._lock:
                self._threads.pop(dataset_id, None)

    def _process(self, dataset_id: str, job_id: str) -> None:
        dataset = self.store.get_dataset(dataset_id)
        job = self.store.get_job(job_id)
        checkpoint = dict(job["checkpoint"])
        completed = {str(value) for value in checkpoint.get("completed_video_ids", [])}

        self._update(job_id, checkpoint, "running", "scan", len(completed), int(job["total_items"]), "重新掃描公開來源")
        try:
            result = GDownDriveClient().scan(parse_drive_link(str(dataset["source_url"])))
        except DriveApiError as exc:
            raise IngestError(str(exc)) from exc
        videos = [video for video in result.get("videos", []) if isinstance(video, dict)]
        if not videos:
            raise IngestError("公開來源中沒有可處理的影片。")

        video_ids = [self._video_id(video) for video in videos]
        generated_splits = assign_group_splits(
            video_ids,
            seed=int(dataset["split_seed"]),
            train_ratio=int(dataset["train_ratio"]),
            valid_ratio=int(dataset["valid_ratio"]),
            test_ratio=int(dataset["test_ratio"]),
        )
        splits = {str(key): str(value) for key, value in dict(checkpoint.get("splits", {})).items()}
        for video_id, split in generated_splits.items():
            splits.setdefault(video_id, split)
        checkpoint["splits"] = splits
        checkpoint.setdefault("frame_counts", {})
        total = len(videos)

        for index, video in enumerate(videos, start=1):
            video_id = self._video_id(video)
            if video_id in completed:
                continue
            name = str(video.get("name") or video_id)
            split = splits[video_id]
            video_path = self._download_video(job_id, checkpoint, video, index, total)
            frame_dir = self.store.media_dir / dataset_id / "frames" / split / video_id
            width, height = self._extract_frames(job_id, checkpoint, video_path, frame_dir, name, index, total)
            frames = sorted(frame_dir.glob("frame-*.jpg"))
            if not frames:
                raise IngestError(f"{name} 沒有抽出任何影格。")
            self.store.add_images(
                dataset_id,
                frames,
                source_group=video_id,
                split=split,
                width=width,
                height=height,
            )
            frame_counts = dict(checkpoint.get("frame_counts", {}))
            frame_counts[video_id] = len(frames)
            checkpoint["frame_counts"] = frame_counts
            self._upload_video_frames(job_id, checkpoint, dataset_id, split, video_id, frame_dir, name, index, total)

            completed.add(video_id)
            checkpoint["completed_video_ids"] = sorted(completed)
            manifest_path = self._write_manifest(dataset, checkpoint)
            self._upload_manifest(dataset_id, manifest_path)
            video_path.unlink(missing_ok=True)
            video_path.with_name(f"{video_path.name}.download-complete").unlink(missing_ok=True)
            self._update(job_id, checkpoint, "running", "complete_video", len(completed), total, f"完成 {index}/{total} · {name}")

        self._update(job_id, checkpoint, "completed", "done", total, total, f"完成 · {sum(dict(checkpoint['frame_counts']).values())} 張影格")

    @staticmethod
    def _video_id(video: dict[str, object]) -> str:
        video_id = str(video.get("id") or "")
        if not DRIVE_ID.fullmatch(video_id):
            raise IngestError("來源影片 ID 格式不正確。")
        return video_id

    def _download_video(
        self,
        job_id: str,
        checkpoint: dict[str, object],
        video: dict[str, object],
        index: int,
        total: int,
    ) -> Path:
        video_id = self._video_id(video)
        name = str(video.get("name") or video_id)
        suffix = PurePosixPath(name.lower()).suffix
        if suffix not in VIDEO_EXTENSIONS:
            suffix = ".mp4"
        download_dir = self.store.media_dir / str(self.store.get_job(job_id)["dataset_id"]) / "videos"
        download_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = download_dir / f"{video_id}{suffix}"
        marker = destination.with_name(f"{destination.name}.download-complete")
        if marker.is_file() and destination.is_file() and destination.stat().st_size > 0:
            return destination

        last_update = 0.0

        def progress(current: int, expected: int | None) -> None:
            nonlocal last_update
            now = time.monotonic()
            if now - last_update < 2:
                return
            last_update = now
            current_mb = current / 1024 / 1024
            total_text = f" / {expected / 1024 / 1024:.0f} MB" if expected else ""
            self._update(
                job_id,
                checkpoint,
                "running",
                "download",
                int(self.store.get_job(job_id)["completed_items"]),
                total,
                f"下載 {index}/{total} · {name} · {current_mb:.0f}{total_text}",
            )

        try:
            result = gdown.download(
                id=video_id,
                output=str(destination),
                quiet=True,
                use_cookies=False,
                resume=True,
                progress=progress,
            )
        except Exception as exc:
            raise IngestError(f"下載失敗：{name}") from exc
        if not result or not destination.is_file() or destination.stat().st_size == 0:
            raise IngestError(f"下載失敗：{name}")
        marker.touch()
        return destination

    def _extract_frames(
        self,
        job_id: str,
        checkpoint: dict[str, object],
        video_path: Path,
        frame_dir: Path,
        name: str,
        index: int,
        total: int,
    ) -> tuple[int | None, int | None]:
        frame_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        marker = frame_dir / ".extract-complete"
        width, height = self._probe_dimensions(video_path)
        if marker.is_file():
            return width, height
        self._update(
            job_id,
            checkpoint,
            "running",
            "extract",
            int(self.store.get_job(job_id)["completed_items"]),
            total,
            f"抽圖 {index}/{total} · {name} · 1 FPS",
        )
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(video_path), "-vf", "fps=1", "-q:v", "3",
            str(frame_dir / "frame-%08d.jpg"),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise IngestError(f"抽圖失敗：{name}")
        marker.touch()
        return width, height

    @staticmethod
    def _probe_dimensions(video_path: Path) -> tuple[int | None, int | None]:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "json", str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None, None
        try:
            stream = json.loads(result.stdout)["streams"][0]
            return int(stream["width"]), int(stream["height"])
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            return None, None

    def _upload_video_frames(
        self,
        job_id: str,
        checkpoint: dict[str, object],
        dataset_id: str,
        split: str,
        video_id: str,
        frame_dir: Path,
        name: str,
        index: int,
        total: int,
    ) -> None:
        self._update(
            job_id,
            checkpoint,
            "running",
            "upload",
            int(self.store.get_job(job_id)["completed_items"]),
            total,
            f"上傳 {index}/{total} · {name}",
        )
        remote = f"surftrack-drive:SurfTrackDatasets/{dataset_id}/frames/{split}/{video_id}"
        self._run_rclone(["copy", str(frame_dir), remote, "--include", "*.jpg", "--transfers", "4"])

    def _upload_manifest(self, dataset_id: str, manifest_path: Path) -> None:
        remote = f"surftrack-drive:SurfTrackDatasets/{dataset_id}/manifest.json"
        self._run_rclone(["copyto", str(manifest_path), remote])

    def _run_rclone(self, arguments: list[str]) -> None:
        command = self.settings.rclone_command()
        if command is None:
            raise IngestError("rclone 設定不存在。")
        result = subprocess.run([*command, *arguments], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise IngestError("上傳私人 Google Drive 失敗。")

    def _write_manifest(self, dataset: dict[str, object], checkpoint: dict[str, object]) -> Path:
        dataset_dir = self.store.media_dir / str(dataset["id"])
        dataset_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        manifest_path = dataset_dir / "manifest.json"
        temporary_path = dataset_dir / "manifest.json.tmp"
        payload: dict[str, Any] = {
            "dataset_id": dataset["id"],
            "name": dataset["name"],
            "sharer_name": dataset["sharer_name"],
            "source_provider": dataset["source_provider"],
            "source_url": dataset["source_url"],
            "source_title": dataset["source_title"],
            "source_video_count": dataset["source_video_count"],
            "frame_rate": 1,
            "splits": checkpoint["splits"],
            "frame_counts": checkpoint["frame_counts"],
            "completed_video_ids": checkpoint["completed_video_ids"],
        }
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(manifest_path)
        return manifest_path

    def _update(
        self,
        job_id: str,
        checkpoint: dict[str, object],
        state: str,
        step: str,
        completed: int,
        total: int,
        message: str,
    ) -> None:
        self.store.update_job(
            job_id,
            state=state,
            step=step,
            completed_items=completed,
            total_items=total,
            message=message,
            checkpoint=checkpoint,
        )

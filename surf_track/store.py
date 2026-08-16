from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ACTION_FIELDS = ("chasing_wave", "takeoff", "surfing")


class StoreError(RuntimeError):
    """Raised when persisted SurfTrack data is invalid or unavailable."""


class SurfTrackStore:
    """Small SQLite state store for resumable jobs, datasets and annotations."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir.resolve()
        self.database_path = self.data_dir / "surftrack.sqlite3"
        self.media_dir = self.data_dir / "media"
        self.secrets_dir = self.data_dir / "secrets"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.secrets_dir.mkdir(parents=True, exist_ok=True)
        self._secure_runtime_directories()
        self._initialize()
        self._secure_database()
        self.recover_interrupted_jobs()
        self.recover_interrupted_training_runs()

    def _secure_runtime_directories(self) -> None:
        """Runtime state can contain private source URLs and must stay user-only."""
        for path in (self.data_dir, self.media_dir, self.secrets_dir):
            path.chmod(0o700)

    def _secure_database(self) -> None:
        if self.database_path.exists():
            self.database_path.chmod(0o600)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_provider TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    source_title TEXT NOT NULL DEFAULT '',
                    sharer_name TEXT NOT NULL,
                    source_video_count INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'ready',
                    split_seed INTEGER NOT NULL DEFAULT 42,
                    train_ratio INTEGER NOT NULL DEFAULT 80,
                    valid_ratio INTEGER NOT NULL DEFAULT 10,
                    test_ratio INTEGER NOT NULL DEFAULT 10,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
                    local_path TEXT NOT NULL,
                    source_group TEXT NOT NULL,
                    split TEXT NOT NULL CHECK(split IN ('train', 'valid', 'test')),
                    width INTEGER,
                    height INTEGER,
                    complete_for_detection INTEGER NOT NULL DEFAULT 0,
                    crop_x REAL,
                    crop_y REAL,
                    crop_width REAL,
                    crop_height REAL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS annotations (
                    id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    width REAL NOT NULL,
                    height REAL NOT NULL,
                    chasing_wave INTEGER NOT NULL DEFAULT 0,
                    paddling INTEGER NOT NULL DEFAULT 0,
                    takeoff INTEGER NOT NULL DEFAULT 0,
                    surfing INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT REFERENCES datasets(id) ON DELETE CASCADE,
                    job_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    step TEXT NOT NULL,
                    completed_items INTEGER NOT NULL DEFAULT 0,
                    total_items INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    checkpoint_json TEXT NOT NULL DEFAULT '{}',
                    resumable INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
                    state TEXT NOT NULL,
                    epoch INTEGER NOT NULL DEFAULT 0,
                    total_epochs INTEGER NOT NULL DEFAULT 0,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS images_dataset_idx ON images(dataset_id);
                CREATE INDEX IF NOT EXISTS annotations_image_idx ON annotations(image_id);
                CREATE INDEX IF NOT EXISTS jobs_updated_idx ON jobs(updated_at DESC);
                """
            )
            self._migrate_dataset_source_fields(connection)
            self._migrate_image_crop_fields(connection)
            self._migrate_job_dataset_field(connection)

    @staticmethod
    def _migrate_dataset_source_fields(connection: sqlite3.Connection) -> None:
        """Add attribution fields to databases created by SurfTrack 0.2."""
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(datasets)").fetchall()
        }
        migrations = {
            "source_provider": "TEXT NOT NULL DEFAULT 'unknown'",
            "source_url": "TEXT NOT NULL DEFAULT ''",
            "source_title": "TEXT NOT NULL DEFAULT ''",
            "sharer_name": "TEXT NOT NULL DEFAULT ''",
            "source_video_count": "INTEGER NOT NULL DEFAULT 0",
        }
        for column, definition in migrations.items():
            if column not in columns:
                connection.execute(f"ALTER TABLE datasets ADD COLUMN {column} {definition}")

    @staticmethod
    def _migrate_image_crop_fields(connection: sqlite3.Connection) -> None:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(images)").fetchall()
        }
        for column in ("crop_x", "crop_y", "crop_width", "crop_height"):
            if column not in columns:
                connection.execute(f"ALTER TABLE images ADD COLUMN {column} REAL")

    @staticmethod
    def _migrate_job_dataset_field(connection: sqlite3.Connection) -> None:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if "dataset_id" not in columns:
            connection.execute(
                "ALTER TABLE jobs ADD COLUMN dataset_id TEXT REFERENCES datasets(id) ON DELETE CASCADE"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS jobs_dataset_idx ON jobs(dataset_id, updated_at DESC)"
        )

    def recover_interrupted_jobs(self) -> int:
        """Mark in-flight work resumable instead of losing progress after shutdown."""
        now = self._now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                   SET state = 'paused',
                       message = '上次執行中斷，可從檢查點繼續',
                       updated_at = ?
                 WHERE state IN ('running', 'stopping') AND resumable = 1
                """,
                (now,),
            )
            return cursor.rowcount

    def recover_interrupted_training_runs(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE training_runs
                   SET state = 'failed',
                       metrics_json = '{"error":"上次訓練因程式關閉而中斷，請重新按 Train。"}',
                       updated_at = ?
                 WHERE state IN ('queued', 'running')
                """,
                (self._now(),),
            )
            return cursor.rowcount

    def workspace_snapshot(self) -> dict[str, object]:
        with self._connect() as connection:
            datasets = [dict(row) for row in connection.execute(
                """
                SELECT d.*,
                       COUNT(DISTINCT i.id) AS image_count,
                       COUNT(DISTINCT CASE WHEN a.id IS NOT NULL THEN i.id END) AS labeled_image_count,
                       COUNT(DISTINCT CASE WHEN i.complete_for_detection = 1 THEN i.id END) AS detector_ready_count
                  FROM datasets d
             LEFT JOIN images i ON i.dataset_id = d.id
             LEFT JOIN annotations a ON a.image_id = i.id
              GROUP BY d.id
              ORDER BY d.created_at DESC
                """
            )]
            jobs = [dict(row) for row in connection.execute(
                "SELECT * FROM jobs ORDER BY updated_at DESC LIMIT 20"
            )]
            runs = [dict(row) for row in connection.execute(
                "SELECT * FROM training_runs ORDER BY updated_at DESC LIMIT 10"
            )]

        for job in jobs:
            job["checkpoint"] = json.loads(str(job.pop("checkpoint_json")))
            total = int(job["total_items"])
            job["progress"] = int(job["completed_items"]) / total if total else 0
            job["resumable"] = bool(job["resumable"])
        for run in runs:
            run["metrics"] = json.loads(str(run.pop("metrics_json")))
        return {"datasets": datasets, "jobs": jobs, "training_runs": runs}

    def create_training_run(self, dataset_id: str, *, total_epochs: int) -> dict[str, object]:
        self.get_dataset(dataset_id)
        now = self._now()
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        with self._connect() as connection:
            active = connection.execute(
                "SELECT 1 FROM training_runs WHERE dataset_id = ? AND state IN ('queued', 'running')",
                (dataset_id,),
            ).fetchone()
            if active is not None:
                raise StoreError("這個 Dataset 已經在訓練。")
            connection.execute(
                """
                INSERT INTO training_runs(
                    id, dataset_id, state, epoch, total_epochs, metrics_json, created_at, updated_at
                ) VALUES (?, ?, 'queued', 0, ?, '{}', ?, ?)
                """,
                (run_id, dataset_id, total_epochs, now, now),
            )
        return self.get_training_run(run_id)

    def get_training_run(self, run_id: str) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise StoreError("找不到這個訓練工作。")
        result = dict(row)
        result["metrics"] = json.loads(str(result.pop("metrics_json")))
        return result

    def update_training_run(
        self,
        run_id: str,
        *,
        state: str,
        epoch: int,
        metrics: dict[str, object],
    ) -> dict[str, object]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE training_runs
                   SET state = ?, epoch = ?, metrics_json = ?, updated_at = ?
                 WHERE id = ?
                """,
                (state, epoch, json.dumps(metrics, ensure_ascii=False), self._now(), run_id),
            )
            if cursor.rowcount != 1:
                raise StoreError("找不到這個訓練工作。")
        return self.get_training_run(run_id)

    def list_action_samples(self, dataset_id: str) -> list[dict[str, object]]:
        self.get_dataset(dataset_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT i.id AS image_id, i.local_path, i.split,
                       a.id AS annotation_id, a.x, a.y, a.width, a.height,
                       a.chasing_wave, a.takeoff, a.surfing
                  FROM images i
                  JOIN annotations a ON a.image_id = i.id
                 WHERE i.dataset_id = ?
              ORDER BY i.created_at, a.id
                """,
                (dataset_id,),
            ).fetchall()
        return [
            {
                **dict(row),
                "path": self.media_dir / str(row["local_path"]),
                "labels": [int(row[field]) for field in ACTION_FIELDS],
            }
            for row in rows
        ]

    def mark_annotated_images_detection_ready(self, dataset_id: str) -> int:
        self.get_dataset(dataset_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE images
                   SET complete_for_detection = 1
                 WHERE dataset_id = ?
                   AND EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = images.id)
                """,
                (dataset_id,),
            )
            return cursor.rowcount

    def create_dataset(
        self,
        name: str,
        *,
        source_provider: str,
        source_url: str,
        sharer_name: str,
        source_title: str = "",
        source_video_count: int = 0,
        split_seed: int = 42,
    ) -> dict[str, object]:
        cleaned_name = name.strip()
        if not cleaned_name:
            raise StoreError("Dataset 名稱不可為空。")
        cleaned_sharer = sharer_name.strip()
        if not cleaned_sharer:
            raise StoreError("分享者／來源名稱不可為空。")
        cleaned_provider = source_provider.strip()
        if cleaned_provider not in {"google_drive", "facebook"}:
            raise StoreError("來源必須是 Google Drive 或 Facebook。")
        cleaned_url = source_url.strip()
        if not cleaned_url.startswith("https://"):
            raise StoreError("Dataset 必須保存正規化後的 HTTPS 來源網址。")
        if source_video_count < 0:
            raise StoreError("影片數量不可小於 0。")
        now = self._now()
        dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets(
                    id, name, source_provider, source_url, source_title,
                    sharer_name, source_video_count, split_seed, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    cleaned_name,
                    cleaned_provider,
                    cleaned_url,
                    source_title.strip(),
                    cleaned_sharer,
                    int(source_video_count),
                    split_seed,
                    now,
                    now,
                ),
            )
        return {
            "id": dataset_id,
            "name": cleaned_name,
            "source_provider": cleaned_provider,
            "source_url": cleaned_url,
            "source_title": source_title.strip(),
            "sharer_name": cleaned_sharer,
            "source_video_count": int(source_video_count),
            "split_seed": split_seed,
        }

    def get_dataset(self, dataset_id: str) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            raise StoreError("找不到這個 Dataset。")
        return dict(row)

    def start_ingest_job(self, dataset_id: str) -> dict[str, object]:
        dataset = self.get_dataset(dataset_id)
        if dataset["source_provider"] != "google_drive":
            raise StoreError("目前開始處理只支援公開 Google Drive Dataset。")
        now = self._now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM jobs
                 WHERE dataset_id = ? AND job_type = 'ingest'
              ORDER BY created_at DESC LIMIT 1
                """,
                (dataset_id,),
            ).fetchone()
            if row is None:
                job_id = f"job_{uuid.uuid4().hex[:12]}"
                checkpoint = {"completed_video_ids": [], "splits": {}, "frame_counts": {}}
                connection.execute(
                    """
                    INSERT INTO jobs(
                        id, dataset_id, job_type, state, step, total_items,
                        message, checkpoint_json, created_at, updated_at
                    ) VALUES (?, ?, 'ingest', 'queued', 'scan', ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        dataset_id,
                        int(dataset["source_video_count"]),
                        "等待掃描來源",
                        json.dumps(checkpoint, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            else:
                job_id = str(row["id"])
                if row["state"] in {"paused", "failed"}:
                    connection.execute(
                        """
                        UPDATE jobs
                           SET state = 'queued', step = 'scan', message = '準備繼續處理', updated_at = ?
                         WHERE id = ?
                        """,
                        (now, job_id),
                    )
            if row is None or row["state"] != "completed":
                connection.execute(
                    "UPDATE datasets SET state = 'processing', updated_at = ? WHERE id = ?",
                    (now, dataset_id),
                )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise StoreError("找不到這個處理工作。")
        job = dict(row)
        job["checkpoint"] = json.loads(str(job.pop("checkpoint_json")))
        total = int(job["total_items"])
        job["progress"] = int(job["completed_items"]) / total if total else 0
        job["resumable"] = bool(job["resumable"])
        return job

    def update_job(
        self,
        job_id: str,
        *,
        state: str,
        step: str,
        completed_items: int,
        total_items: int,
        message: str,
        checkpoint: dict[str, object],
    ) -> dict[str, object]:
        now = self._now()
        with self._connect() as connection:
            row = connection.execute("SELECT dataset_id FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise StoreError("找不到這個處理工作。")
            connection.execute(
                """
                UPDATE jobs
                   SET state = ?, step = ?, completed_items = ?, total_items = ?,
                       message = ?, checkpoint_json = ?, updated_at = ?
                 WHERE id = ?
                """,
                (
                    state,
                    step,
                    completed_items,
                    total_items,
                    message,
                    json.dumps(checkpoint, ensure_ascii=False),
                    now,
                    job_id,
                ),
            )
            dataset_state = "ready" if state == "completed" else "error" if state == "failed" else "processing"
            connection.execute(
                "UPDATE datasets SET state = ?, updated_at = ? WHERE id = ?",
                (dataset_state, now, row["dataset_id"]),
            )
        return self.get_job(job_id)

    def add_image(
        self,
        dataset_id: str,
        local_path: Path,
        *,
        source_group: str,
        split: str,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, object]:
        if split not in {"train", "valid", "test"}:
            raise StoreError("split 必須是 train、valid 或 test。")
        resolved = local_path.resolve()
        try:
            relative_path = resolved.relative_to(self.media_dir)
        except ValueError as exc:
            raise StoreError("影像必須位於 SurfTrack media 目錄內。") from exc
        image_id = f"img_{uuid.uuid4().hex[:12]}"
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM images WHERE dataset_id = ? AND local_path = ?",
                (dataset_id, relative_path.as_posix()),
            ).fetchone()
            if existing is not None:
                return {"id": existing["id"], "dataset_id": dataset_id, "split": split}
            connection.execute(
                """
                INSERT INTO images(id, dataset_id, local_path, source_group, split, width, height, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (image_id, dataset_id, relative_path.as_posix(), source_group, split, width, height, self._now()),
            )
        return {"id": image_id, "dataset_id": dataset_id, "split": split}

    def add_images(
        self,
        dataset_id: str,
        local_paths: list[Path],
        *,
        source_group: str,
        split: str,
        width: int | None = None,
        height: int | None = None,
    ) -> int:
        if split not in {"train", "valid", "test"}:
            raise StoreError("split 必須是 train、valid 或 test。")
        relative_paths: list[str] = []
        for local_path in local_paths:
            try:
                relative_paths.append(local_path.resolve().relative_to(self.media_dir).as_posix())
            except ValueError as exc:
                raise StoreError("影像必須位於 SurfTrack media 目錄內。") from exc
        inserted = 0
        with self._connect() as connection:
            existing = {
                str(row[0])
                for row in connection.execute(
                    "SELECT local_path FROM images WHERE dataset_id = ?", (dataset_id,)
                )
            }
            for relative_path in relative_paths:
                if relative_path in existing:
                    continue
                connection.execute(
                    """
                    INSERT INTO images(id, dataset_id, local_path, source_group, split, width, height, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"img_{uuid.uuid4().hex[:12]}", dataset_id, relative_path,
                        source_group, split, width, height, self._now(),
                    ),
                )
                inserted += 1
        return inserted

    def list_label_images(self, dataset_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT i.*,
                       COUNT(a.id) AS annotation_count
                  FROM images i
             LEFT JOIN annotations a ON a.image_id = i.id
                 WHERE i.dataset_id = ?
              GROUP BY i.id
              ORDER BY i.created_at, i.id
                """,
                (dataset_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "split": row["split"],
                "source_group": row["source_group"],
                "width": row["width"],
                "height": row["height"],
                "complete_for_detection": bool(row["complete_for_detection"]),
                "annotation_count": row["annotation_count"],
                "image_url": f"/api/v1/images/{row['id']}/content",
            }
            for row in rows
        ]

    def get_image_path(self, image_id: str) -> Path:
        with self._connect() as connection:
            row = connection.execute("SELECT local_path FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None:
            raise StoreError("找不到這張影像。")
        path = (self.media_dir / str(row["local_path"])).resolve()
        try:
            path.relative_to(self.media_dir)
        except ValueError as exc:
            raise StoreError("影像路徑超出允許範圍。") from exc
        if not path.is_file():
            raise StoreError("影像檔案不存在。")
        return path

    def get_annotations(self, image_id: str) -> dict[str, object]:
        with self._connect() as connection:
            image = connection.execute(
                """
                SELECT complete_for_detection, crop_x, crop_y, crop_width, crop_height
                  FROM images WHERE id = ?
                """,
                (image_id,),
            ).fetchone()
            if image is None:
                raise StoreError("找不到這張影像。")
            rows = connection.execute(
                "SELECT * FROM annotations WHERE image_id = ? ORDER BY updated_at, id", (image_id,)
            ).fetchall()
        crop_region = None
        if image["crop_x"] is not None:
            crop_region = {
                "x": image["crop_x"],
                "y": image["crop_y"],
                "width": image["crop_width"],
                "height": image["crop_height"],
            }
        return {
            "image_id": image_id,
            "complete_for_detection": bool(image["complete_for_detection"]),
            "crop_region": crop_region,
            "annotations": [
                {
                    "id": row["id"],
                    "x": row["x"],
                    "y": row["y"],
                    "width": row["width"],
                    "height": row["height"],
                    **{field: bool(row[field]) for field in ACTION_FIELDS},
                }
                for row in rows
            ],
        }

    def save_annotations(
        self,
        image_id: str,
        annotations: list[dict[str, Any]],
        *,
        complete_for_detection: bool,
        crop_region: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        normalized: list[dict[str, object]] = []
        for item in annotations:
            coordinates = [float(item.get(field, -1)) for field in ("x", "y", "width", "height")]
            x, y, width, height = coordinates
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
                raise StoreError("bbox 必須是影像範圍內的 normalized coordinates。")
            normalized.append(
                {
                    "id": str(item.get("id") or f"box_{uuid.uuid4().hex[:12]}"),
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    **{field: bool(item.get(field, False)) for field in ACTION_FIELDS},
                }
            )

        normalized_crop: dict[str, float] | None = None
        if crop_region is not None:
            coordinates = [float(crop_region.get(field, -1)) for field in ("x", "y", "width", "height")]
            x, y, width, height = coordinates
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
                raise StoreError("裁切區域必須位於影像範圍內。")
            normalized_crop = {"x": x, "y": y, "width": width, "height": height}
            crop_right = x + width
            crop_bottom = y + height
            for item in normalized:
                if (
                    float(item["x"]) < x
                    or float(item["y"]) < y
                    or float(item["x"]) + float(item["width"]) > crop_right + 0.000001
                    or float(item["y"]) + float(item["height"]) > crop_bottom + 0.000001
                ):
                    raise StoreError("bbox 必須位於裁切區域內。")

        now = self._now()
        with self._connect() as connection:
            exists = connection.execute("SELECT 1 FROM images WHERE id = ?", (image_id,)).fetchone()
            if exists is None:
                raise StoreError("找不到這張影像。")
            connection.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
            for item in normalized:
                connection.execute(
                    """
                    INSERT INTO annotations(
                        id, image_id, x, y, width, height,
                        chasing_wave, takeoff, surfing, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["id"], image_id, item["x"], item["y"], item["width"], item["height"],
                        int(item["chasing_wave"]), int(item["takeoff"]), int(item["surfing"]), now,
                    ),
                )
            connection.execute(
                """
                UPDATE images
                   SET complete_for_detection = ?, crop_x = ?, crop_y = ?,
                       crop_width = ?, crop_height = ?
                 WHERE id = ?
                """,
                (
                    int(complete_for_detection),
                    normalized_crop["x"] if normalized_crop else None,
                    normalized_crop["y"] if normalized_crop else None,
                    normalized_crop["width"] if normalized_crop else None,
                    normalized_crop["height"] if normalized_crop else None,
                    image_id,
                ),
            )
        return {
            "image_id": image_id,
            "annotation_count": len(normalized),
            "complete_for_detection": complete_for_detection,
            "crop_region": normalized_crop,
            "saved_at": now,
        }

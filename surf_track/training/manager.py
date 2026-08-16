from __future__ import annotations

import random
import threading

from surf_track.store import StoreError, SurfTrackStore


class TrainingError(RuntimeError):
    pass


class TrainingManager:
    def __init__(self, store: SurfTrackStore):
        self.store = store
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    def preview(self, dataset_id: str, *, limit: int = 25) -> dict[str, object]:
        samples = self.store.list_action_samples(dataset_id)
        model_path = self.store.data_dir / "models" / dataset_id / "actions_mobilenet_v3_small.pt"
        if not model_path.is_file():
            raise TrainingError("尚未有可預覽的動作模型，請先完成 Train。")

        grouped: dict[str, list[dict[str, object]]] = {}
        for sample in samples:
            grouped.setdefault(str(sample["image_id"]), []).append(sample)
        evaluation_ids = [
            image_id for image_id, boxes in grouped.items()
            if boxes[0]["split"] in {"valid", "test"}
        ]
        train_ids = [
            image_id for image_id, boxes in grouped.items()
            if boxes[0]["split"] == "train"
        ]
        selector = random.Random(42)
        selector.shuffle(evaluation_ids)
        selector.shuffle(train_ids)
        selected_ids = (evaluation_ids + train_ids)[:limit]
        selected_samples = [sample for image_id in selected_ids for sample in grouped[image_id]]
        if not selected_samples:
            raise TrainingError("沒有可預覽的標注圖片。")

        from surf_track.training.action import ACTION_NAMES, predict_action_samples

        probabilities = predict_action_samples(selected_samples, model_path)
        predictions = dict(zip((str(sample["annotation_id"]) for sample in selected_samples), probabilities))
        images: list[dict[str, object]] = []
        for image_id in selected_ids:
            boxes = grouped[image_id]
            images.append({
                "image_id": image_id,
                "image_url": f"/api/v1/images/{image_id}/thumbnail",
                "split": boxes[0]["split"],
                "boxes": [
                    {
                        "x": box["x"],
                        "y": box["y"],
                        "width": box["width"],
                        "height": box["height"],
                        "probabilities": dict(zip(ACTION_NAMES, predictions[str(box["annotation_id"])])),
                    }
                    for box in boxes
                ],
            })
        return {
            "mode": "action_predictions_on_human_boxes",
            "message": "框為人工標注；框內數字為動作模型機率。",
            "images": images,
        }

    def thumbnail_path(self, image_id: str) -> str:
        from PIL import Image, ImageOps

        source = self.store.get_image_path(image_id)
        cache_dir = self.store.data_dir / "cache" / "thumbnails"
        cache_dir.mkdir(parents=True, exist_ok=True)
        destination = cache_dir / f"{image_id}.jpg"
        if destination.is_file() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return str(destination)
        with self._lock:
            if destination.is_file() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
                return str(destination)
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.thumbnail((640, 640))
                temporary = destination.with_suffix(".tmp.jpg")
                image.save(temporary, format="JPEG", quality=82, optimize=True)
                temporary.replace(destination)
        return str(destination)

    def start(self, dataset_id: str, *, epochs: int = 8) -> dict[str, object]:
        samples = self.store.list_action_samples(dataset_id)
        if not samples:
            raise TrainingError("尚未標注任何可訓練的人物框。")
        try:
            run = self.store.create_training_run(dataset_id, total_epochs=epochs)
        except StoreError as exc:
            raise TrainingError(str(exc)) from exc
        run_id = str(run["id"])
        thread = threading.Thread(
            target=self._run,
            args=(run_id, dataset_id, epochs, samples),
            name=f"surftrack-train-{run_id}",
            daemon=True,
        )
        with self._lock:
            self._threads[run_id] = thread
        thread.start()
        return run

    def _run(self, run_id: str, dataset_id: str, epochs: int, samples: list[dict[str, object]]) -> None:
        metrics: dict[str, object] = {"stage": "載入預訓練小模型"}
        self.store.update_training_run(run_id, state="running", epoch=0, metrics=metrics)
        try:
            from surf_track.training.action import train_action_model

            output_path = self.store.data_dir / "models" / dataset_id / "actions_mobilenet_v3_small.pt"

            def on_epoch(epoch: int, current: dict[str, float]) -> None:
                self.store.update_training_run(run_id, state="running", epoch=epoch, metrics=current)

            final_metrics = train_action_model(samples, output_path, epochs=epochs, on_epoch=on_epoch)
            final_metrics["model_path"] = str(output_path.relative_to(self.store.data_dir))
            final_metrics["detector_state"] = "waiting_for_complete_images"
            self.store.update_training_run(run_id, state="completed", epoch=epochs, metrics=final_metrics)
        except Exception as exc:
            metrics["error"] = str(exc)
            self.store.update_training_run(run_id, state="failed", epoch=0, metrics=metrics)
        finally:
            with self._lock:
                self._threads.pop(run_id, None)

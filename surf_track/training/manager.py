from __future__ import annotations

import random
import threading
from pathlib import Path

from surf_track.store import StoreError, SurfTrackStore

# One combined model trained across the selected sharers, not one model per dataset.
ACTION_MODEL_PATH = Path("models") / "action" / "actions_mobilenet_v3_small.pt"
DETECTOR_MODEL_PATH = Path("models") / "detector" / "surfer_fcos_mobilenet_v3_large.pt"


class TrainingError(RuntimeError):
    pass


class TrainingManager:
    def __init__(self, store: SurfTrackStore):
        self.store = store
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    def preview(self, *, limit: int = 16, seed: int = 42) -> dict[str, object]:
        """`seed` picks which sample of images to draw, so the UI can ask for another batch."""
        samples = self.store.list_action_samples()
        model_path = self.store.data_dir / ACTION_MODEL_PATH
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
        selector = random.Random(seed)
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

    def start(self, sharers: list[str], *, host: str, epochs: int = 24) -> dict[str, object]:
        from surf_track.training.stream import (
            DETECTOR_EPOCHS, DETECTOR_MIN_TRAIN_IMAGES, DETECTOR_MIN_VALID_IMAGES,
            StreamError, validate_host,
        )

        if not sharers:
            raise TrainingError("請至少勾選一位分享者的資料集。")
        # Host first: it reaches ssh, and it is validated before the run row exists so a
        # typo does not leave a failed run behind.
        try:
            target = validate_host(host)
        except StreamError as exc:
            raise TrainingError(str(exc)) from exc
        samples = self.store.list_action_samples(sharers)
        if not samples:
            raise TrainingError("勾選的分享者還沒有任何已標注的人物框。")
        # The detector needs whole frames where every surfer is boxed, which lag the action
        # boxes; too few and the run trains the action model alone rather than failing.
        detector_images = self.store.list_detector_images(sharers)
        splits = [image["split"] for image in detector_images]
        detector_epochs = DETECTOR_EPOCHS if (
            splits.count("train") >= DETECTOR_MIN_TRAIN_IMAGES
            and splits.count("valid") >= DETECTOR_MIN_VALID_IMAGES
        ) else 0
        try:
            run = self.store.create_training_run(sharers, total_epochs=epochs + detector_epochs)
        except StoreError as exc:
            raise TrainingError(str(exc)) from exc
        run_id = str(run["id"])
        thread = threading.Thread(
            target=self._run,
            args=(run_id, epochs, samples, target, detector_images, detector_epochs),
            name=f"surftrack-train-{run_id}",
            daemon=True,
        )
        with self._lock:
            self._threads[run_id] = thread
        thread.start()
        return run

    def _run(
        self,
        run_id: str,
        epochs: int,
        samples: list[dict[str, object]],
        host: str,
        detector_images: list[dict[str, object]],
        detector_epochs: int,
    ) -> None:
        metrics: dict[str, object] = {"stage": f"連線遠端 GPU 主機 {host}"}
        self.store.update_training_run(run_id, state="running", epoch=0, metrics=metrics)
        try:
            # Training only ever runs on the remote GPU host; frames stay on this machine.
            from surf_track.training.stream import feed, feed_detector

            output_path = self.store.data_dir / ACTION_MODEL_PATH

            def on_epoch(epoch: int, current: dict[str, float]) -> None:
                self.store.update_training_run(run_id, state="running", epoch=epoch, metrics=current)

            final_metrics = feed(
                samples,
                output_path,
                host=host,
                epochs=epochs,
                on_epoch=on_epoch,
                cache_dir=self.store.data_dir / "cache" / "crops",
            )
            final_metrics["model_path"] = str(output_path.relative_to(self.store.data_dir))

            if not detector_epochs:
                final_metrics["detector_state"] = "waiting_for_complete_images"
            else:
                # Sequentially, on the same run row and the same 6GB card: the progress bar
                # counts action epochs first, then detector epochs.
                detector_path = self.store.data_dir / DETECTOR_MODEL_PATH

                def on_detector_epoch(epoch: int, current: dict[str, float]) -> None:
                    self.store.update_training_run(
                        run_id, state="running", epoch=epochs + epoch,
                        metrics={**final_metrics, **current},
                    )

                try:
                    detector_metrics = feed_detector(
                        detector_images,
                        detector_path,
                        host=host,
                        epochs=detector_epochs,
                        on_epoch=on_detector_epoch,
                        cache_dir=self.store.data_dir / "cache" / "frames",
                    )
                except Exception as exc:
                    # The action model is already trained and written; failing the whole run
                    # here would throw its metrics away and show the Train page nothing.
                    final_metrics["detector_state"] = "failed"
                    final_metrics["error"] = f"動作模型已完成，但 detector 訓練失敗：{exc}"
                else:
                    # Only the detector_* keys: the rest (remote_gpu, streamed_mb,
                    # frames_built) would otherwise overwrite what the action run reported.
                    final_metrics.update({
                        name: value for name, value in detector_metrics.items()
                        if name.startswith("detector_")
                    })
                    final_metrics["detector_state"] = "trained"
                    final_metrics["detector_model_path"] = str(
                        detector_path.relative_to(self.store.data_dir)
                    )
            self.store.update_training_run(
                run_id, state="completed", epoch=epochs + detector_epochs, metrics=final_metrics,
            )
        except Exception as exc:
            metrics["error"] = str(exc)
            self.store.update_training_run(run_id, state="failed", epoch=0, metrics=metrics)
        finally:
            with self._lock:
                self._threads.pop(run_id, None)

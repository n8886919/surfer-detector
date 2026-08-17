"""Train the action model on a remote CUDA host without copying the dataset there.

Frames stay on this machine. The feeder crops each annotation locally, streams only
the crops for the batches currently in flight over one SSH pipe, and the remote worker
keeps them in memory, trains on the GPU, and streams the checkpoint back. The remote
host never writes dataset bytes to disk.

    # on this machine
    python -m surf_track.training.stream feed --sharer surfrec.tw --host nolan@192.168.31.128

    # started by the feeder over ssh, not by hand
    python -m surf_track.training.stream serve
"""
from __future__ import annotations

import argparse
import json
import random
import re
import struct
import subprocess
import sys
import threading
from io import BytesIO
from pathlib import Path

IMAGE_SIZE = 160
LOWRES_EVAL_SIDE = 64
VALID_BATCH_SIZE = 32
CROP_QUALITY = 95
DEFAULT_REMOTE_ROOT = "~/surf-track"
_FRAME = struct.Struct(">II")
# The host is passed straight to ssh, which has no "--" terminator: a value starting with
# "-" would be read as an option (e.g. -oProxyCommand=...) and run commands locally.
_HOST_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._+-]*@[A-Za-z0-9][A-Za-z0-9.-]*")


class StreamError(RuntimeError):
    pass


def validate_host(host: object) -> str:
    if not isinstance(host, str) or not _HOST_PATTERN.fullmatch(host.strip()):
        raise StreamError("遠端 GPU 主機格式必須是 user@host，例如 nolan@192.168.31.128。")
    return host.strip()


def _log(message: str) -> None:
    """Worker progress goes to stderr; stdout carries the binary protocol."""
    print(message, file=sys.stderr, flush=True)


# --- wire protocol -------------------------------------------------------------------

def send_message(stream, header: dict[str, object], payload: bytes = b"") -> None:
    encoded = json.dumps(header, separators=(",", ":")).encode()
    stream.write(_FRAME.pack(len(encoded), len(payload)))
    stream.write(encoded)
    if payload:
        stream.write(payload)
    stream.flush()


def _read_exactly(stream, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("stream closed mid-message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_message(stream) -> tuple[dict[str, object], bytes] | None:
    prefix = stream.read(_FRAME.size)
    if not prefix:
        return None
    if len(prefix) < _FRAME.size:
        prefix += _read_exactly(stream, _FRAME.size - len(prefix))
    header_length, payload_length = _FRAME.unpack(prefix)
    header = json.loads(_read_exactly(stream, header_length))
    payload = _read_exactly(stream, payload_length) if payload_length else b""
    return header, payload


# --- local side ----------------------------------------------------------------------

class CropCache:
    """Crops are deterministic per annotation, so decode each frame at most once."""

    def __init__(self, directory: Path, image_size: int = IMAGE_SIZE):
        self.directory = directory
        self.image_size = image_size
        self.directory.mkdir(parents=True, exist_ok=True)
        self.built = 0
        self.reused = 0
        self._resize = None

    def _resizer(self):
        if self._resize is None:
            from torchvision import transforms

            self._resize = transforms.Resize((self.image_size, self.image_size))
        return self._resize

    def crop_bytes(self, sample: dict[str, object]) -> bytes:
        destination = self.directory / f"{sample['annotation_id']}_{self.image_size}.jpg"
        if destination.is_file():
            self.reused += 1
            return destination.read_bytes()

        from PIL import Image

        from surf_track.training.action import crop_for_sample

        with Image.open(Path(str(sample["path"]))) as source:
            crop = self._resizer()(crop_for_sample(source.convert("RGB"), sample))
        buffer = BytesIO()
        crop.save(buffer, format="JPEG", quality=CROP_QUALITY)
        blob = buffer.getvalue()
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(blob)
        temporary.replace(destination)
        self.built += 1
        return blob


class _WorkerLink:
    """Owns the ssh subprocess and the reader thread draining its replies."""

    def __init__(self, host: str, remote_root: str, prefetch: int, ack_timeout: float, on_epoch):
        self.on_epoch = on_epoch
        command = f"cd {remote_root} && exec .venv/bin/python -m surf_track.training.stream serve"
        self.process = subprocess.Popen(
            ["ssh", "-T", "-o", "BatchMode=yes", host, command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self.permits = threading.Semaphore(prefetch)
        self.prefetch = prefetch
        self.ack_timeout = ack_timeout
        self.failure: str | None = None
        self.checkpoint: bytes | None = None
        self.final_metrics: dict[str, object] | None = None
        self.ready = threading.Event()
        self.closed = threading.Event()
        self.gpu = ""
        self._reader = threading.Thread(target=self._drain, name="surftrack-stream-reader", daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        try:
            while True:
                message = receive_message(self.process.stdout)
                if message is None:
                    break
                header, payload = message
                kind = header.get("type")
                if kind == "ack":
                    self.permits.release()
                elif kind == "ready":
                    self.gpu = str(header.get("gpu", ""))
                    self.ready.set()
                elif kind == "metrics":
                    self.on_epoch(int(header["epoch"]), header["metrics"])
                elif kind == "checkpoint":
                    self.final_metrics = header.get("metrics")
                    self.checkpoint = payload
                elif kind == "error":
                    self.failure = str(header.get("message"))
        except (EOFError, OSError) as exc:
            self.failure = self.failure or f"worker link broken: {exc}"
        finally:
            if self.failure is None and self.checkpoint is None:
                code = self.process.poll()
                if code:
                    self.failure = f"遠端 worker 以 exit code {code} 結束（詳見上方 stderr）。"
            self.closed.set()
            self.ready.set()
            # Never leave the sender blocked on a worker that is gone.
            for _ in range(self.prefetch + 8):
                self.permits.release()

    def reserve_slot(self) -> None:
        # A stalled worker must surface as an error rather than hanging the feeder forever.
        if not self.permits.acquire(timeout=self.ack_timeout):
            raise StreamError(
                f"遠端 worker 超過 {self.ack_timeout:.0f} 秒沒有回應 ack；"
                "請檢查上方 [worker] stderr 訊息。"
            )
        if self.failure:
            raise StreamError(self.failure)
        if self.closed.is_set():
            raise StreamError("遠端 worker 已結束，無法繼續送 batch。")

    def send(self, header: dict[str, object], payload: bytes = b"") -> None:
        try:
            send_message(self.process.stdin, header, payload)
        except (BrokenPipeError, OSError) as exc:
            raise StreamError(self.failure or f"無法寫入遠端 worker: {exc}") from exc

    def finish(self) -> None:
        try:
            self.process.stdin.close()
        except OSError:
            pass
        self._reader.join(timeout=600)
        self.process.wait(timeout=60)


def _batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def feed(
    samples: list[dict[str, object]],
    output_path: Path,
    *,
    host: str,
    epochs: int,
    on_epoch,
    cache_dir: Path,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    batch_size: int = 16,
    prefetch: int = 4,
    ack_timeout: float = 300.0,
    seed: int = 42,
) -> dict[str, object]:
    """Train on `host`'s GPU from local crops; write the returned checkpoint to output_path.

    This is the only training path — there is no local trainer. Raises StreamError if the
    host is malformed, unreachable, has no CUDA, or stops acking.
    """
    target = validate_host(host)
    grouped = {split: [s for s in samples if s["split"] == split] for split in ("train", "valid", "test")}
    if len(grouped["train"]) < 12 or len(grouped["valid"]) < 3:
        raise StreamError("至少需要 12 個 train 框和 3 個 valid 框才能開始動作訓練。")

    positives = [
        sum(int(sample["labels"][index]) for sample in grouped["train"])
        for index in range(3)
    ]
    pos_weight = [min((len(grouped["train"]) - p) / max(p, 1), 12.0) for p in positives]

    cache = CropCache(cache_dir)
    link = _WorkerLink(target, remote_root, prefetch, ack_timeout, on_epoch)
    link.ready.wait(timeout=300)
    if link.failure or link.closed.is_set():
        raise StreamError(link.failure or f"連不上遠端 GPU 主機 {target}，請確認 ssh 金鑰與主機位址。")
    on_epoch(0, {
        "stage": f"已連上遠端 GPU {link.gpu}",
        "train_boxes": float(len(grouped["train"])),
        "valid_boxes": float(len(grouped["valid"])),
    })

    shuffler = random.Random(seed)
    sent_bytes = 0
    link.send({
        "type": "config",
        "epochs": epochs,
        "image_size": IMAGE_SIZE,
        "pos_weight": pos_weight,
        "train_boxes": len(grouped["train"]),
        "valid_boxes": len(grouped["valid"]),
        "seed": seed,
    })

    for epoch in range(1, epochs + 1):
        order = list(range(len(grouped["train"])))
        shuffler.shuffle(order)
        planned = [
            ("train", [grouped["train"][index] for index in chunk])
            for chunk in _batches(order, batch_size)
        ]
        planned += [("valid", chunk) for chunk in _batches(grouped["valid"], VALID_BATCH_SIZE)]
        for split, chunk in planned:
            blobs = [cache.crop_bytes(sample) for sample in chunk]
            payload = b"".join(blobs)
            sent_bytes += len(payload)
            link.reserve_slot()
            link.send({
                "type": "batch",
                "epoch": epoch,
                "split": split,
                "sizes": [len(blob) for blob in blobs],
                "labels": [[int(value) for value in sample["labels"]] for sample in chunk],
            }, payload)
        link.send({"type": "epoch_end", "epoch": epoch})

    link.send({"type": "finish"})
    link.finish()

    if link.failure:
        raise StreamError(link.failure)
    if not link.checkpoint:
        raise StreamError("遠端沒有回傳 checkpoint。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_bytes(link.checkpoint)
    temporary.replace(output_path)

    metrics = dict(link.final_metrics or {})
    metrics["remote_gpu"] = link.gpu
    metrics["streamed_mb"] = round(sent_bytes / 1e6, 1)
    metrics["crops_built"] = float(cache.built)
    return metrics


def _feed_cli(args: argparse.Namespace) -> int:
    from surf_track.store import SurfTrackStore
    from surf_track.training.manager import ACTION_MODEL_PATH

    store = SurfTrackStore(Path(args.data_dir))
    samples = store.list_action_samples(args.sharer)
    if not samples:
        print("勾選的分享者還沒有任何已標注的人物框。", file=sys.stderr)
        return 1

    def on_epoch(epoch: int, metrics: dict[str, object]) -> None:
        if "action_macro_f1" not in metrics:
            print(f"  {metrics.get('stage', '')}", flush=True)
            return
        print(
            f"  epoch {epoch}/{args.epochs}"
            f"  loss={metrics['train_loss']:.4f}"
            f"  macroF1={metrics['action_macro_f1']:.3f}"
            f"  lowres={metrics['action_macro_f1_lowres']:.3f}"
            f"  cw={metrics['chasing_wave_f1']:.3f}"
            f"  tk={metrics['takeoff_f1']:.3f}"
            f"  sf={metrics['surfing_f1']:.3f}",
            flush=True,
        )

    output_path = store.data_dir / ACTION_MODEL_PATH
    try:
        metrics = feed(
            samples,
            output_path,
            host=args.host,
            epochs=args.epochs,
            on_epoch=on_epoch,
            cache_dir=store.data_dir / "cache" / "crops",
            remote_root=args.remote_root,
            batch_size=args.batch_size,
            prefetch=args.prefetch,
            ack_timeout=args.ack_timeout,
            seed=args.seed,
        )
    except StreamError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"streamed         {metrics['streamed_mb']} MB over {args.epochs} epochs")
    print(f"checkpoint       {output_path}")
    print(f"best macro F1    {metrics['action_macro_f1']:.3f}")
    return 0


# --- remote side ---------------------------------------------------------------------

def _decode_batch(payload: bytes, sizes: list[int], labels: list[list[int]], transform):
    import torch
    from PIL import Image

    tensors = []
    offset = 0
    for size in sizes:
        with Image.open(BytesIO(payload[offset:offset + size])) as source:
            tensors.append(transform(source.convert("RGB")))
        offset += size
    return torch.stack(tensors), torch.tensor(labels, dtype=torch.float32)


def serve() -> int:
    stdout = sys.stdout.buffer
    try:
        import torch
        from torch import nn
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        from surf_track.training.action import ACTION_NAMES, f1_scores, macro_f1
        from surf_track.training.augmentation import action_transform

        if not torch.cuda.is_available():
            send_message(stdout, {"type": "error", "message": "遠端沒有可用的 CUDA GPU；這個 worker 不支援 CPU 訓練。"})
            return 1
        device = torch.device("cuda")
        _log(f"[worker] cuda ready: {torch.cuda.get_device_name(0)}")
        send_message(stdout, {"type": "ready", "gpu": torch.cuda.get_device_name(0)})

        stdin = sys.stdin.buffer
        first = receive_message(stdin)
        if first is None or first[0].get("type") != "config":
            send_message(stdout, {"type": "error", "message": "沒有收到 config。"})
            return 1
        config = first[0]
        _log(f"[worker] config: {config['epochs']} epochs, {config['train_boxes']} train boxes")
        epochs = int(config["epochs"])
        train_boxes = int(config["train_boxes"])
        valid_boxes = int(config["valid_boxes"])

        torch.manual_seed(int(config["seed"]))
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        for block in model.features[-3:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(ACTION_NAMES))
        model.to(device)

        loss_function = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(config["pos_weight"], dtype=torch.float32)
        ).to(device)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=8e-4, weight_decay=1e-4
        )
        train_transform = action_transform(training=True)
        valid_transform = action_transform(training=False)
        # Second eval tier at a forced 64 px: the clean tier cannot show whether the
        # resolution augmentation bought anything, because 96.8% of crops are downsampled.
        lowres_transform = action_transform(training=False, degrade_to=LOWRES_EVAL_SIDE)

        best_score = -1.0
        best_state: dict[str, object] | None = None
        best_metrics: dict[str, float] = {}
        loss_total = 0.0
        valid_logits: list[object] = []
        lowres_logits: list[object] = []
        valid_targets: list[object] = []

        _log("[worker] model ready, waiting for batches")
        received = 0
        while True:
            message = receive_message(stdin)
            if message is None:
                _log("[worker] stdin closed")
                break
            header, payload = message
            kind = header.get("type")
            received += 1
            if received <= 6:
                _log(f"[worker] message {received}: {kind} ({len(payload)} bytes)")

            if kind == "batch":
                training = header["split"] == "train"
                if training:
                    images, targets = _decode_batch(
                        payload, header["sizes"], header["labels"], train_transform
                    )
                    model.train()
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_function(model(images), targets)
                    loss.backward()
                    optimizer.step()
                    loss_total += float(loss.detach()) * len(images)
                else:
                    model.eval()
                    # Same boxes scored twice; costs no extra wire bytes.
                    with torch.inference_mode():
                        for transform, sink in (
                            (valid_transform, valid_logits),
                            (lowres_transform, lowres_logits),
                        ):
                            batch, targets = _decode_batch(
                                payload, header["sizes"], header["labels"], transform
                            )
                            sink.append(model(batch.to(device, non_blocking=True)).cpu())
                    valid_targets.append(targets)
                send_message(stdout, {"type": "ack"})
                if received <= 6:
                    _log(f"[worker] acked message {received}")

            elif kind == "epoch_end":
                targets = torch.cat(valid_targets)
                scores, positives = f1_scores(targets, torch.cat(valid_logits))
                low_scores, _ = f1_scores(targets, torch.cat(lowres_logits))
                clean_macro = macro_f1(scores, positives)
                lowres_macro = macro_f1(low_scores, positives)
                metrics = {
                    "train_loss": loss_total / train_boxes,
                    "action_macro_f1": clean_macro,
                    "action_macro_f1_lowres": lowres_macro,
                    **{f"{name}_f1": score for name, score in zip(ACTION_NAMES, scores)},
                    **{f"{name}_f1_lowres": score for name, score in zip(ACTION_NAMES, low_scores)},
                    # Without these a 0.000 cannot be told from "this class had no positives".
                    **{f"{name}_valid_positives": float(count)
                       for name, count in zip(ACTION_NAMES, positives)},
                    "train_boxes": float(train_boxes),
                    "valid_boxes": float(valid_boxes),
                }
                # Selecting on the clean tier alone would throw away the robustness the
                # resolution augmentation just paid for.
                selection = (clean_macro + lowres_macro) / 2
                if selection >= best_score:
                    best_score = selection
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                    best_metrics = {**metrics, "selection_score": selection}
                send_message(stdout, {
                    "type": "metrics", "epoch": header["epoch"], "epochs": epochs, "metrics": metrics,
                })
                loss_total = 0.0
                valid_logits.clear()
                lowres_logits.clear()
                valid_targets.clear()

            elif kind == "finish":
                break

        if best_state is None:
            send_message(stdout, {"type": "error", "message": "沒有完成任何 epoch。"})
            return 1

        buffer = BytesIO()
        torch.save({
            "model": best_state,
            "architecture": "mobilenet_v3_small",
            "actions": ACTION_NAMES,
            "image_size": int(config["image_size"]),
            "metrics": best_metrics,
            "trained_on": torch.cuda.get_device_name(0),
        }, buffer)
        send_message(stdout, {"type": "checkpoint", "metrics": best_metrics}, buffer.getvalue())
        return 0
    except Exception as exc:  # surface the remote failure to the feeder before dying
        import traceback

        traceback.print_exc(file=sys.stderr)
        try:
            send_message(stdout, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        except OSError:
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    feeder = subparsers.add_parser("feed", help="stream local crops to a remote CUDA host")
    feeder.add_argument("--sharer", action="append", metavar="NAME", default=None,
                        help="repeat per sharer; omit to train on every annotated dataset")
    feeder.add_argument("--host", required=True, help="ssh target, e.g. nolan@192.168.31.128")
    feeder.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    feeder.add_argument("--epochs", type=int, default=24)
    feeder.add_argument("--batch-size", type=int, default=16)
    feeder.add_argument("--prefetch", type=int, default=4, help="batches allowed in flight")
    feeder.add_argument("--ack-timeout", type=float, default=300.0,
                        help="fail instead of hanging if the worker goes quiet this long")
    feeder.add_argument("--seed", type=int, default=42)
    feeder.add_argument("--data-dir", default="var")

    subparsers.add_parser("serve", help="remote worker; started over ssh by feed")

    args = parser.parse_args(argv)
    return _feed_cli(args) if args.command == "feed" else serve()


if __name__ == "__main__":
    raise SystemExit(main())

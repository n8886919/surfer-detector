from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from surf_track import __version__
from surf_track.config import Settings
from surf_track.drive import DriveApiError, DriveLinkError, GDownDriveClient, PublicDriveClient, parse_drive_link
from surf_track.facebook import FacebookLinkError, parse_facebook_link
from surf_track.facebook.links import FACEBOOK_HOSTS
from surf_track.ingest import IngestError, IngestManager
from surf_track.store import StoreError, SurfTrackStore
from surf_track.training import TrainingError, TrainingManager


WEB_ROOT = Path(__file__).resolve().parent / "web"
MAX_REQUEST_BYTES = 16 * 1024


class SurfTrackServer(ThreadingHTTPServer):
    settings: Settings
    store: SurfTrackStore
    ingest_manager: IngestManager
    training_manager: TrainingManager


class RequestHandler(BaseHTTPRequestHandler):
    server_version = f"SurfTrack/{__version__}"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path == "/api/v1/health":
            self._send_json({"status": "ok", "version": __version__})
            return
        if path == "/api/v1/config":
            self._send_json(self.server.settings.public_config())  # type: ignore[attr-defined]
            return
        if path == "/api/v1/workspace":
            self._send_json(self.server.store.workspace_snapshot())  # type: ignore[attr-defined]
            return
        if path == "/api/v1/training-preview":
            try:
                result = self.server.training_manager.preview()  # type: ignore[attr-defined]
                self._send_json(result)
            except (TrainingError, StoreError) as exc:
                self._send_json(
                    {"error": {"code": "PREVIEW_UNAVAILABLE", "message": str(exc)}},
                    HTTPStatus.CONFLICT,
                )
            return
        dataset_match = re.fullmatch(r"/api/v1/datasets/([A-Za-z0-9_-]+)/images", path)
        if dataset_match:
            images = self.server.store.list_label_images(dataset_match.group(1))  # type: ignore[attr-defined]
            self._send_json({"images": images})
            return
        annotation_match = re.fullmatch(r"/api/v1/images/([A-Za-z0-9_-]+)/annotations", path)
        if annotation_match:
            try:
                result = self.server.store.get_annotations(annotation_match.group(1))  # type: ignore[attr-defined]
                self._send_json(result)
            except StoreError as exc:
                self._send_json({"error": {"code": "NOT_FOUND", "message": str(exc)}}, HTTPStatus.NOT_FOUND)
            return
        image_match = re.fullmatch(r"/api/v1/images/([A-Za-z0-9_-]+)/content", path)
        if image_match:
            try:
                image_path = self.server.store.get_image_path(image_match.group(1))  # type: ignore[attr-defined]
                self._send_file(image_path, cache_control="private, max-age=300")
            except StoreError as exc:
                self._send_json({"error": {"code": "NOT_FOUND", "message": str(exc)}}, HTTPStatus.NOT_FOUND)
            return
        thumbnail_match = re.fullmatch(r"/api/v1/images/([A-Za-z0-9_-]+)/thumbnail", path)
        if thumbnail_match:
            try:
                thumbnail = self.server.training_manager.thumbnail_path(thumbnail_match.group(1))  # type: ignore[attr-defined]
                self._send_file(Path(thumbnail), cache_control="private, max-age=3600")
            except (StoreError, OSError) as exc:
                self._send_json({"error": {"code": "NOT_FOUND", "message": str(exc)}}, HTTPStatus.NOT_FOUND)
            return

        assets = {
            "/": WEB_ROOT / "index.html",
            "/index.html": WEB_ROOT / "index.html",
            "/assets/styles.css": WEB_ROOT / "styles.css",
            "/assets/app.js": WEB_ROOT / "app.js",
        }
        asset = assets.get(path)
        if asset is None or not asset.is_file():
            self._send_json({"error": {"code": "NOT_FOUND", "message": "找不到這個頁面。"}}, HTTPStatus.NOT_FOUND)
            return
        self._send_file(asset)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path == "/api/v1/sources/inspect":
            self._inspect_source()
            return
        if path == "/api/v1/drive/scan":
            self._inspect_source(force_provider="drive")
            return
        if path == "/api/v1/datasets":
            self._create_dataset()
            return
        ingest_match = re.fullmatch(r"/api/v1/datasets/([A-Za-z0-9_-]+)/ingest", path)
        if ingest_match:
            try:
                job = self.server.ingest_manager.start(ingest_match.group(1))  # type: ignore[attr-defined]
                self._send_json({"job": job}, HTTPStatus.ACCEPTED)
            except (IngestError, StoreError) as exc:
                self._send_json(
                    {"error": {"code": "INGEST_NOT_STARTED", "message": str(exc)}},
                    HTTPStatus.CONFLICT,
                )
            return
        if path == "/api/v1/train":
            self._start_training()
            return
        self._send_json({"error": {"code": "NOT_FOUND", "message": "找不到這個 API。"}}, HTTPStatus.NOT_FOUND)

    def _start_training(self) -> None:
        """Training always runs on a remote CUDA host; there is no local fallback."""
        try:
            payload = self._read_json()
            host = payload.get("host")
            if not isinstance(host, str) or not host.strip():
                raise TrainingError("請先填入遠端 GPU 主機，格式為 user@host。")
            sharers = payload.get("sharers")
            if not isinstance(sharers, list) or not all(isinstance(name, str) for name in sharers):
                raise TrainingError("請至少勾選一位分享者的資料集。")
            run = self.server.training_manager.start(sharers, host=host)  # type: ignore[attr-defined]
            self._send_json({"run": run}, HTTPStatus.ACCEPTED)
        except (TrainingError, StoreError) as exc:
            self._send_json(
                {"error": {"code": "TRAINING_NOT_STARTED", "message": str(exc)}},
                HTTPStatus.CONFLICT,
            )
        except (ValueError, json.JSONDecodeError):
            self._send_json(
                {"error": {"code": "INVALID_JSON", "message": "請求內容不是有效的 JSON。"}},
                HTTPStatus.BAD_REQUEST,
            )

    def _create_dataset(self) -> None:
        try:
            payload = self._read_json()
            required = ("name", "source_provider", "source_url", "sharer_name")
            if any(not isinstance(payload.get(field), str) for field in required):
                raise StoreError("Dataset 名稱、分享者及來源資訊不可缺少。")
            video_count = payload.get("source_video_count", 0)
            if not isinstance(video_count, int):
                raise StoreError("影片數量必須是整數。")
            provider = str(payload["source_provider"])
            raw_source_url = str(payload["source_url"])
            if provider == "google_drive":
                source_url = parse_drive_link(raw_source_url).canonical_url
            elif provider == "facebook":
                source_url = parse_facebook_link(raw_source_url).canonical_url
            else:
                raise StoreError("來源必須是 Google Drive 或 Facebook。")
            dataset = self.server.store.create_dataset(  # type: ignore[attr-defined]
                str(payload["name"]),
                source_provider=provider,
                source_url=source_url,
                source_title=str(payload.get("source_title") or ""),
                sharer_name=str(payload["sharer_name"]),
                source_video_count=video_count,
            )
            self._send_json({"dataset": dataset}, HTTPStatus.CREATED)
        except (StoreError, DriveLinkError, FacebookLinkError) as exc:
            self._send_json(
                {"error": {"code": "INVALID_DATASET", "message": str(exc)}},
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        except (ValueError, json.JSONDecodeError):
            self._send_json(
                {"error": {"code": "INVALID_JSON", "message": "請求內容不是有效的 JSON。"}},
                HTTPStatus.BAD_REQUEST,
            )

    def do_PUT(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        annotation_match = re.fullmatch(r"/api/v1/images/([A-Za-z0-9_-]+)/annotations", path)
        if not annotation_match:
            self._send_json({"error": {"code": "NOT_FOUND", "message": "找不到這個 API。"}}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
            annotations = payload.get("annotations")
            if not isinstance(annotations, list):
                raise StoreError("annotations 必須是陣列。")
            crop_region = payload.get("crop_region")
            if crop_region is not None and not isinstance(crop_region, dict):
                raise StoreError("crop_region 必須是物件或 null。")
            result = self.server.store.save_annotations(  # type: ignore[attr-defined]
                annotation_match.group(1),
                annotations,
                complete_for_detection=bool(annotations),
                crop_region=crop_region,
            )
            self._send_json(result)
        except StoreError as exc:
            self._send_json(
                {"error": {"code": "INVALID_ANNOTATION", "message": str(exc)}},
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        except (ValueError, json.JSONDecodeError):
            self._send_json(
                {"error": {"code": "INVALID_JSON", "message": "請求內容不是有效的 JSON。"}},
                HTTPStatus.BAD_REQUEST,
            )

    def _inspect_source(self, *, force_provider: str | None = None) -> None:
        try:
            payload = self._read_json()
            raw_url = payload.get("url")
            if not isinstance(raw_url, str):
                raise ValueError("missing URL")
            normalized_url = raw_url.strip()
            if normalized_url.startswith(("facebook.com/", "www.facebook.com/", "m.facebook.com/")):
                normalized_url = f"https://{normalized_url}"
            host = (urlparse(normalized_url).hostname or "").lower()

            if force_provider != "drive" and host in FACEBOOK_HOSTS:
                facebook_link = parse_facebook_link(raw_url)
                self._send_json(
                    {
                        "state": "auth_required",
                        "source": facebook_link.as_dict(),
                        "videos": [],
                        "summary": {"video_count": 0, "items_scanned": 0, "total_bytes": 0},
                        "security": {
                            "credentials_received": False,
                            "cookies_read": False,
                            "planned_auth_mode": "dedicated_local_browser_profile",
                        },
                        "message": "Facebook 連結已安全收下；追蹤參數已移除。尚未讀取貼文內容或你的登入狀態。",
                    }
                )
                return

            if host not in {"drive.google.com", "www.drive.google.com"}:
                self._send_json(
                    {
                        "error": {
                            "code": "UNSUPPORTED_SOURCE",
                            "message": "目前接受 Google Drive 或 Facebook 貼文連結。",
                        }
                    },
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                )
                return

            result = self._drive_scan_result(raw_url)
            self._send_json(result)
        except FacebookLinkError as exc:
            self._send_json(
                {"error": {"code": "INVALID_FACEBOOK_LINK", "message": str(exc)}},
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        except DriveLinkError as exc:
            self._send_json(
                {"error": {"code": "INVALID_DRIVE_LINK", "message": str(exc)}},
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        except DriveApiError as exc:
            self._send_json(
                {"error": {"code": "DRIVE_SCAN_FAILED", "message": str(exc)}},
                HTTPStatus.BAD_GATEWAY,
            )
        except (ValueError, json.JSONDecodeError):
            self._send_json(
                {"error": {"code": "INVALID_JSON", "message": "請求內容不是有效的 JSON。"}},
                HTTPStatus.BAD_REQUEST,
            )

    def _drive_scan_result(self, raw_url: str) -> dict[str, object]:
            link = parse_drive_link(raw_url)
            settings = self.server.settings  # type: ignore[attr-defined]
            backend = settings.resolved_drive_backend()
            if backend == "validate":
                return {
                    "state": "link_ready",
                    "source": {"provider": "google_drive", **link.as_dict()},
                    "videos": [],
                    "summary": {"video_count": 0, "items_scanned": 0, "folders_scanned": 0, "total_bytes": 0},
                    "setup_required": "gdown>=6.1",
                    "message": "連結格式正確。安裝專案依賴後即可直接列出公開影片。",
                }
            if backend == "gdown":
                result = GDownDriveClient().scan(link)
            elif backend == "api_key" and settings.google_api_key:
                result = PublicDriveClient(settings.google_api_key).scan(link)
            else:
                raise DriveApiError(f"不支援的 Drive backend：{backend}")
            result_source = result.get("source")
            if isinstance(result_source, dict):
                result_source["provider"] = "google_drive"
            return result

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            raise ValueError("invalid content length")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, *, cache_control: str = "no-cache") -> None:
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") or content_type == "application/javascript" else content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; style-src-attr 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        )

    def log_message(self, message: str, *args: object) -> None:
        print(f"{self.address_string()} - {message % args}")


def build_server(settings: Settings | None = None) -> SurfTrackServer:
    effective_settings = settings or Settings.from_env()
    server = SurfTrackServer((effective_settings.host, effective_settings.port), RequestHandler)
    server.settings = effective_settings
    server.store = SurfTrackStore(effective_settings.data_dir)
    server.ingest_manager = IngestManager(server.store, effective_settings)
    server.training_manager = TrainingManager(server.store)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SurfTrack local UI.")
    parser.add_argument("--host", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Bind port (default: 8080)")
    args = parser.parse_args()
    settings = Settings.from_env(host=args.host, port=args.port)
    server = build_server(settings)
    print(f"SurfTrack UI: http://{settings.host}:{server.server_port}")
    print(f"Drive scan: {settings.resolved_drive_backend()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSurfTrack stopped.")
    finally:
        server.server_close()

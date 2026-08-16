from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from surf_track.drive.links import DriveLink, DriveResourceKind


DRIVE_API_ROOT = "https://www.googleapis.com/drive/v3/files"
FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
VIDEO_EXTENSIONS = {".3gp", ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


class DriveApiError(RuntimeError):
    """A safe, user-facing Google Drive API error."""


@dataclass(slots=True)
class GDownDriveClient:
    """List public Drive links without requiring a Google account or API key."""

    def scan(self, link: DriveLink) -> dict[str, object]:
        scan_root = Path.cwd() / ".surftrack-drive-scan"
        try:
            import gdown

            if link.kind is DriveResourceKind.FOLDER:
                files = gdown.download_folder(
                    url=link.canonical_url,
                    output=f"{scan_root}/",
                    quiet=True,
                    skip_download=True,
                    use_cookies=False,
                )
            else:
                file = gdown.download(
                    url=link.canonical_url,
                    quiet=True,
                    skip_download=True,
                    use_cookies=False,
                )
                files = [file]
        except Exception as exc:
            message = str(exc).strip()
            if "permission" in message.lower() or "access" in message.lower():
                safe_message = "無法讀取這個 Drive 連結；請確認它已設為知道連結的任何人都能下載。"
            else:
                safe_message = "無法列出這個公開 Drive 來源；可能是 Google 暫時限流，請稍後再試。"
            raise DriveApiError(safe_message) from exc

        videos: list[dict[str, object]] = []
        discovered = list(files or [])
        for item in discovered:
            path = str(getattr(item, "path", ""))
            if PurePosixPath(path.lower()).suffix not in VIDEO_EXTENSIONS:
                continue
            videos.append(
                {
                    "id": str(getattr(item, "id", "")),
                    "name": PurePosixPath(path).name,
                    "path": path,
                    "mime_type": "video/unknown",
                    "size_bytes": 0,
                    "duration_ms": None,
                    "width": None,
                    "height": None,
                    "modified_time": None,
                    "web_view_link": f"https://drive.google.com/file/d/{getattr(item, 'id', '')}/view",
                }
            )

        root_name = "Google Drive"
        paths = [PurePosixPath(str(getattr(item, "path", ""))) for item in discovered]
        local_paths = [Path(str(getattr(item, "local_path", ""))) for item in discovered]
        if link.kind is DriveResourceKind.FOLDER and local_paths:
            try:
                root_name = local_paths[0].relative_to(scan_root).parts[0]
            except (ValueError, IndexError):
                if paths and len(paths[0].parts) > 1:
                    root_name = paths[0].parts[0]
        elif paths:
            root_name = paths[0].name

        return {
            "state": "scanned",
            "source": {
                **link.as_dict(),
                "name": root_name,
                "resolved_kind": link.kind.value,
            },
            "videos": videos,
            "summary": {
                "video_count": len(videos),
                "items_scanned": len(discovered),
                "folders_scanned": None,
                "total_bytes": 0,
            },
        }


def is_video_file(item: dict[str, object]) -> bool:
    mime_type = str(item.get("mimeType") or "").lower()
    name = str(item.get("name") or "")
    return mime_type.startswith("video/") or PurePosixPath(name.lower()).suffix in VIDEO_EXTENSIONS


@dataclass(slots=True)
class PublicDriveClient:
    """Read public Drive metadata through the supported Drive v3 API."""

    api_key: str
    timeout_seconds: float = 20.0
    max_items: int = 10_000

    def _get_json(self, url: str, parameters: dict[str, str]) -> dict[str, object]:
        parameters = {**parameters, "key": self.api_key}
        request = Request(f"{url}?{urlencode(parameters)}", headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in {401, 403, 404}:
                message = "Drive 無法讀取這個連結；請確認它已設為知道連結的任何人都能檢視與下載。"
            elif exc.code == 429:
                message = "Google Drive API 暫時超過額度，請稍後再試。"
            else:
                message = f"Google Drive API 回傳 HTTP {exc.code}。"
            raise DriveApiError(message) from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DriveApiError("目前無法連線到 Google Drive API，請檢查網路後重試。") from exc

    def get_item(self, resource_id: str) -> dict[str, object]:
        fields = "id,name,mimeType,size,modifiedTime,videoMediaMetadata,webViewLink,shortcutDetails"
        return self._get_json(f"{DRIVE_API_ROOT}/{resource_id}", {"fields": fields})

    def iter_children(self, folder_id: str) -> Iterator[dict[str, object]]:
        page_token: str | None = None
        while True:
            parameters = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "pageSize": "1000",
                "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime,videoMediaMetadata,webViewLink,shortcutDetails)",
                "orderBy": "folder,name",
            }
            if page_token:
                parameters["pageToken"] = page_token
            payload = self._get_json(DRIVE_API_ROOT, parameters)
            for item in payload.get("files", []):
                if isinstance(item, dict):
                    yield item
            next_page = payload.get("nextPageToken")
            if not isinstance(next_page, str) or not next_page:
                return
            page_token = next_page

    def scan(self, link: DriveLink) -> dict[str, object]:
        root = self.get_item(link.resource_id)
        root_mime = str(root.get("mimeType") or "")
        videos: list[dict[str, object]] = []
        folders_scanned = 0
        items_scanned = 0

        if root_mime != FOLDER_MIME:
            items_scanned = 1
            if is_video_file(root):
                videos.append(self._normalize_video(root, path=str(root.get("name") or "video")))
        else:
            pending: list[tuple[str, str]] = [(link.resource_id, str(root.get("name") or "Drive"))]
            while pending:
                folder_id, folder_path = pending.pop()
                folders_scanned += 1
                for item in self.iter_children(folder_id):
                    items_scanned += 1
                    if items_scanned > self.max_items:
                        raise DriveApiError(
                            f"掃描超過 {self.max_items:,} 個項目，請改用較小的公開資料夾。"
                        )
                    name = str(item.get("name") or "未命名")
                    item_path = f"{folder_path}/{name}"
                    mime_type = str(item.get("mimeType") or "")
                    if mime_type == FOLDER_MIME:
                        pending.append((str(item["id"]), item_path))
                    elif mime_type == SHORTCUT_MIME:
                        shortcut = item.get("shortcutDetails")
                        if isinstance(shortcut, dict) and shortcut.get("targetMimeType") == FOLDER_MIME:
                            pending.append((str(shortcut["targetId"]), item_path))
                    elif is_video_file(item):
                        videos.append(self._normalize_video(item, path=item_path))

        total_bytes = sum(int(video.get("size_bytes") or 0) for video in videos)
        return {
            "state": "scanned",
            "source": {
                **link.as_dict(),
                "name": str(root.get("name") or "Google Drive"),
                "resolved_kind": "folder" if root_mime == FOLDER_MIME else "file",
            },
            "videos": videos,
            "summary": {
                "video_count": len(videos),
                "items_scanned": items_scanned,
                "folders_scanned": folders_scanned,
                "total_bytes": total_bytes,
            },
        }

    @staticmethod
    def _normalize_video(item: dict[str, object], *, path: str) -> dict[str, object]:
        metadata = item.get("videoMediaMetadata")
        duration_ms = None
        width = None
        height = None
        if isinstance(metadata, dict):
            duration_ms = metadata.get("durationMillis")
            width = metadata.get("width")
            height = metadata.get("height")
        return {
            "id": str(item.get("id") or ""),
            "name": str(item.get("name") or "未命名影片"),
            "path": path,
            "mime_type": str(item.get("mimeType") or "application/octet-stream"),
            "size_bytes": int(item.get("size") or 0),
            "duration_ms": int(duration_ms) if duration_ms is not None else None,
            "width": int(width) if width is not None else None,
            "height": int(height) if height is not None else None,
            "modified_time": item.get("modifiedTime"),
            "web_view_link": item.get("webViewLink"),
        }

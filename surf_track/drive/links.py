from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from urllib.parse import parse_qs, urlparse


_DRIVE_ID = re.compile(r"^[A-Za-z0-9_-]{10,}$")
_ALLOWED_HOSTS = {"drive.google.com", "www.drive.google.com"}


class DriveLinkError(ValueError):
    """Raised when a value is not a supported Google Drive share URL."""


class DriveResourceKind(str, Enum):
    FILE = "file"
    FOLDER = "folder"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DriveLink:
    resource_id: str
    kind: DriveResourceKind
    original_url: str

    @property
    def canonical_url(self) -> str:
        if self.kind is DriveResourceKind.FOLDER:
            return f"https://drive.google.com/drive/folders/{self.resource_id}"
        if self.kind is DriveResourceKind.FILE:
            return f"https://drive.google.com/file/d/{self.resource_id}/view"
        return f"https://drive.google.com/open?id={self.resource_id}"

    def as_dict(self) -> dict[str, str]:
        return {
            "resource_id": self.resource_id,
            "kind": self.kind.value,
            "canonical_url": self.canonical_url,
        }


def parse_drive_link(value: str) -> DriveLink:
    raw = value.strip()
    if not raw:
        raise DriveLinkError("請貼上 Google Drive 分享連結。")

    if raw.startswith("drive.google.com/") or raw.startswith("www.drive.google.com/"):
        raw = f"https://{raw}"

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _ALLOWED_HOSTS:
        raise DriveLinkError("目前只接受 drive.google.com 的分享連結。")

    segments = [segment for segment in parsed.path.split("/") if segment]
    resource_id: str | None = None
    kind = DriveResourceKind.UNKNOWN

    if "folders" in segments:
        index = segments.index("folders")
        if index + 1 < len(segments):
            resource_id = segments[index + 1]
            kind = DriveResourceKind.FOLDER
    elif "d" in segments:
        index = segments.index("d")
        if index + 1 < len(segments):
            resource_id = segments[index + 1]
            kind = DriveResourceKind.FILE

    if not resource_id:
        query_id = parse_qs(parsed.query).get("id", [None])[0]
        resource_id = query_id

    if not resource_id or not _DRIVE_ID.fullmatch(resource_id):
        raise DriveLinkError("連結中找不到有效的 Google Drive 檔案或資料夾 ID。")

    return DriveLink(resource_id=resource_id, kind=kind, original_url=value)

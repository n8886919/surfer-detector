import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from surf_track.drive.client import GDownDriveClient, is_video_file
from surf_track.drive.links import DriveLinkError, DriveResourceKind, parse_drive_link


def test_parse_folder_link() -> None:
    link = parse_drive_link("https://drive.google.com/drive/folders/1AbCdEfGhijKLMnOP")
    assert link.resource_id == "1AbCdEfGhijKLMnOP"
    assert link.kind is DriveResourceKind.FOLDER


def test_parse_file_link() -> None:
    link = parse_drive_link("https://drive.google.com/file/d/1AbCdEfGhijKLMnOP/view?usp=sharing")
    assert link.resource_id == "1AbCdEfGhijKLMnOP"
    assert link.kind is DriveResourceKind.FILE


def test_parse_open_link_has_unknown_kind() -> None:
    link = parse_drive_link("https://drive.google.com/open?id=1AbCdEfGhijKLMnOP")
    assert link.kind is DriveResourceKind.UNKNOWN


@pytest.mark.parametrize(
    "value",
    ["", "https://example.com/file/d/1AbCdEfGhijKLMnOP", "https://drive.google.com/drive/folders/no"],
)
def test_reject_invalid_drive_link(value: str) -> None:
    with pytest.raises(DriveLinkError):
        parse_drive_link(value)


def test_video_detection_prefers_mime_but_accepts_known_extensions() -> None:
    assert is_video_file({"name": "clip.bin", "mimeType": "video/mp4"})
    assert is_video_file({"name": "clip.MOV", "mimeType": "application/octet-stream"})
    assert not is_video_file({"name": "notes.txt", "mimeType": "text/plain"})


def test_gdown_scan_is_anonymous_and_keeps_only_videos(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def download_folder(**kwargs):
        calls.update(kwargs)
        root = Path(str(kwargs["output"])) / "真正的資料夾名稱"
        return [
            SimpleNamespace(id="video-id", path="Session/ride.MP4", local_path=root / "Session/ride.MP4"),
            SimpleNamespace(id="note-id", path="Session/notes.txt", local_path=root / "Session/notes.txt"),
        ]

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download_folder=download_folder))
    link = parse_drive_link("https://drive.google.com/drive/folders/1AbCdEfGhijKLMnOP")
    result = GDownDriveClient().scan(link)

    assert calls["use_cookies"] is False
    assert calls["skip_download"] is True
    assert result["source"]["name"] == "真正的資料夾名稱"
    assert result["summary"]["items_scanned"] == 2
    assert [video["name"] for video in result["videos"]] == ["ride.MP4"]

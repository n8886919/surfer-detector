from __future__ import annotations

import json
import threading
from urllib.request import Request, urlopen

import pytest

from surf_track.config import Settings
from surf_track.server import build_server


@pytest.fixture
def local_server(tmp_path):
    server = build_server(
        Settings(host="127.0.0.1", port=0, drive_backend="validate", data_dir=tmp_path / "var")
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def read_json(url: str) -> dict[str, object]:
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def test_health(local_server: str) -> None:
    assert read_json(f"{local_server}/api/v1/health")["status"] == "ok"


def test_config_version_matches_server(local_server: str) -> None:
    assert read_json(f"{local_server}/api/v1/config")["version"] == "0.2.0"


def test_index_is_served(local_server: str) -> None:
    with urlopen(f"{local_server}/") as response:
        body = response.read().decode("utf-8")
    assert response.status == 200
    assert "SurfTrack" in body


def test_drive_link_validation_without_api_key(local_server: str) -> None:
    body = json.dumps({"url": "https://drive.google.com/drive/folders/1AbCdEfGhijKLMnOP"}).encode()
    request = Request(
        f"{local_server}/api/v1/drive/scan",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    assert payload["state"] == "link_ready"
    assert payload["source"]["kind"] == "folder"
    assert payload["setup_required"] == "gdown>=6.1"


def test_facebook_source_is_sanitized_without_reading_credentials(local_server: str) -> None:
    raw_url = (
        "https://www.facebook.com/groups/1994541457434538/posts/4401885313366795/"
        "?__cft__[0]=tracking-secret&__tn__=feed"
    )
    body = json.dumps({"url": raw_url}).encode()
    request = Request(
        f"{local_server}/api/v1/sources/inspect",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        response_body = response.read().decode("utf-8")
        payload = json.loads(response_body)

    assert payload["state"] == "auth_required"
    assert payload["source"]["provider"] == "facebook"
    assert payload["source"]["canonical_url"].endswith("/posts/4401885313366795/")
    assert payload["security"]["credentials_received"] is False
    assert "tracking-secret" not in response_body


def test_create_dataset_persists_source_attribution(local_server: str) -> None:
    body = json.dumps(
        {
            "name": "north-coast-001",
            "source_provider": "google_drive",
            "source_url": "https://drive.google.com/drive/folders/1AbCdEfGhijKLMnOP",
            "source_title": "North coast clips",
            "sharer_name": "測試分享者",
            "source_video_count": 48,
        }
    ).encode()
    request = Request(
        f"{local_server}/api/v1/datasets",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    assert response.status == 201
    assert payload["dataset"]["sharer_name"] == "測試分享者"

    workspace = read_json(f"{local_server}/api/v1/workspace")
    dataset = workspace["datasets"][0]
    assert dataset["source_provider"] == "google_drive"
    assert dataset["source_video_count"] == 48


def test_annotation_api_persists_partial_labels(tmp_path) -> None:
    server = build_server(
        Settings(host="127.0.0.1", port=0, drive_backend="validate", data_dir=tmp_path / "var")
    )
    dataset = server.store.create_dataset(
        "api-pilot",
        source_provider="google_drive",
        source_url="https://drive.google.com/drive/folders/1AbCdEfGhijKLMnOP",
        sharer_name="測試分享者",
    )
    image_path = server.store.media_dir / "frame.svg"
    image_path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"/>')
    image = server.store.add_image(dataset["id"], image_path, source_group="clip-1", split="valid")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        images = read_json(f"{base_url}/api/v1/datasets/{dataset['id']}/images")["images"]
        assert images[0]["id"] == image["id"]
        payload = {
            "annotations": [
                {
                    "id": "box-1",
                    "x": 0.1,
                    "y": 0.2,
                    "width": 0.3,
                    "height": 0.4,
                    "chasing_wave": True,
                    "takeoff": False,
                    "surfing": False,
                }
            ],
            "crop_region": {"x": 0.05, "y": 0.1, "width": 0.5, "height": 0.6},
        }
        request = Request(
            f"{base_url}/api/v1/images/{image['id']}/annotations",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urlopen(request) as response:
            saved = json.loads(response.read().decode("utf-8"))
        assert saved["annotation_count"] == 1
        restored = read_json(f"{base_url}/api/v1/images/{image['id']}/annotations")
        assert restored["annotations"][0]["chasing_wave"] is True
        assert "paddling" not in restored["annotations"][0]
        assert restored["complete_for_detection"] is True
        assert restored["crop_region"] == payload["crop_region"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

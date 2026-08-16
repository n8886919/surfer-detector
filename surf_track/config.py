from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from surf_track import __version__


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings kept outside the browser bundle."""

    host: str = "127.0.0.1"
    port: int = 8080
    google_api_key: str | None = None
    drive_backend: str = "auto"
    data_dir: Path = Path("var")
    target_device: str = "Jetson Orin Nano 8GB"
    target_hz: int = 10

    @classmethod
    def from_env(cls, *, host: str | None = None, port: int | None = None) -> "Settings":
        return cls(
            host=host or os.environ.get("SURF_TRACK_HOST", "127.0.0.1"),
            port=port or int(os.environ.get("SURF_TRACK_PORT", "8080")),
            google_api_key=os.environ.get("SURF_TRACK_GOOGLE_API_KEY") or None,
            drive_backend=os.environ.get("SURF_TRACK_DRIVE_BACKEND", "auto"),
            data_dir=Path(os.environ.get("SURF_TRACK_DATA_DIR", "var")),
        )

    def resolved_drive_backend(self) -> str:
        if self.drive_backend != "auto":
            return self.drive_backend
        if find_spec("gdown") is not None:
            return "gdown"
        if self.google_api_key:
            return "api_key"
        return "validate"

    def rclone_command(self) -> list[str] | None:
        rclone = self.data_dir.resolve() / "bin" / "rclone"
        config = self.data_dir.resolve() / "secrets" / "rclone.conf"
        password_file = Path.home() / ".config" / "surftrack" / "rclone-config.pass"
        if not (rclone.is_file() and config.is_file() and password_file.is_file()):
            return None
        return [
            str(rclone),
            "--config",
            str(config),
            "--password-command",
            f"/usr/bin/cat {password_file}",
        ]

    def private_drive_connected(self) -> bool:
        command = self.rclone_command()
        if command is None:
            return False
        try:
            result = subprocess.run(
                [*command, "listremotes"],
                capture_output=True,
                check=False,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and "surftrack-drive:" in result.stdout.splitlines()

    def public_config(self) -> dict[str, object]:
        private_drive_connected = self.private_drive_connected()
        return {
            "app_name": "SurfTrack",
            "version": __version__,
            "target_device": self.target_device,
            "target_hz": self.target_hz,
            "stream_count": 1,
            "bbox_target": "surfer_body",
            "action_labels": ["追浪", "起乘", "衝浪"],
            "drive_scan_configured": self.resolved_drive_backend() in {"gdown", "api_key"},
            "drive_scan_backend": self.resolved_drive_backend(),
            "private_drive_connected": private_drive_connected,
            "private_drive_backend": "rclone" if private_drive_connected else "not_configured",
            "credential_storage": "local_private_file",
        }

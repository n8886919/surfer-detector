#!/usr/bin/env python3
"""Single local entry point for SurfTrack."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def use_project_virtualenv() -> None:
    root = Path(__file__).resolve().parent
    virtualenv_root = root / ".venv"
    virtual_python = root / ".venv" / "bin" / "python"
    if virtual_python.is_file() and Path(sys.prefix).resolve() != virtualenv_root.resolve():
        os.execv(str(virtual_python), [str(virtual_python), str(Path(__file__).resolve()), *sys.argv[1:]])


use_project_virtualenv()

from surf_track.server import main  # noqa: E402


if __name__ == "__main__":
    main()

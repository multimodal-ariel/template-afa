from __future__ import annotations

import os
from typing import Any, Optional


def add_prefix_to_dict(
    d: dict[str, Any], prefix: str, sep: Optional[str] = "/"
) -> dict[str, Any]:
    return {f"{prefix}{sep}{k}": v for k, v in d.items()}


def get_project_root_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from __future__ import annotations

import os
from typing import Any, Optional


def add_prefix_to_dict(
    d: dict[str, Any], prefix: str, sep: Optional[str] = "/"
) -> dict[str, Any]:
    """add prefix to dictionary

    Args:
        d (dict[str, Any]): a dictionary
        prefix (str): prefix to be prependes to each key
        sep (Optional[str], optional): seperator between `prefix` and key. Defaults to "/".

    Returns:
        dict[str, Any]: new dictionary with `prefix` prepended to each key of `d`.
    """
    return {f"{prefix}{sep}{k}": v for k, v in d.items()}


def get_project_root_dir() -> str:
    """get the absolute path to the project root

    Returns:
        str: absoluate path to project root.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

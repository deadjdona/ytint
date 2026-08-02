"""Centralised configuration loader for the ytint pipeline.

Replaces the identical ``load_config()`` function that was copy-pasted into
every pipeline stage and both Streamlit apps.  Every module should now do::

    from engine.config_loader import load_config
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Tuple


def _find_project_root(start: Path | None = None) -> Path:
    """Walk upward from *start* until we find the ``config/`` directory."""
    if start is None:
        # Default to caller's directory – works for any file under src/
        start = Path(__file__).resolve().parent

    cursor = start
    while cursor != cursor.parent:
        if (cursor / "config").is_dir():
            return cursor
        cursor = cursor.parent

    raise FileNotFoundError(
        "❌ Critical Configuration Alignment Failure:\n"
        f"Could not locate a 'config/' directory above {start}"
    )


def load_config(*, root_dir: Path | None = None) -> dict:
    """Load ``config/settings.yaml`` and resolve all relative paths.

    Parameters
    ----------
    root_dir : Path, optional
        Explicit project root.  When *None* the root is auto-detected by
        walking upward from this file.

    Returns
    -------
    dict
        The parsed YAML configuration with every ``paths.*`` value
        converted to an absolute :class:`pathlib.Path`.
    """
    if root_dir is None:
        root_dir = _find_project_root()

    config_path = root_dir / "config" / "settings.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"❌ Critical Configuration Alignment Failure:\n"
            f"Could not locate 'config/settings.yaml'.\n"
            f"Resolved root searched: {root_dir}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Translate every relative path to an absolute path anchored to root_dir
    config["paths"]["raw_db"] = str(root_dir / config["paths"]["raw_db"])
    config["paths"]["interim_dir"] = str(root_dir / config["paths"]["interim_dir"])
    config["paths"]["output_dir"] = str(root_dir / config["paths"]["output_dir"])

    # Stash root for callers that need it (e.g. runner.py)
    config["_root_dir"] = root_dir

    return config


def get_paths(config: dict) -> Tuple[Path, Path, Path]:
    """Return ``(raw_db, interim_dir, output_dir)`` as absolute Paths."""
    return (
        Path(config["paths"]["raw_db"]),
        Path(config["paths"]["interim_dir"]),
        Path(config["paths"]["output_dir"]),
    )

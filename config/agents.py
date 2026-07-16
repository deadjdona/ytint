from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:  # pragma: no cover - import error handled at runtime
    yaml = None


DEFAULT_NAMES = [Path(os.getenv("AGENTS_CONFIG", "agents.yaml")), Path("config/agents.yaml")]


def _find_config_path() -> Optional[Path]:
    for p in DEFAULT_NAMES:
        if p.exists():
            return p
    return None


def load_agents(path: Optional[str] = None) -> Dict[str, Any]:
    """Load agents configuration from YAML.

    The loader looks for a YAML file at `path`, then the `AGENTS_CONFIG` env var, then
    `agents.yaml` in the repo root and `config/agents.yaml`.

    If an agent defines `api_key_env`, that environment variable will be read and
    injected into the returned configuration as `api_key`.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with 'pip install pyyaml'")

    config_path = Path(path) if path else _find_config_path()
    if config_path is None:
        raise FileNotFoundError("No agents configuration file found (tried AGENTS_CONFIG, agents.yaml, config/agents.yaml)")

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    agents = data.get("agents", {})
    for name, cfg in agents.items():
        env = cfg.get("api_key_env")
        if env and env in os.environ:
            cfg["api_key"] = os.environ[env]
    return agents


def get_agent(name: str, path: Optional[str] = None) -> Dict[str, Any]:
    agents = load_agents(path)
    if name not in agents:
        raise KeyError(f"Agent '{name}' not found in configuration")
    return agents[name]


__all__ = ["load_agents", "get_agent"]

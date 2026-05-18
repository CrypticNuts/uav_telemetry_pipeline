"""
config.py — Pipeline configuration (paths, defaults).

Resolution order for each external-tool path:

1. Constructor argument (used by tests / programmatic callers)
2. Environment variable ($DRONESECURITY_PATH, $PROTO17_PATH)
3. ``config.yaml`` next to this file, key ``dronesecurity_path`` / ``proto17_path``
4. Hard-coded default under ``~/projects/PFE``

The DroneSecurity decoder is invoked as a subprocess; because that script
imports ``distutils.log`` (removed from Python 3.12+), the subprocess MUST
run inside the DroneSecurity venv. This module exposes the resolved venv
interpreter path via ``PipelineConfig.dronesecurity_python``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_HOME = Path.home()
_DEFAULT_DRONESECURITY = _HOME / "projects" / "PFE" / "DroneSecurity"
_DEFAULT_PROTO17 = _HOME / "projects" / "PFE" / "dji_droneid"
_CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"


def _load_yaml_config() -> dict:
    if not _CONFIG_FILE.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.debug("PyYAML not installed; ignoring %s", _CONFIG_FILE)
        return {}
    try:
        data = yaml.safe_load(_CONFIG_FILE.read_text()) or {}
        if not isinstance(data, dict):
            logger.warning("config.yaml is not a mapping; ignoring")
            return {}
        return data
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse config.yaml: %s", exc)
        return {}


@dataclass
class PipelineConfig:
    """Resolved pipeline configuration."""

    dronesecurity_path: Path = field(default_factory=lambda: _DEFAULT_DRONESECURITY)
    proto17_path: Path = field(default_factory=lambda: _DEFAULT_PROTO17)
    results_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "results"
    )

    @classmethod
    def load(
        cls,
        dronesecurity_path: Path | str | None = None,
        proto17_path: Path | str | None = None,
        results_dir: Path | str | None = None,
    ) -> "PipelineConfig":
        yaml_cfg = _load_yaml_config()

        ds = (
            dronesecurity_path
            or os.environ.get("DRONESECURITY_PATH")
            or yaml_cfg.get("dronesecurity_path")
            or _DEFAULT_DRONESECURITY
        )
        p17 = (
            proto17_path
            or os.environ.get("PROTO17_PATH")
            or yaml_cfg.get("proto17_path")
            or _DEFAULT_PROTO17
        )
        res = (
            results_dir
            or os.environ.get("PIPELINE_RESULTS_DIR")
            or yaml_cfg.get("results_dir")
            or (Path(__file__).resolve().parent / "results")
        )

        return cls(
            dronesecurity_path=Path(ds).expanduser().resolve(),
            proto17_path=Path(p17).expanduser().resolve(),
            results_dir=Path(res).expanduser().resolve(),
        )

    @property
    def dronesecurity_src(self) -> Path:
        return self.dronesecurity_path / "src"

    @property
    def dronesecurity_python(self) -> Path:
        """Path to the interpreter inside the DroneSecurity venv.

        Falls back to ``sys.executable`` if the venv is missing. The
        DroneSecurityDecoder will warn (not crash) in that case so users
        see the actionable message instead of a ModuleNotFoundError for
        ``distutils``.
        """
        candidate = self.dronesecurity_path / ".venv" / "bin" / "python"
        if candidate.exists():
            return candidate
        candidate3 = self.dronesecurity_path / ".venv" / "bin" / "python3"
        if candidate3.exists():
            return candidate3
        import sys

        logger.warning(
            "DroneSecurity venv interpreter not found under %s — "
            "falling back to %s. Decoder may crash on Python 3.12+ due to "
            "the removed 'distutils' module.",
            self.dronesecurity_path / ".venv" / "bin",
            sys.executable,
        )
        return Path(sys.executable)

    @property
    def dronesecurity_offline_script(self) -> Path:
        return self.dronesecurity_src / "droneid_receiver_offline.py"

    @property
    def proto17_matlab_dir(self) -> Path:
        return self.proto17_path / "matlab"

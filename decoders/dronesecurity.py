"""
dronesecurity.py — Primary decoder via RUB-SysSec/DroneSecurity subprocess.

Invokes ``droneid_receiver_offline.py`` inside the DroneSecurity venv
(needed because that codebase imports ``distutils.log``, which is gone
from stdlib in Python 3.12+; the venv has setuptools installed which
re-provides the shim).

Stdout of the receiver is a free-text log with embedded JSON blocks of
the form::

    ## Drone-ID Payload ##
    {
        "pkt_len": 88,
        ...
        "crc-packet": "ed5a",
        "crc-calculated": "ed5a"
    }

This module extracts every JSON block and maps it to the pipeline's
canonical ``TelemetryFrame`` shape.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from ..config import PipelineConfig
from .base import BaseDecoder, TelemetryFrame

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\"crc-calculated\"[^{}]*\}", re.DOTALL)


class DroneSecurityDecoder(BaseDecoder):
    """Subprocess wrapper around DroneSecurity's offline receiver.

    Parameters
    ----------
    config : PipelineConfig | None
        Resolved pipeline configuration. If None, a fresh one is loaded
        from env/yaml/defaults.
    timeout_s : float
        Hard wall-clock cap on the subprocess. mini2_sm at 50 Msps takes
        ~3 minutes on a typical laptop; phantom captures at 30.72 Msps
        on multi-GB files can be several times longer.
    extra_args : list[str] | None
        Extra CLI flags appended to the receiver invocation (e.g.
        ``["-l"]`` for legacy drones).
    """

    name = "dronesecurity"

    def __init__(
        self,
        config: PipelineConfig | None = None,
        timeout_s: float = 1800.0,
        extra_args: list[str] | None = None,
    ) -> None:
        self.config = config or PipelineConfig.load()
        self.timeout_s = float(timeout_s)
        self.extra_args = list(extra_args or [])

    def is_available(self) -> bool:
        script = self.config.dronesecurity_offline_script
        if not script.is_file():
            logger.warning("DroneSecurity offline script missing: %s", script)
            return False
        return True

    def decode(self, iq_file_path: str, sample_rate: float) -> list[TelemetryFrame]:
        if not self.is_available():
            return []

        script = self.config.dronesecurity_offline_script
        iq_path = Path(iq_file_path).expanduser().resolve()
        if not iq_path.is_file():
            logger.error("IQ file not found: %s", iq_path)
            return []

        cmd = [
            str(self.config.dronesecurity_python),
            str(script),
            "-i", str(iq_path),
            "-s", str(float(sample_rate)),
            *self.extra_args,
        ]
        logger.info("Running DroneSecurity: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.config.dronesecurity_src,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            logger.error("DroneSecurity timed out after %.0fs", self.timeout_s)
            return []
        except OSError as exc:
            logger.error("Failed to launch DroneSecurity: %s", exc)
            return []

        if proc.returncode != 0:
            logger.warning(
                "DroneSecurity exited %d; stderr tail: %s",
                proc.returncode,
                proc.stderr.strip().splitlines()[-3:] if proc.stderr else "(empty)",
            )

        frames = self._parse_stdout(proc.stdout)
        logger.info(
            "DroneSecurity produced %d frame(s) (%d CRC OK)",
            len(frames),
            sum(1 for f in frames if f.get("crc_ok")),
        )
        return frames

    def _parse_stdout(self, stdout: str) -> list[TelemetryFrame]:
        frames: list[TelemetryFrame] = []
        for match in _JSON_BLOCK_RE.finditer(stdout):
            blob = match.group(0)
            try:
                payload = json.loads(blob)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed payload block")
                continue
            frame = self._map_payload(payload)
            if frame is not None:
                frames.append(frame)
        return frames

    @staticmethod
    def _map_payload(payload: dict) -> TelemetryFrame | None:
        try:
            crc_ok = payload.get("crc-packet") == payload.get("crc-calculated")
            frame: TelemetryFrame = {
                "lat": float(payload.get("latitude", 0.0)),
                "lon": float(payload.get("longitude", 0.0)),
                "altitude_m": float(payload.get("altitude", 0.0)),
                "height_m": float(payload.get("height", 0.0)),
                "vel_north": float(payload.get("v_north", 0)),
                "vel_east": float(payload.get("v_east", 0)),
                "vel_up": float(payload.get("v_up", 0)),
                "yaw": int(payload.get("d_1_angle", 0)),
                "serial": str(payload.get("serial_number", "")).rstrip("\x00 "),
                "device_type": str(payload.get("device_type") or ""),
                "app_lat": float(payload.get("app_lat", 0.0)),
                "app_lon": float(payload.get("app_lon", 0.0)),
                "home_lat": float(payload.get("latitude_home", 0.0)),
                "home_lon": float(payload.get("longitude_home", 0.0)),
                "sequence_number": int(payload.get("sequence_number", 0)),
                "gps_time_ms": int(payload.get("gps_time", 0)),
                "crc_ok": bool(crc_ok),
                "decoder": "dronesecurity",
                "timestamp_sample": -1,
            }
            return frame
        except (TypeError, ValueError) as exc:
            logger.debug("Could not map payload: %s", exc)
            return None

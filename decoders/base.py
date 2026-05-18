"""
base.py — Common interface for all DroneID decoder backends.

Every concrete decoder implements ``decode(iq_file_path, sample_rate) ->
list[TelemetryFrame]`` where ``TelemetryFrame`` is a TypedDict with the
fields requested by the pipeline brief.

Implementations must NOT raise on decode failure; instead they should log
the error and return an empty list. The runner depends on this behavior to
fall back to the next decoder.
"""

from __future__ import annotations

import abc
import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


class TelemetryFrame(TypedDict, total=False):
    """Single decoded DroneID telemetry frame in the pipeline's canonical form."""

    lat: float                # drone latitude (deg). 0.0 = no GPS fix
    lon: float                # drone longitude (deg). 0.0 = no GPS fix
    altitude_m: float         # drone altitude (m, above sea level)
    height_m: float           # drone height above takeoff (m)
    vel_north: float          # velocity north component (decimeters/s, raw)
    vel_east: float           # velocity east component (decimeters/s, raw)
    vel_up: float             # velocity vertical component (decimeters/s, raw)
    yaw: int                  # heading angle (raw drone units)
    serial: str               # drone serial number (16-char ASCII)
    device_type: str          # human-readable drone model
    app_lat: float            # pilot/app latitude
    app_lon: float            # pilot/app longitude
    home_lat: float           # home-point latitude
    home_lon: float           # home-point longitude
    sequence_number: int
    gps_time_ms: int          # GPS time, ms since Unix epoch
    crc_ok: bool              # CRC-16 check result
    decoder: str              # backend identifier ("dronesecurity"/"proto17"/"native")
    timestamp_sample: int     # sample index within the input capture (-1 if unknown)


class BaseDecoder(abc.ABC):
    """Abstract base class for telemetry decoders.

    Concrete subclasses must override :meth:`decode` and :attr:`name`. The
    :meth:`is_available` default returns True; override it to advertise
    runtime prerequisites (e.g. octave installed, venv present).
    """

    name: str = "base"

    @abc.abstractmethod
    def decode(
        self,
        iq_file_path: str,
        sample_rate: float,
    ) -> list[TelemetryFrame]:
        """Decode telemetry frames from an IQ capture.

        Parameters
        ----------
        iq_file_path : str
            Path to the IQ capture file (raw interleaved float32 by default).
        sample_rate : float
            Sample rate of the capture in Hz.

        Returns
        -------
        list[TelemetryFrame]
            One dict per decoded frame. May be empty. Frames with failed
            CRC are still returned with ``crc_ok=False`` so the caller can
            inspect them; the runner counts ``crc_ok=True`` frames only
            when deciding whether to fall through to the next decoder.
        """

    def is_available(self) -> bool:
        """Return False to make the runner skip this decoder."""
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"

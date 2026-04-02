"""
droneid_parser.py — Parse DroneSecurity decoder output into DroneIDFrame objects.

Handles two input formats:
  1. Raw stdout text from droneid_receiver_offline.py — extracts JSON blocks
     that appear after "## Drone-ID Payload ##" markers.
  2. Pre-parsed dict with DroneSecurity field names — for direct integration.

Field mapping is based exclusively on what droneid_packet.py actually emits:
    serial_number, latitude, longitude, altitude, height,
    v_north, v_east, v_up, d_1_angle, gps_time,
    app_lat, app_lon, latitude_home, longitude_home,
    device_type, uuid, sequence_number,
    crc-packet, crc-calculated
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone

from telemetry.models import DroneIDFrame, GPSCoordinate, InputClassification

logger = logging.getLogger(__name__)

# Regex to find JSON blocks following the payload marker.
# The marker line is printed by droneid_receiver_offline.py before each
# json.dumps(self.droneid, indent=4) call.
_PAYLOAD_MARKER = "## Drone-ID Payload ##"
_JSON_BLOCK_RE = re.compile(r"\{[^{}]+\}", re.DOTALL)

# Validation limits
_MAX_SPEED_MS = 200.0      # ~720 km/h — well above any consumer drone
_MAX_ALTITUDE_M = 15000.0  # reasonable ceiling for GPS altitude


class DroneIDParser:
    """Parse DroneSecurity decoder output into structured DroneID frames.

    Parameters
    ----------
    input_classification : InputClassification
        How the source IQ data was classified (propagated into every frame).
    """

    def __init__(
        self,
        input_classification: InputClassification = InputClassification.CASE_A,
    ) -> None:
        self.input_classification = input_classification

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_stdout(self, stdout: str) -> list[DroneIDFrame]:
        """Extract all DroneID frames from raw decoder stdout text.

        Looks for "## Drone-ID Payload ##" markers followed by a JSON
        block.  Each valid JSON block is parsed into a DroneIDFrame.

        Parameters
        ----------
        stdout : str
            Full stdout captured from the decoder subprocess.

        Returns
        -------
        list[DroneIDFrame]
            Successfully parsed frames (may be empty).
        """
        if not stdout or _PAYLOAD_MARKER not in stdout:
            logger.debug("No payload markers found in stdout")
            return []

        frames: list[DroneIDFrame] = []

        # Split on the marker — everything after a marker up to the next
        # marker (or end of string) may contain a JSON block.
        sections = stdout.split(_PAYLOAD_MARKER)[1:]  # drop text before first marker
        logger.debug("Found %d payload section(s) in stdout", len(sections))

        for i, section in enumerate(sections):
            match = _JSON_BLOCK_RE.search(section)
            if not match:
                logger.warning("Payload section %d: no JSON block found", i)
                continue

            json_text = match.group(0)
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Payload section %d: invalid JSON: %s", i, exc
                )
                continue

            frame = self.parse_dict(data)
            if frame is not None:
                frames.append(frame)
                logger.info(
                    "Parsed frame %d: serial=%s model=%s crc=%s",
                    i, frame.serial_number,
                    frame.manufacturer,
                    "OK" if data.get("crc-packet") == data.get("crc-calculated") else "FAIL",
                )

        logger.info(
            "Parsed %d frame(s) from %d payload section(s)",
            len(frames), len(sections),
        )
        return frames

    def parse_dict(self, data: dict) -> DroneIDFrame | None:
        """Parse a DroneSecurity-format dict into a DroneIDFrame.

        Field names match droneid_packet.py exactly — no guessing.

        Parameters
        ----------
        data : dict
            Dictionary with DroneSecurity field names as keys.

        Returns
        -------
        DroneIDFrame | None
            Parsed frame, or None if required fields are missing/invalid.
        """
        try:
            serial = str(data.get("serial_number", "")).strip()
            if not serial:
                logger.warning("Missing serial_number — skipping frame")
                return None

            # --- Device type → manufacturer field ---
            device_type = data.get("device_type")
            manufacturer = f"DJI {device_type}" if device_type else "DJI"

            # --- Drone position ---
            drone_pos = self._build_coordinate(
                lat=data.get("latitude"),
                lon=data.get("longitude"),
                alt=data.get("altitude"),
                label="drone",
            )

            # --- Pilot / app position ---
            pilot_pos = self._build_coordinate(
                lat=data.get("app_lat"),
                lon=data.get("app_lon"),
                alt=None,  # not provided by the decoder
                label="pilot/app",
            )

            # --- Home position ---
            home_pos = self._build_coordinate(
                lat=data.get("latitude_home"),
                lon=data.get("longitude_home"),
                alt=None,
                label="home",
            )

            # --- Velocity ---
            v_north = self._clamp_speed(data.get("v_north"), "v_north")
            v_east = self._clamp_speed(data.get("v_east"), "v_east")
            v_up = self._clamp_speed(data.get("v_up"), "v_up")

            speed_h: float | None = None
            if v_north is not None and v_east is not None:
                speed_h = math.sqrt(v_north ** 2 + v_east ** 2)

            # --- Height AGL ---
            height_agl = self._clamp_optional(
                data.get("height"), -500.0, _MAX_ALTITUDE_M, "height"
            )

            # --- Heading from d_1_angle ---
            heading: float | None = None
            raw_angle = data.get("d_1_angle")
            if raw_angle is not None:
                # d_1_angle is a raw int16; convert to 0–360 degrees
                heading = float(raw_angle) % 360.0

            # --- Timestamp from gps_time ---
            timestamp = self._parse_gps_time(data.get("gps_time"))

            # --- CRC check → decode confidence ---
            crc_ok = (
                data.get("crc-packet") is not None
                and data.get("crc-packet") == data.get("crc-calculated")
            )
            confidence = 1.0 if crc_ok else 0.5

            frame = DroneIDFrame(
                serial_number=serial,
                manufacturer=manufacturer,
                drone_position=drone_pos,
                pilot_position=pilot_pos,
                home_position=home_pos,
                speed_horizontal_ms=speed_h,
                speed_vertical_ms=v_up,
                heading_deg=heading,
                height_agl_m=height_agl,
                timestamp=timestamp,
                input_classification=self.input_classification,
                decode_confidence=confidence,
            )

            return frame

        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Failed to parse DroneID dict: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Coordinate builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_coordinate(
        lat: float | None,
        lon: float | None,
        alt: float | None,
        label: str,
    ) -> GPSCoordinate | None:
        """Build a GPSCoordinate with validation.

        Returns None if lat/lon are missing, zero, or out of range.
        """
        if lat is None or lon is None:
            return None

        lat_f = float(lat)
        lon_f = float(lon)

        # Zero coordinates mean "not set" in DJI's protocol
        if lat_f == 0.0 and lon_f == 0.0:
            logger.debug("Skipping zero %s coordinates", label)
            return None

        if not (-90.0 <= lat_f <= 90.0):
            logger.warning("Invalid %s latitude: %f — skipping", label, lat_f)
            return None
        if not (-180.0 <= lon_f <= 180.0):
            logger.warning("Invalid %s longitude: %f — skipping", label, lon_f)
            return None

        alt_f = float(alt) if alt is not None else 0.0
        if abs(alt_f) > _MAX_ALTITUDE_M:
            logger.warning(
                "Suspicious %s altitude: %.1f m — clamping", label, alt_f
            )
            alt_f = max(-_MAX_ALTITUDE_M, min(_MAX_ALTITUDE_M, alt_f))

        return GPSCoordinate(latitude=lat_f, longitude=lon_f, altitude_m=alt_f)

    # ------------------------------------------------------------------
    # Velocity / numeric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_speed(value: int | float | None, label: str) -> float | None:
        """Validate and return a speed value, or None."""
        if value is None:
            return None
        v = float(value)
        if abs(v) > _MAX_SPEED_MS:
            logger.warning("Suspicious %s: %.1f m/s — clamping", label, v)
            return max(-_MAX_SPEED_MS, min(_MAX_SPEED_MS, v))
        return v

    @staticmethod
    def _clamp_optional(
        value: float | None,
        lo: float,
        hi: float,
        label: str,
    ) -> float | None:
        """Clamp an optional numeric value to [lo, hi]."""
        if value is None:
            return None
        v = float(value)
        if not (lo <= v <= hi):
            logger.warning("Suspicious %s: %.1f — clamping to [%.0f, %.0f]", label, v, lo, hi)
            return max(lo, min(hi, v))
        return v

    @staticmethod
    def _parse_gps_time(gps_time_ms: int | None) -> datetime | None:
        """Convert DJI gps_time (ms since epoch) to a datetime.

        Returns None if the timestamp is missing or clearly invalid.
        """
        if gps_time_ms is None:
            return None
        try:
            ts = int(gps_time_ms)
            # DJI gps_time is milliseconds since Unix epoch
            dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
            # Sanity: reject dates before 2015 or after 2035
            if dt.year < 2015 or dt.year > 2035:
                logger.warning("GPS time out of sane range: %s", dt)
                return None
            return dt
        except (ValueError, OSError, OverflowError) as exc:
            logger.warning("Failed to parse gps_time %s: %s", gps_time_ms, exc)
            return None

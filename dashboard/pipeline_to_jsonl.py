"""
pipeline_to_jsonl.py — Convert a pipeline output JSON to dashboard .jsonl.

The pipeline emits a single JSON document with a ``frames`` list per
recording (see ``results/mavic_air_2_DECODE.json``). The dashboard wants
newline-delimited JSON, one frame per line, in its own schema.

This adapter performs the mapping:

    lat, lon            ->  drone_lat, drone_lon   (null if zero-fix)
    altitude_m          ->  drone_alt
    sqrt(vN^2+vE^2+vU^2)->  drone_speed
    app_lat, app_lon    ->  pilot_lat, pilot_lon   (home_* fallback)
    serial              ->  serial_number
    crc_ok              ->  crc_valid
    gps_time_ms         ->  timestamp (ISO 8601, UTC)
    decoder             ->  decoder

Usage
-----
    python pipeline_to_jsonl.py results/mavic_air_2_DECODE.json -o mavic.jsonl
    python pipeline_to_jsonl.py results/mini2_sm_telemetry.json -o mini2.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _iso_ts(gps_time_ms: int | float | None, fallback_idx: int) -> str:
    """Convert a Unix-ms timestamp to ISO 8601 UTC; fall back to "now+i" ticks."""
    try:
        if gps_time_ms and float(gps_time_ms) > 0:
            return datetime.fromtimestamp(
                float(gps_time_ms) / 1000.0, tz=timezone.utc
            ).isoformat()
    except (TypeError, ValueError):
        pass
    # No usable GPS time — synthesise a monotonically-increasing stamp so the
    # dashboard ordering is still meaningful.
    return datetime.now(timezone.utc).replace(microsecond=fallback_idx % 1000).isoformat()


def _speed_mps(frame: dict) -> float:
    """Magnitude of the (vN, vE, vU) velocity vector if present, else 0."""
    vN = float(frame.get("vel_north", 0.0) or 0.0)
    vE = float(frame.get("vel_east", 0.0) or 0.0)
    vU = float(frame.get("vel_up", 0.0) or 0.0)
    return math.sqrt(vN * vN + vE * vE + vU * vU)


def _valid_latlon(lat: float, lon: float) -> bool:
    """Return True iff (lat, lon) is a non-zero, in-range geographic point."""
    if lat == 0.0 and lon == 0.0:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def _convert_frame(frame: dict, idx: int) -> dict:
    lat = float(frame.get("lat", 0.0) or 0.0)
    lon = float(frame.get("lon", 0.0) or 0.0)
    # Zero-fix or out-of-range frames carry no usable drone position; emit
    # null so the dashboard skips the marker update instead of snapping to
    # (0,0) or to nonsense coords from a CRC-failed corrupt frame.
    has_fix = _valid_latlon(lat, lon)

    pilot_lat = float(frame.get("app_lat") or frame.get("home_lat") or 0.0)
    pilot_lon = float(frame.get("app_lon") or frame.get("home_lon") or 0.0)
    if not _valid_latlon(pilot_lat, pilot_lon):
        pilot_lat = pilot_lon = None

    return {
        "timestamp":     _iso_ts(frame.get("gps_time_ms"), idx),
        "decoder":       frame.get("decoder", "unknown"),
        "drone_lat":     lat if has_fix else None,
        "drone_lon":     lon if has_fix else None,
        "drone_alt":     float(frame.get("altitude_m", 0.0) or 0.0),
        "drone_speed":   _speed_mps(frame),
        "pilot_lat":     pilot_lat,
        "pilot_lon":     pilot_lon,
        "serial_number": frame.get("serial", ""),
        "crc_valid":     bool(frame.get("crc_ok", False)),
        # Extra context fields are ignored by the dashboard but useful to keep
        # in the jsonl for downstream tooling.
        "sequence_number": frame.get("sequence_number"),
        "device_type":   frame.get("device_type"),
    }


def convert(in_path: Path, out_path: Path, skip_no_fix: bool = False) -> int:
    """Read pipeline JSON, write dashboard .jsonl. Returns frames written."""
    doc = json.loads(in_path.read_text(encoding="utf-8"))
    frames = doc.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError(f"{in_path}: 'frames' is not a list")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, fr in enumerate(frames):
            converted = _convert_frame(fr, i)
            if skip_no_fix and converted["drone_lat"] is None:
                continue
            fh.write(json.dumps(converted) + "\n")
            written += 1
    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Pipeline output JSON (e.g. results/mavic_air_2_DECODE.json)")
    p.add_argument("-o", "--out", required=True, help="Output .jsonl path")
    p.add_argument("--skip-no-fix", action="store_true",
                   help="Drop frames that have no drone GPS fix (lat=lon=0)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    in_path  = Path(args.input).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    n = convert(in_path, out_path, skip_no_fix=args.skip_no_fix)
    print(f"[adapter] {in_path.name}: wrote {n} frame(s) -> {out_path}")


if __name__ == "__main__":
    main()

"""
generate_demo.py — Synthetic DroneID telemetry generator for the dashboard.

Produces a ``.jsonl`` file (one JSON object per line) mimicking a DJI
drone flying a circular patrol pattern around a fixed pilot position.
Default output matches the schema consumed by ``dashboard/app.py``.

Usage
-----
    python generate_demo.py                     # write demo.jsonl in CWD
    python generate_demo.py --out demo.jsonl --frames 300
    python generate_demo.py --live              # append every second
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path


# ----------------------------------------------------------------- geography
# Pilot anchored in Tunis, Tunisia (Bardo museum area). The drone orbits at a
# configurable radius and altitude.
DEFAULT_PILOT_LAT = 36.8090
DEFAULT_PILOT_LON = 10.1330
DEFAULT_ALT_M     = 80.0
DEFAULT_RADIUS_M  = 120.0     # patrol radius
DEFAULT_SPEED_MPS = 8.0       # tangential ground speed
SERIAL_NUMBER     = "1WNBH3900201N1"

_METRES_PER_DEG_LAT = 111_320.0
_DECODERS = ("droneid", "fallback")


# ------------------------------------------------------------------- helpers


def _metres_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat_deg))


def _frame(i: int, pilot_lat: float, pilot_lon: float,
           radius_m: float, alt_m: float, speed_mps: float,
           rng: random.Random) -> dict:
    """Return one synthetic telemetry frame in the dashboard's schema."""
    # Angular position on the circular patrol path.
    # Period chosen so that tangential speed matches ``speed_mps`` exactly.
    omega = speed_mps / radius_m  # rad / s
    theta = omega * i
    dlat_m = radius_m * math.sin(theta)
    dlon_m = radius_m * math.cos(theta)
    drone_lat = pilot_lat + dlat_m / _METRES_PER_DEG_LAT
    drone_lon = pilot_lon + dlon_m / _metres_per_deg_lon(pilot_lat)

    # Add light measurement noise for realism.
    drone_lat += rng.gauss(0.0, 5e-6)
    drone_lon += rng.gauss(0.0, 5e-6)
    altitude  = alt_m + rng.gauss(0.0, 0.4)
    speed     = speed_mps + rng.gauss(0.0, 0.2)

    decoder  = _DECODERS[i % len(_DECODERS)]
    # Sprinkle CRC failures (~5%) and bias them slightly to the fallback
    # decoder, which is what we'd expect from a less-robust frame path.
    crc_fail = rng.random() < (0.08 if decoder == "fallback" else 0.03)

    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    return {
        "timestamp": ts,
        "decoder": decoder,
        "drone_lat": drone_lat,
        "drone_lon": drone_lon,
        "drone_alt": altitude,
        "drone_speed": max(0.0, speed),
        "pilot_lat": pilot_lat,
        "pilot_lon": pilot_lon,
        "serial_number": SERIAL_NUMBER,
        "crc_valid": not crc_fail,
    }


# ---------------------------------------------------------------------- main


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", "-o", default="demo.jsonl",
                   help="Output .jsonl file (default: demo.jsonl)")
    p.add_argument("--frames", "-n", type=int, default=300,
                   help="Number of frames to generate (default: 300)")
    p.add_argument("--live", action="store_true",
                   help="Append frames at 1 Hz instead of writing in one shot")
    p.add_argument("--pilot-lat", type=float, default=DEFAULT_PILOT_LAT)
    p.add_argument("--pilot-lon", type=float, default=DEFAULT_PILOT_LON)
    p.add_argument("--altitude", type=float, default=DEFAULT_ALT_M)
    p.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M)
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED_MPS)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.live else "w"
    with out_path.open(mode, encoding="utf-8") as fh:
        for i in range(args.frames):
            frame = _frame(i, args.pilot_lat, args.pilot_lon,
                           args.radius, args.altitude, args.speed, rng)
            fh.write(json.dumps(frame) + "\n")
            fh.flush()
            if args.live:
                time.sleep(1.0)

    print(f"[demo] wrote {args.frames} frame(s) to {out_path}")


if __name__ == "__main__":
    main()

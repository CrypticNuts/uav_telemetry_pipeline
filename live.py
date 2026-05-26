"""
live.py — Live DroneID telemetry pipeline driven directly off a USRP.

Same decode chain as the offline ``run_pipeline.py`` runner, but the
source is a USRP B-series radio rather than a ``.cfile`` on disk. Each
received chunk is fed to :meth:`NativeDecoder.decode_samples` and the
resulting frames are mapped to the dashboard's ``.jsonl`` schema and
appended to a file the dashboard can tail.

Optionally:

* ``--save-iq`` persists every received chunk to disk (same format and
  filename convention as ``DroneSecurity_with_iq_save``) so the same
  capture can be replayed later through the offline pipeline.
* ``--kalman`` pipes decoded frames through the project's constant-
  velocity Kalman tracker before they reach the dashboard, smoothing
  GPS jitter and rejecting outliers.

Usage
-----
::

    python -m uav_telemetry_pipeline.live \\
        --jsonl dashboard/telemetry_stream.jsonl \\
        --sample-rate 50e6

    python -m uav_telemetry_pipeline.live \\
        --jsonl run1.jsonl --sample-rate 50e6 \\
        --save-iq --iq-dir captures/run1 --max-chunks 10

Requirements
------------
* USRP B200 / B205 (any UHD device the host can enumerate)
* ``uhd`` Python bindings and ``uhd-host`` system package installed
* The bundled DroneSecurity source tree (``NativeDecoder`` reuses it)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from uav_telemetry_pipeline.config import PipelineConfig
from uav_telemetry_pipeline.decoders import NativeDecoder, TelemetryFrame

logger = logging.getLogger("pipeline.live")


# DJI DroneID center-frequency scan list (MHz) — same set the upstream
# DroneSecurity live receiver uses. The pipeline locks onto whichever
# frequency first yields a CRC-OK frame.
DEFAULT_FREQUENCIES_MHZ: tuple[float, ...] = (
    2414.5, 2429.502441, 2434.5, 2444.5, 2459.5, 2474.5,
    5721.5, 5731.5, 5741.5, 5756.5, 5761.5, 5771.5,
    5786.5, 5801.5, 5816.5, 5831.5,
)

RECV_BUFFER_LEN = 1000


# ---------------------------------------------------------------- IQ persistence


class IQRecorder:
    """Writes received chunks to ``.cfile`` and maintains an index."""

    def __init__(self, out_dir: Path) -> None:
        self.dir = out_dir.expanduser().resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.csv"
        self.lock = threading.Lock()
        self.count = 0
        logger.info("[iq] saving chunks to %s", self.dir)

    @staticmethod
    def _filename(center_freq_hz: float, fs_hz: float) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3]
        return (
            f"iq_{ts}_{center_freq_hz/1e6:.3f}MHz"
            f"_{fs_hz/1e6:.3f}Msps.cfile"
        )

    def write(self, samples: np.ndarray, center_freq_hz: float, fs_hz: float) -> Path:
        path = self.dir / self._filename(center_freq_hz, fs_hz)
        if samples.dtype != np.complex64:
            samples = samples.astype(np.complex64, copy=False)
        samples.tofile(path)
        n_bytes = path.stat().st_size
        with self.lock:
            new = self.count == 0 and not self.index_path.exists()
            with self.index_path.open("a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["filename", "utc_iso", "center_freq_hz",
                                "sample_rate_hz", "num_samples", "file_bytes"])
                w.writerow([
                    path.name,
                    datetime.now(timezone.utc).isoformat(),
                    f"{center_freq_hz:.0f}",
                    f"{fs_hz:.0f}",
                    int(len(samples)),
                    int(n_bytes),
                ])
            self.count += 1
        logger.info("[iq] saved %s (%.1f MB)", path.name, n_bytes / 1e6)
        return path


# ----------------------------------------------------------- jsonl conversion


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_latlon(lat: float, lon: float) -> bool:
    if lat == 0.0 and lon == 0.0:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def frame_to_dashboard_jsonl(
    frame: TelemetryFrame,
    center_freq_hz: float | None = None,
) -> dict[str, Any]:
    """Map a pipeline ``TelemetryFrame`` to the dashboard's flat schema.

    Mirrors ``dashboard/pipeline_to_jsonl.py`` so the live and replay
    paths produce byte-identical lines for the same decoded frame.
    Adds an optional ``center_freq_hz`` field (live-only metadata —
    the dashboard ignores extra keys).
    """
    lat = float(frame.get("lat", 0.0) or 0.0)
    lon = float(frame.get("lon", 0.0) or 0.0)
    has_fix = _valid_latlon(lat, lon)

    pilot_lat = float(frame.get("app_lat") or frame.get("home_lat") or 0.0)
    pilot_lon = float(frame.get("app_lon") or frame.get("home_lon") or 0.0)
    pilot_valid = _valid_latlon(pilot_lat, pilot_lon)

    vN = float(frame.get("vel_north", 0.0) or 0.0)
    vE = float(frame.get("vel_east", 0.0) or 0.0)
    vU = float(frame.get("vel_up", 0.0) or 0.0)
    speed = math.sqrt(vN * vN + vE * vE + vU * vU)

    gps_ms = frame.get("gps_time_ms")
    try:
        if gps_ms and float(gps_ms) > 0:
            ts = datetime.fromtimestamp(
                float(gps_ms) / 1000.0, tz=timezone.utc
            ).isoformat()
        else:
            ts = _iso_now()
    except (TypeError, ValueError):
        ts = _iso_now()

    out: dict[str, Any] = {
        "timestamp":     ts,
        "decoder":       frame.get("decoder", "native"),
        "drone_lat":     lat if has_fix else None,
        "drone_lon":     lon if has_fix else None,
        "drone_alt":     float(frame.get("altitude_m", 0.0) or 0.0),
        "drone_speed":   float(speed),
        "pilot_lat":     pilot_lat if pilot_valid else None,
        "pilot_lon":     pilot_lon if pilot_valid else None,
        "serial_number": str(frame.get("serial", "") or ""),
        "crc_valid":     bool(frame.get("crc_ok", False)),
    }
    if center_freq_hz is not None:
        out["center_freq_hz"] = float(center_freq_hz)
    return out


class JsonlSink:
    """Append-only newline-delimited JSON writer with line-flush semantics."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append so a restart adds to the same stream — the
        # dashboard tails from EOF, so old frames are not re-emitted.
        self._fh = self.path.open("a", buffering=1)
        self.count = 0
        logger.info("[jsonl] writing frames to %s", self.path)

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.count += 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


# --------------------------------------------------------- USRP receive helpers


def _make_usrp(device_args: str = "type=b200, recv_frame_size=8200,num_recv_frames=512"):
    """Lazily import ``uhd`` so the module imports without the radio."""
    try:
        import uhd  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "uhd python bindings not found. Install UHD with the python "
            "wheel or 'sudo apt install uhd-host libuhd-dev python3-uhd'."
        ) from exc
    import uhd
    return uhd, uhd.usrp.MultiUSRP(device_args)


def _configure_rx(uhd, usrp, sample_rate: float, gain: float | None,
                  duration_s: float) -> tuple[int, Any, Any, np.ndarray]:
    usrp.set_rx_antenna("RX2", 0)
    if gain and gain > 0:
        usrp.set_rx_gain(float(gain), 0)
    else:
        usrp.set_rx_agc(True, 0)

    usrp.set_rx_rate(sample_rate, 0)
    num_samps = int(duration_s * sample_rate)

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    metadata = uhd.types.RXMetadata()
    streamer = usrp.get_rx_stream(st_args)
    recv_buf = np.zeros((1, RECV_BUFFER_LEN), dtype=np.complex64)
    return num_samps, metadata, streamer, recv_buf


def _receive_chunk(uhd, num_samps: int, metadata, streamer, recv_buf) -> np.ndarray | None:
    samples = np.zeros(num_samps, dtype=np.complex64)

    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = int(num_samps)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)

    n_iters = num_samps // RECV_BUFFER_LEN
    for i in range(n_iters):
        streamer.recv(recv_buf, metadata, timeout=1.4)
        if "ERROR_CODE_TIMEOUT" in str(metadata.strerror()):
            return None
        samples[i * RECV_BUFFER_LEN:(i + 1) * RECV_BUFFER_LEN] = recv_buf[0]
    return samples


# --------------------------------------------------------------- session glue


class LiveSession:
    """Drives the radio, decodes chunks, and writes dashboard-format jsonl.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI args. See :func:`_parse_args` for the schema.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = PipelineConfig.load()
        self.exit_event = threading.Event()
        self.decoder = NativeDecoder(config=self.config, legacy=args.legacy)
        if not self.decoder.is_available():
            raise SystemExit(
                f"NativeDecoder unavailable — DroneSecurity src not found at "
                f"{self.config.dronesecurity_src}. Set DRONESECURITY_PATH or "
                f"edit config.py."
            )
        self.sink = JsonlSink(Path(args.jsonl))
        self.recorder = IQRecorder(Path(args.iq_dir)) if args.save_iq else None
        self.tracker = None
        if args.kalman:
            from uav_telemetry_pipeline.tracking import KalmanTracker
            self.tracker = KalmanTracker()
            logger.info("[live] Kalman tracker enabled")

        self.interesting_freq: float = 0.0
        self.chunks_done = 0
        self.frames_emitted = 0
        self.sample_offset_base = 0

    # ------------------------------------------------------------------ run

    def run(self) -> int:
        signal.signal(signal.SIGINT, lambda *_: self.exit_event.set())

        uhd, usrp = _make_usrp()
        num_samps, metadata, streamer, recv_buf = _configure_rx(
            uhd, usrp, self.args.sample_rate, self.args.gain, self.args.duration,
        )

        freqs_mhz = self.args.frequencies or list(DEFAULT_FREQUENCIES_MHZ)
        logger.info("[live] sweeping %d frequencies @ %.2f Msps, %.2fs/band",
                    len(freqs_mhz), self.args.sample_rate / 1e6, self.args.duration)

        try:
            self._scan_loop(uhd, usrp, freqs_mhz,
                            num_samps, metadata, streamer, recv_buf)
        finally:
            self.sink.close()
            logger.info("[live] session ended — %d chunks, %d frames emitted",
                        self.chunks_done, self.frames_emitted)
        return 0

    # ----------------------------------------------------------- scan loop

    def _scan_loop(self, uhd, usrp, freqs_mhz, num_samps, metadata,
                   streamer, recv_buf) -> None:
        fixed_runs = 0
        while not self.exit_event.is_set():
            for freq_mhz in freqs_mhz:
                if self.exit_event.is_set():
                    break

                c_freq = (self.interesting_freq
                          if self.interesting_freq else freq_mhz * 1e6)
                if not usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(c_freq), 0):
                    logger.warning("[live] tune to %.3f MHz failed", c_freq / 1e6)
                    continue
                logger.info("[live] tuned to %.3f MHz", c_freq / 1e6)

                samples = _receive_chunk(
                    uhd, num_samps, metadata, streamer, recv_buf,
                )
                if samples is None:
                    logger.warning("[live] recv timeout on %.3f MHz", c_freq / 1e6)
                    continue

                if self.recorder is not None:
                    try:
                        self.recorder.write(samples, c_freq, self.args.sample_rate)
                    except Exception as exc:
                        logger.warning("[live] iq save failed: %s", exc)

                # Decode in-process — no tempfile detour.
                frames = self._decode(samples)
                self.chunks_done += 1

                crc_ok = sum(1 for f in frames if f.get("crc_ok"))
                if crc_ok > 0:
                    self.interesting_freq = c_freq
                    fixed_runs = 0
                    logger.info("[live] locked onto %.3f MHz", c_freq / 1e6)
                    self._emit_frames(frames, c_freq)
                else:
                    fixed_runs += 1
                    if frames:
                        # Emit CRC-failed frames too — dashboard turns them red.
                        self._emit_frames(frames, c_freq)

                if fixed_runs > 10:
                    self.interesting_freq = 0.0
                    fixed_runs = 0

                self.sample_offset_base += len(samples)

                if (self.args.max_chunks
                        and self.chunks_done >= self.args.max_chunks):
                    logger.info("[live] reached --max-chunks=%d, stopping",
                                self.args.max_chunks)
                    self.exit_event.set()
                    break

    # -------------------------------------------------------------- decode

    def _decode(self, samples: np.ndarray) -> list[TelemetryFrame]:
        t0 = time.monotonic()
        try:
            frames = self.decoder.decode_samples(
                samples,
                self.args.sample_rate,
                sample_offset_base=self.sample_offset_base,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[live] decode failed: %s", exc)
            return []
        dt = time.monotonic() - t0
        logger.info("[live] decoded chunk in %.2fs (%d frames)", dt, len(frames))
        return frames

    # -------------------------------------------------------------- emit

    def _emit_frames(self, frames: list[TelemetryFrame], center_freq_hz: float) -> None:
        for f in frames:
            if self.tracker is not None and f.get("crc_ok"):
                self.tracker.update(f)
            record = frame_to_dashboard_jsonl(f, center_freq_hz=center_freq_hz)
            self.sink.write(record)
            self.frames_emitted += 1


# ---------------------------------------------------------------- CLI


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--jsonl", required=True, type=str,
                   help="Output .jsonl path (dashboard tails this)")
    p.add_argument("--sample-rate", "-s", type=float, default=50e6,
                   help="USRP RX rate in Hz (default 50e6)")
    p.add_argument("--duration", "-t", type=float, default=1.3,
                   help="Capture duration per band in seconds (default 1.3)")
    p.add_argument("--gain", "-g", type=float, default=0,
                   help="RX gain in dB (0 = AGC, default)")
    p.add_argument("--legacy", "-l", action="store_true",
                   help="Mavic Pro / Mavic 2 mode (different CP/ZC layout)")
    p.add_argument("--frequencies", type=float, nargs="+", default=None,
                   help="Override scan list (space-separated MHz values). "
                        "Default: standard DJI DroneID 2.4 + 5.8 GHz set.")
    p.add_argument("--save-iq", action="store_true",
                   help="Persist each received chunk as a .cfile to --iq-dir")
    p.add_argument("--iq-dir", type=str, default="iq_captures",
                   help="Directory for saved IQ chunks (default ./iq_captures)")
    p.add_argument("--kalman", action="store_true",
                   help="Run decoded frames through KalmanTracker before emission")
    p.add_argument("--max-chunks", type=int, default=0,
                   help="Stop after N chunks (0 = run until Ctrl-C)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    session = LiveSession(args)
    return session.run()


if __name__ == "__main__":
    sys.exit(main())

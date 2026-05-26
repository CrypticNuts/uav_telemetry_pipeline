"""
native.py — In-process decoder reusing DroneSecurity modules as a library.

DroneSecurity has no separate ``turbofec`` step in this fork — the full
OFDM/QPSK/scramble/CRC chain is pure Python (``Packet``, ``qpsk.Decoder``,
``DroneIDPacket``). NativeDecoder imports those modules directly and runs
the chain on bursts detected by ``SpectrumCapture``.

Behaviorally this is similar to the primary subprocess decoder, but it
runs in-process (no subprocess spawn cost, no JSON re-parsing) and is the
correct path when the primary subprocess is unusable for some reason
(e.g. distutils/venv broken).
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
from pathlib import Path

import numpy as np

from ..config import PipelineConfig
from .base import BaseDecoder, TelemetryFrame

logger = logging.getLogger(__name__)


class NativeDecoder(BaseDecoder):
    """In-process DroneID decoder driven by DroneSecurity primitives.

    Parameters
    ----------
    config : PipelineConfig | None
        Resolved pipeline configuration; used to add DroneSecurity's
        ``src/`` directory to ``sys.path``.
    legacy : bool
        Set True for Mavic Pro / Mavic 2 captures (different CP/ZC layout).
    """

    name = "native"

    def __init__(
        self,
        config: PipelineConfig | None = None,
        legacy: bool = False,
    ) -> None:
        self.config = config or PipelineConfig.load()
        self.legacy = bool(legacy)
        self._modules: tuple | None = None

    def is_available(self) -> bool:
        return self.config.dronesecurity_src.is_dir()

    def _load_modules(self) -> tuple:
        """Lazily import DroneSecurity modules with src/ on sys.path."""
        if self._modules is not None:
            return self._modules

        src_dir = str(self.config.dronesecurity_src)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        try:
            from SpectrumCapture import SpectrumCapture  # type: ignore
            from Packet import Packet  # type: ignore
            from qpsk import Decoder  # type: ignore
            from droneid_packet import DroneIDPacket  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot import DroneSecurity modules from {src_dir}: {exc}. "
                "Ensure setuptools is installed in the active interpreter "
                "(provides the distutils shim required by SpectrumCapture)."
            ) from exc

        self._modules = (SpectrumCapture, Packet, Decoder, DroneIDPacket)
        return self._modules

    def decode(self, iq_file_path: str, sample_rate: float) -> list[TelemetryFrame]:
        if not self.is_available():
            logger.warning("NativeDecoder unavailable: %s missing",
                           self.config.dronesecurity_src)
            return []

        iq_path = Path(iq_file_path).expanduser().resolve()
        if not iq_path.is_file():
            logger.error("IQ file not found: %s", iq_path)
            return []

        raw = np.memmap(iq_path, mode="r", dtype="<f").astype(np.float32).view(np.complex64)
        return self.decode_samples(np.asarray(raw), sample_rate)

    def decode_samples(
        self,
        samples: np.ndarray,
        sample_rate: float,
        sample_offset_base: int = 0,
    ) -> list[TelemetryFrame]:
        """Decode an in-memory complex64 IQ buffer.

        This is the path used by :mod:`uav_telemetry_pipeline.live` — it
        avoids round-tripping through a temp file when samples arrive
        directly from the SDR. ``sample_offset_base`` lets a live caller
        report a monotonic sample index across chunks (e.g. running
        sample count since session start) so downstream timestamps
        stay ordered.
        """
        if not self.is_available():
            logger.warning("NativeDecoder unavailable: %s missing",
                           self.config.dronesecurity_src)
            return []

        try:
            SpectrumCapture, Packet, Decoder, DroneIDPacket = self._load_modules()
        except RuntimeError as exc:
            logger.error("%s", exc)
            return []

        if samples.dtype != np.complex64:
            samples = samples.astype(np.complex64, copy=False)

        chunk = int(0.5 * sample_rate)
        n_chunks = max(1, len(samples) // chunk + (1 if len(samples) % chunk else 0))
        logger.info(
            "NativeDecoder: %d samples, %d chunk(s) of %d",
            len(samples), n_chunks, chunk,
        )

        frames: list[TelemetryFrame] = []
        sample_cursor = 0
        for i in range(n_chunks):
            start = i * chunk
            end = min(start + chunk, len(samples))
            sample_cursor = start
            slice_ = np.asarray(samples[start:end])
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    capture = SpectrumCapture(
                        slice_,
                        Fs=float(sample_rate),
                        legacy=self.legacy,
                    )
            except Exception as exc:
                logger.debug("Chunk %d: SpectrumCapture failed: %s", i, exc)
                continue

            for pkt_idx in range(len(capture.packets)):
                frame = self._decode_one(
                    capture, pkt_idx, Packet, Decoder, DroneIDPacket,
                    sample_offset=sample_offset_base + sample_cursor,
                )
                if frame is not None:
                    frames.append(frame)

        crc_ok = sum(1 for f in frames if f.get("crc_ok"))
        logger.info("NativeDecoder produced %d frame(s) (%d CRC OK)", len(frames), crc_ok)
        return frames

    def _decode_one(
        self,
        capture,
        pkt_idx: int,
        Packet,
        Decoder,
        DroneIDPacket,
        sample_offset: int,
    ) -> TelemetryFrame | None:
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                packet_data = capture.get_packet_samples(pktnum=pkt_idx)
                packet = Packet(packet_data, legacy=self.legacy)
                symbols = packet.get_symbol_data(skip_zc=True)
                decoder = Decoder(symbols)

                for phase_corr in range(4):
                    decoder.raw_data_to_symbol_bits(phase_corr)
                    duml = decoder.magic()
                    try:
                        payload = DroneIDPacket(duml)
                    except Exception:
                        continue

                    crc_ok = payload.check_crc()
                    return self._map_payload(payload.droneid, crc_ok, sample_offset)
        except Exception as exc:
            logger.debug("Native packet %d failed: %s", pkt_idx, exc)
            return None
        return None

    @staticmethod
    def _map_payload(payload: dict, crc_ok: bool, sample_offset: int) -> TelemetryFrame:
        return {
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
            "decoder": "native",
            "timestamp_sample": int(sample_offset),
        }

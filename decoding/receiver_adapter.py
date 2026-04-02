"""
receiver_adapter.py — Abstract adapter interface for external DroneID decoders.

Wraps external reference implementations (e.g., from RUB-SysSec/DroneSecurity)
so they can be called uniformly from this pipeline. The adapter handles:
  - Input format conversion (our complex64 arrays → whatever the tool expects)
  - Output normalization (tool-specific output → DecodeResult)
  - Error handling for missing or misconfigured external tools

This module does NOT reimplement the OFDM/PHY-layer decoding.
"""

from __future__ import annotations

import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from telemetry.models import DecodeResult

logger = logging.getLogger(__name__)


class ReceiverAdapter(ABC):
    """Abstract adapter for external DroneID receiver implementations.

    Subclass this to integrate a specific external decoder tool.
    Each adapter must implement availability checking and two decode
    paths: from an in-memory IQ burst array, and from a file on disk.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the external decoder is installed and reachable."""
        ...

    @abstractmethod
    def decode_burst(
        self,
        iq_burst: np.ndarray,
        sample_rate_hz: float,
        center_freq_hz: float,
        burst_index: int = 0,
    ) -> DecodeResult:
        """Decode a single IQ burst extracted from the pipeline.

        The adapter is responsible for serializing the burst to whatever
        format the backend expects (typically a temp .fc32 file), invoking
        the decoder, and capturing all output.

        Parameters
        ----------
        iq_burst : np.ndarray
            Complex64 IQ samples for one detected burst.
        sample_rate_hz : float
            Sample rate of the IQ data.
        center_freq_hz : float
            Center frequency of the IQ data.
        burst_index : int
            Index of this burst in the pipeline's detection list.

        Returns
        -------
        DecodeResult
            Structured result with subprocess output and any parsed frames.
        """
        ...

    @abstractmethod
    def decode_file(
        self,
        file_path: Path,
        sample_rate_hz: float,
        center_freq_hz: float,
    ) -> DecodeResult:
        """Decode DroneID frames from a capture file on disk.

        Parameters
        ----------
        file_path : Path
            Path to the raw IQ capture file.
        sample_rate_hz : float
            Sample rate of the capture.
        center_freq_hz : float
            Center frequency of the capture.

        Returns
        -------
        DecodeResult
            Structured result with subprocess output and any parsed frames.
        """
        ...

    # ------------------------------------------------------------------
    # Shared utilities for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def export_burst_fc32(iq_burst: np.ndarray, dest: Path) -> Path:
        """Write a complex64 burst to an interleaved float32 file.

        Parameters
        ----------
        iq_burst : np.ndarray
            Complex IQ samples.
        dest : Path
            Output file path (should end in .fc32).

        Returns
        -------
        Path
            The written file path.
        """
        interleaved = np.empty(iq_burst.size * 2, dtype=np.float32)
        interleaved[0::2] = iq_burst.real.astype(np.float32)
        interleaved[1::2] = iq_burst.imag.astype(np.float32)
        interleaved.tofile(dest)
        logger.debug(
            "Exported %d IQ samples to %s (%d bytes)",
            iq_burst.size, dest, interleaved.nbytes,
        )
        return dest

    @staticmethod
    def create_temp_dir(prefix: str = "uav_decode_") -> Path:
        """Create a temporary directory for decoder I/O.

        The caller is responsible for cleanup (or use as a context manager
        via tempfile.TemporaryDirectory).

        Returns
        -------
        Path
            Path to the created temp directory.
        """
        tmp = Path(tempfile.mkdtemp(prefix=prefix))
        logger.debug("Created temp directory: %s", tmp)
        return tmp

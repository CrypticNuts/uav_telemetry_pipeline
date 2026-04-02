"""
load_verified_iq.py — Load verified complex IQ captures (Case A).

Handles standard SDR capture formats:
  - .fc32 / .cf32 — interleaved float32 I/Q
  - .cs8 / .sc8   — interleaved signed int8 I/Q
  - .cs16 / .sc16 — interleaved signed int16 I/Q
  - .npy          — numpy complex64/complex128 arrays

These are captures known to have correct center frequency, bandwidth,
and sample rate for DJI DroneID decoding.
"""

import logging
from pathlib import Path

import numpy as np

from telemetry.models import InputClassification

logger = logging.getLogger(__name__)

# Supported raw binary formats and their numpy dtypes (interleaved pairs)
_RAW_FORMATS: dict[str, np.dtype] = {
    ".fc32": np.dtype(np.float32),
    ".cf32": np.dtype(np.float32),
    ".cs8": np.dtype(np.int8),
    ".sc8": np.dtype(np.int8),
    ".cs16": np.dtype(np.int16),
    ".sc16": np.dtype(np.int16),
}


def load_verified_iq(
    file_path: Path,
    sample_rate_hz: float,
    center_freq_hz: float,
    fmt: str | None = None,
) -> tuple[np.ndarray, InputClassification, dict[str, float]]:
    """Load a verified IQ capture file.

    Parameters
    ----------
    file_path : Path
        Path to the IQ capture file.
    sample_rate_hz : float
        Sample rate in Hz (must be known for Case A).
    center_freq_hz : float
        Center frequency in Hz (must be known for Case A).
    fmt : str | None
        Explicit format override (e.g. "fc32", "sc16"). Use this for
        files without an extension. If None, the format is inferred
        from the file extension.

    Returns
    -------
    tuple[np.ndarray, InputClassification, dict[str, float]]
        - Complex64 IQ array
        - Classification (always CASE_A)
        - Metadata dict with sample_rate_hz and center_freq_hz

    Raises
    ------
    FileNotFoundError
        If file_path does not exist.
    ValueError
        If the file format is not supported.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"IQ file not found: {file_path}")

    # Resolve format: explicit override > file extension
    if fmt is not None:
        suffix = fmt if fmt.startswith(".") else f".{fmt}"
        suffix = suffix.lower()
        logger.info("Using explicit format override: %s", suffix)
    else:
        suffix = file_path.suffix.lower()

    metadata = {
        "sample_rate_hz": sample_rate_hz,
        "center_freq_hz": center_freq_hz,
    }

    if suffix == ".npy":
        iq = np.load(file_path)
        if not np.iscomplexobj(iq):
            raise ValueError(f"Expected complex array in {file_path}, got {iq.dtype}")
        iq = iq.astype(np.complex64)

    elif suffix in _RAW_FORMATS:
        component_dtype = _RAW_FORMATS[suffix]
        raw = np.fromfile(file_path, dtype=component_dtype)
        if raw.size % 2 != 0:
            logger.warning(
                "Odd number of samples in %s — dropping last sample", file_path
            )
            raw = raw[:-1]
        iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

    else:
        supported = list(_RAW_FORMATS.keys()) + [".npy"]
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported: {supported}. "
            f"For extensionless files, pass --fmt (e.g. --fmt fc32)."
        )

    logger.info(
        "Loaded %d verified IQ samples from %s (fs=%.1f MHz, fc=%.1f MHz) — CASE_A",
        len(iq),
        file_path.name,
        sample_rate_hz / 1e6,
        center_freq_hz / 1e6,
    )

    return iq, InputClassification.CASE_A, metadata

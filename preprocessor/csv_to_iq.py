"""
csv_to_iq.py — Convert CSV-formatted RF captures to complex IQ arrays.

This handles exploratory datasets (Case B) such as DroneDetect/DroneRF
where raw data is stored as CSV with real-valued columns. The output is
a complex64 numpy array suitable for downstream detection stages.

WARNING: CSV-sourced data may lack the bandwidth, center frequency, or
sample rate needed for DroneID decoding. This module marks its output
as Case B (unverified) so downstream stages can handle it accordingly.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from telemetry.models import InputClassification

logger = logging.getLogger(__name__)


def convert_csv_to_iq(
    csv_path: Path,
    i_column: str = "I",
    q_column: str = "Q",
    sample_rate_hz: float | None = None,
) -> tuple[np.ndarray, InputClassification]:
    """Load a CSV file and return a complex IQ array with classification.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file containing RF samples.
    i_column : str
        Name of the column containing in-phase (I) samples.
    q_column : str
        Name of the column containing quadrature (Q) samples.
    sample_rate_hz : float | None
        Known sample rate in Hz, if available. None means unknown.

    Returns
    -------
    tuple[np.ndarray, InputClassification]
        Complex64 IQ array and its classification (always CASE_B for CSV sources).

    Raises
    ------
    FileNotFoundError
        If csv_path does not exist.
    KeyError
        If the expected columns are not found in the CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("Loading CSV from %s", csv_path)
    df = pd.read_csv(csv_path)

    if i_column not in df.columns or q_column not in df.columns:
        available = list(df.columns)
        raise KeyError(
            f"Expected columns '{i_column}' and '{q_column}', "
            f"found: {available}"
        )

    i_samples = df[i_column].to_numpy(dtype=np.float32)
    q_samples = df[q_column].to_numpy(dtype=np.float32)
    iq = (i_samples + 1j * q_samples).astype(np.complex64)

    logger.info(
        "Loaded %d samples from CSV (sample_rate=%s Hz) — classified as CASE_B",
        len(iq),
        sample_rate_hz or "unknown",
    )

    return iq, InputClassification.CASE_B

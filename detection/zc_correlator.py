"""
zc_correlator.py — Zadoff-Chu sequence correlator for DroneID detection.

DJI DroneID uses Zadoff-Chu (ZC) sequences as synchronization symbols
within its OFDM frame structure. This module generates the expected ZC
sequence and cross-correlates it against IQ data to locate frame starts.

References:
    - DJI DroneID uses ZC root indices documented in RUB-SysSec/DroneSecurity
    - 3GPP TS 36.211 for general ZC sequence definition
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ZadoffChuCorrelator:
    """Cross-correlate IQ data with a Zadoff-Chu reference sequence.

    Parameters
    ----------
    root_index : int
        ZC sequence root index (u). Must be coprime with sequence length.
    seq_length : int
        Length of the ZC sequence (N_zc). Must be a prime number for
        ideal correlation properties.
    detection_threshold : float
        Normalized correlation threshold (0.0–1.0) for frame detection.
    """

    def __init__(
        self,
        root_index: int = 600,
        seq_length: int = 601,
        detection_threshold: float = 0.5,
    ) -> None:
        self.root_index = root_index
        self.seq_length = seq_length
        self.detection_threshold = detection_threshold
        self._reference: np.ndarray | None = None

    def generate_reference(self) -> np.ndarray:
        """Generate the Zadoff-Chu reference sequence.

        Returns
        -------
        np.ndarray
            Complex64 ZC sequence of length seq_length.
        """
        n = np.arange(self.seq_length)
        zc = np.exp(
            -1j * np.pi * self.root_index * n * (n + 1) / self.seq_length
        ).astype(np.complex64)
        self._reference = zc
        logger.debug(
            "Generated ZC reference: root=%d, length=%d",
            self.root_index,
            self.seq_length,
        )
        return zc

    def correlate(self, iq: np.ndarray) -> np.ndarray:
        """Cross-correlate IQ data with the ZC reference.

        Parameters
        ----------
        iq : np.ndarray
            Complex64 IQ samples to search.

        Returns
        -------
        np.ndarray
            Normalized correlation magnitude (float32).

        Raises
        ------
        NotImplementedError
            Full correlation pipeline is not yet implemented.
        """
        raise NotImplementedError(
            "ZadoffChuCorrelator.correlate() is not yet implemented. "
            "Will perform frequency-domain cross-correlation using "
            "the generated ZC reference and return peak locations."
        )

    def find_frame_starts(self, iq: np.ndarray) -> list[int]:
        """Locate DroneID frame start indices in IQ data.

        Parameters
        ----------
        iq : np.ndarray
            Complex64 IQ samples.

        Returns
        -------
        list[int]
            Sample indices where DroneID frames likely begin.

        Raises
        ------
        NotImplementedError
            Not yet implemented.
        """
        raise NotImplementedError(
            "ZadoffChuCorrelator.find_frame_starts() is not yet implemented. "
            "Will use correlate() output + peak detection."
        )

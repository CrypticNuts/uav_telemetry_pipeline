"""
mmse_decoder.py — NativeDecoder variant that swaps LS for MMSE equalization.

Architecture
------------
DroneSecurity's ``Packet`` class computes the channel estimate inside
``__init__`` (via ``estimate_channel`` on the two ZC pilot symbols) and stores
it in ``self.channel``. Downstream, ``symbol_equalized`` reads ``self.channel``
to equalize each data symbol.

MMSEDecoder lets ``Packet`` build itself normally (so we get its sync, FFO
correction, and the active-carrier ``symbols_freq_domain``), then **replaces
``packet.channel`` with an MMSE estimate** before calling
``get_symbol_data``. No DroneSecurity source is modified.

Noise variance is estimated from the LS residual rather than guard bins,
because ``Packet`` discards the guard bins inside ``helpers.tfft``. The
residual estimator is documented in
:meth:`MMSEEqualizer.estimate_noise_var_residual`.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
from pathlib import Path

import numpy as np

from ..config import PipelineConfig
from .base import TelemetryFrame
from .mmse_equalizer import MMSEEqualizer, MMSEEqualizerConfig
from .native import NativeDecoder

logger = logging.getLogger(__name__)


class MMSEDecoder(NativeDecoder):
    """In-process DroneID decoder using MMSE channel equalization.

    Parameters
    ----------
    config : PipelineConfig | None
        Resolved pipeline configuration (used to locate DroneSecurity's ``src/``).
    legacy : bool
        Set True for Mavic Pro / Mavic 2 captures (different CP/ZC layout).
    equalizer : MMSEEqualizer | None
        Inject a pre-configured equalizer; default uses DroneID OcuSync 2.0
        geometry (NCARRIERS=601, NFFT=1024).
    """

    name = "native_mmse"

    def __init__(
        self,
        config: PipelineConfig | None = None,
        legacy: bool = False,
        equalizer: MMSEEqualizer | None = None,
    ) -> None:
        super().__init__(config=config, legacy=legacy)
        self.equalizer = equalizer or MMSEEqualizer(MMSEEqualizerConfig())

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

                self._replace_channel_with_mmse(packet)

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
                    frame = self._map_payload(payload.droneid, crc_ok, sample_offset)
                    frame["decoder"] = self.name
                    return frame
        except Exception as exc:
            logger.debug("MMSE packet %d failed: %s", pkt_idx, exc)
            return None
        return None

    def _replace_channel_with_mmse(self, packet) -> None:
        """Recompute ``packet.channel`` using MMSE instead of LS.

        Reads the same ZC pilots ``Packet`` already used during construction
        and pulls the ZC root sequences from the DroneSecurity source. The
        averaged MMSE channel replaces the LS one in-place.
        """
        zc_indices = list(packet.ZC_SYMBOL_IDX)
        zc_roots = self._lookup_zc_roots(packet)

        zc_seqs = [self._zc_seq_for_root(packet, r) for r in zc_roots]

        rx_pilots = [packet.symbols_freq_domain[i] for i in zc_indices]
        noise_var = float(np.mean([
            self.equalizer.estimate_noise_var_residual(rx, tx)
            for rx, tx in zip(rx_pilots, zc_seqs)
        ]))

        channel, _ = self.equalizer.equalize_packet(
            symbols_freq_domain=packet.symbols_freq_domain,
            zc_symbol_indices=zc_indices,
            zc_sequences=zc_seqs,
            noise_var=noise_var,
        )
        packet.channel = channel

    @staticmethod
    def _lookup_zc_roots(packet) -> tuple[int, int]:
        """Recover the ZC root indices DroneSecurity used for this packet.

        ``Packet`` auto-detects the roots via ``find_zc_seq`` and stores them
        as attributes ``zc_seq_1`` / ``zc_seq_2`` in some forks; fall back to
        the OcuSync 2.0 defaults (600, 147) otherwise.
        """
        r1 = getattr(packet, "zc_seq_1", None)
        r2 = getattr(packet, "zc_seq_2", None)
        if r1 is not None and r2 is not None:
            return int(r1), int(r2)
        return 600, 147

    @staticmethod
    def _zc_seq_for_root(packet, root: int) -> np.ndarray:
        """Generate the expected ZC pilot for one root, matching DS conventions."""
        src_dir = str(
            Path(__file__).resolve().parents[2] / "DroneSecurity" / "src"
        )
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from zcsequence import zcsequence_f  # type: ignore

        ncarriers = packet.NCARRIERS
        seq = zcsequence_f(root, ncarriers).astype(np.complex64)
        seq[ncarriers // 2] = 1.0 + 0.0j
        return seq

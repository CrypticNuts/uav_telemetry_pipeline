"""
mmse_equalizer.py — MMSE (Wiener-shrinkage) channel equalizer for DroneID OFDM.

DroneSecurity's `Packet.estimate_channel` uses plain Least-Squares / Zero-Forcing
equalization: H_ls = Y / X. This is unbiased but amplifies noise on subcarriers
where the true channel response is small.

This module implements a Wiener-shrinkage refinement of the LS estimate:

    H_mmse = H_ls * |H_ls|^2 / (|H_ls|^2 + sigma^2_n)

which suppresses low-confidence bins (likely dominated by noise) while
preserving high-magnitude bins (likely real channel taps).

Notes
-----
Strictly speaking the textbook MMSE channel estimator is
``H_ls * |H|^2 / (|H|^2 + sigma_n^2 / sigma_x^2)``. Here the pilot signal X is
a Zadoff-Chu sequence with |X| = 1 by construction, so sigma_x^2 = 1 and the
two forms coincide.

The class is intentionally DroneSecurity-independent so it can be unit-tested
in isolation. The integration with `Packet` lives in `mmse_decoder.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

ArrayC = np.ndarray  # complex array (np.complex64 / np.complex128)
ArrayF = np.ndarray  # real array


@dataclass(frozen=True)
class MMSEEqualizerConfig:
    """Static OFDM geometry. Defaults match DJI DroneID / OcuSync 2.0."""

    ncarriers: int = 601
    nfft: int = 1024
    # Floor on the shrinkage denominator. Prevents division-by-zero on the
    # (forced-to-1) DC carrier and on edge cases where |H_ls|^2 underflows.
    eps: float = 1e-12


class MMSEEqualizer:
    """MMSE / Wiener-shrinkage channel equalizer for DroneID pilots.

    Drop-in replacement for the LS pair
    ``Packet.estimate_channel`` + ``Packet.symbol_equalized``.

    Examples
    --------
    >>> import numpy as np
    >>> eq = MMSEEqualizer()
    >>> rx  = np.ones(601, dtype=np.complex64)
    >>> tx  = np.ones(601, dtype=np.complex64)
    >>> h   = eq.equalize(rx, tx, noise_var=0.0)
    >>> np.allclose(h, 1.0)
    True
    """

    def __init__(self, config: MMSEEqualizerConfig | None = None) -> None:
        self.cfg = config or MMSEEqualizerConfig()
        self._active_idx, self._guard_idx = self._compute_carrier_indices(
            self.cfg.nfft, self.cfg.ncarriers
        )

    # ----------------------------------------------------------- introspection

    @property
    def active_indices(self) -> np.ndarray:
        """FFT bin indices that carry signal (length = ``ncarriers``)."""
        return self._active_idx

    @property
    def guard_indices(self) -> np.ndarray:
        """FFT bin indices outside the active band — used for noise estimation."""
        return self._guard_idx

    # --------------------------------------------------------- noise estimate

    def estimate_noise_var(self, full_fft_symbol: ArrayC) -> float:
        """Estimate noise variance from the guard bins of a full NFFT FFT.

        Parameters
        ----------
        full_fft_symbol : complex array, shape (nfft,)
            One OFDM symbol after the raw ``np.fft.fft`` (no active-carrier
            extraction, no fftshift). Guard bins must still be present.

        Returns
        -------
        sigma2 : float
            Mean magnitude-squared of the guard bins. This is an unbiased
            estimator of the per-subcarrier noise variance under the standard
            AWGN-on-each-bin assumption.
        """
        full = np.asarray(full_fft_symbol)
        if full.shape != (self.cfg.nfft,):
            raise ValueError(
                f"full_fft_symbol must have shape ({self.cfg.nfft},), "
                f"got {full.shape}"
            )
        guard = full[self._guard_idx]
        return float(np.mean(np.abs(guard) ** 2))

    def estimate_noise_var_residual(
        self, received: ArrayC, expected: ArrayC
    ) -> float:
        """Fallback noise estimator when only active carriers are available.

        Uses the residual after a smoothed-LS fit. Less accurate than the
        guard-bin method but works when the caller (e.g. DroneSecurity's
        ``Packet``) has already discarded the guards.
        """
        ls = self._safe_divide(received, expected)
        # Smooth the LS estimate with a short moving average — the residual
        # against the smoothed channel is dominated by noise.
        k = 7
        kernel = np.ones(k, dtype=np.float64) / k
        smooth = np.convolve(np.real(ls), kernel, mode="same") + \
                 1j * np.convolve(np.imag(ls), kernel, mode="same")
        residual = ls - smooth
        return float(np.mean(np.abs(residual) ** 2))

    # --------------------------------------------------------- core equalizer

    def equalize(
        self,
        received_signal: ArrayC,
        expected_signal: ArrayC,
        noise_var: float,
    ) -> ArrayC:
        """Compute the MMSE channel estimate for one pilot symbol.

        Parameters
        ----------
        received_signal : complex array, shape (ncarriers,)
            Active-carrier FFT bins of the received pilot symbol.
        expected_signal : complex array, shape (ncarriers,)
            Reference (transmitted) pilot — typically a Zadoff-Chu sequence.
        noise_var : float
            Per-subcarrier noise variance (e.g. from `estimate_noise_var`).

        Returns
        -------
        H_mmse : complex array, shape (ncarriers,)
            MMSE / Wiener-shrinkage channel estimate.
        """
        rx = np.asarray(received_signal, dtype=np.complex64)
        tx = np.asarray(expected_signal, dtype=np.complex64).copy()
        if rx.shape != (self.cfg.ncarriers,) or tx.shape != (self.cfg.ncarriers,):
            raise ValueError(
                f"received/expected must have shape ({self.cfg.ncarriers},), "
                f"got rx={rx.shape}, tx={tx.shape}"
            )
        # Match DroneSecurity's DC-bin guard: force the centre bin to 1 to
        # avoid the ill-defined ZC value at DC.
        tx[self.cfg.ncarriers // 2] = 1.0 + 0.0j

        h_ls = self._safe_divide(rx, tx)
        mag2 = (h_ls.real * h_ls.real + h_ls.imag * h_ls.imag).astype(np.float64)
        shrink = mag2 / (mag2 + float(noise_var) + self.cfg.eps)
        return (h_ls * shrink.astype(np.complex64)).astype(np.complex64)

    def apply(self, symbol_f: ArrayC, channel: ArrayC) -> ArrayC:
        """Equalize one data symbol given a channel estimate.

        Mirror of DroneSecurity's ``Packet.symbol_equalized``.
        """
        return self._safe_divide(symbol_f, channel)

    # --------------------------------------------- multiplicative MMSE form

    def equalization_weights(
        self,
        received_signal: ArrayC,
        expected_signal: ArrayC,
        noise_var: float,
    ) -> ArrayC:
        """Compute the **multiplicative** MMSE equalizer weights.

        The Wiener-shrunk channel estimate from :meth:`equalize` denoises
        the LS estimate, but when used as a divisor in symbol equalization
        (``Y / H_mmse``) the shrinkage is *undone* — small-magnitude bins
        become large when inverted. The proper MMSE equalizer is a
        per-subcarrier complex weight applied multiplicatively:

            W = conj(H_ls) / (|H_ls|^2 + sigma^2)
            X_est = W * Y

        On null subcarriers (small |H_ls|), ``W`` is small and the
        equalized output is suppressed instead of amplified — that's the
        actual noise-rejection mechanism MMSE provides over LS.

        Returns
        -------
        W : complex array, shape (ncarriers,)
            The MMSE equalization weights. At ``noise_var=0`` this
            reduces to ``1 / H_ls`` (LS inverse).
        """
        rx = np.asarray(received_signal, dtype=np.complex64)
        tx = np.asarray(expected_signal, dtype=np.complex64).copy()
        if rx.shape != (self.cfg.ncarriers,) or tx.shape != (self.cfg.ncarriers,):
            raise ValueError(
                f"received/expected must have shape ({self.cfg.ncarriers},), "
                f"got rx={rx.shape}, tx={tx.shape}"
            )
        tx[self.cfg.ncarriers // 2] = 1.0 + 0.0j

        h_ls = self._safe_divide(rx, tx)
        mag2 = (h_ls.real * h_ls.real + h_ls.imag * h_ls.imag).astype(np.float64)
        denom = (mag2 + float(noise_var) + self.cfg.eps).astype(np.complex64)
        return (np.conj(h_ls) / denom).astype(np.complex64)

    @staticmethod
    def apply_weights(symbol_f: ArrayC, weights: ArrayC) -> ArrayC:
        """Apply multiplicative MMSE weights to one data symbol."""
        return (np.asarray(symbol_f) * np.asarray(weights)).astype(np.complex64)

    # --------------------------------------------------------- packet helper

    def equalize_packet(
        self,
        symbols_freq_domain: Sequence[ArrayC],
        zc_symbol_indices: Sequence[int],
        zc_sequences: Sequence[ArrayC],
        noise_var: float,
    ) -> tuple[ArrayC, list[ArrayC]]:
        """Estimate the channel from pilot symbols and equalize the burst.

        Parameters
        ----------
        symbols_freq_domain : sequence of complex arrays
            One ``(ncarriers,)`` array per OFDM symbol in the packet (active
            carriers only, post-``tfft``).
        zc_symbol_indices : sequence of int
            Symbol positions of the ZC pilots, e.g. ``[3, 5]`` for OcuSync 2.0.
        zc_sequences : sequence of complex arrays
            Expected (transmitted) ZC pilots for each pilot position, same
            order as ``zc_symbol_indices``.
        noise_var : float
            Pre-estimated noise variance.

        Returns
        -------
        channel : complex array, shape (ncarriers,)
            Averaged MMSE channel estimate across all pilots.
        equalized : list of complex arrays
            Each data (non-pilot) symbol after equalization.
        """
        if len(zc_symbol_indices) != len(zc_sequences):
            raise ValueError(
                "zc_symbol_indices and zc_sequences must have the same length"
            )

        per_pilot = []
        for idx, zc in zip(zc_symbol_indices, zc_sequences):
            h = self.equalize(symbols_freq_domain[idx], zc, noise_var)
            per_pilot.append(h)
        channel = np.mean(np.stack(per_pilot, axis=0), axis=0).astype(np.complex64)

        pilots = set(int(i) for i in zc_symbol_indices)
        equalized = [
            self.apply(symbols_freq_domain[i], channel)
            for i in range(len(symbols_freq_domain))
            if i not in pilots
        ]
        return channel, equalized

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _compute_carrier_indices(nfft: int, ncarriers: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (active_bins, guard_bins) in FFT-order (DC at index 0).

        LTE-style mapping: active band = DC + 300 below + 300 above,
        wrapping at the high-index end of the FFT.
        """
        half = ncarriers // 2  # 300 for 601 carriers
        active = np.concatenate([
            np.arange(0, half + 1),                  # DC + 300 above
            np.arange(nfft - half, nfft),            # 300 below (wrapped)
        ])
        all_bins = np.arange(nfft)
        guard = np.setdiff1d(all_bins, active, assume_unique=True)
        return active, guard

    @staticmethod
    def _safe_divide(num: ArrayC, den: ArrayC) -> ArrayC:
        """Element-wise division with zero-denominator handled."""
        den_safe = np.where(np.abs(den) < 1e-30, 1e-30 + 0j, den)
        return (np.asarray(num) / den_safe).astype(np.complex64)

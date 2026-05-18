"""
zc_correlator.py — Zadoff-Chu sequence correlator for DroneID detection.

DJI DroneID uses Zadoff-Chu (ZC) sequences as synchronization symbols within
its OFDM frame structure (LTE-derived). The fine-sync ZC root is **147**;
the coarse-sync root is **600**.

Both are placed at OFDM symbol indices [3, 5] of the 9-symbol burst (per the
DroneSecurity reference implementation — these are 0-indexed, normal-CP).

The correlator generates the time-domain ZC waveform at the input sample
rate, runs an FFT-based normalized cross-correlation against the IQ stream,
and returns peak indices subject to a minimum spacing of one burst length.

References
----------
- 3GPP TS 36.211 (ZC definition)
- RUB-SysSec/DroneSecurity src/zcsequence.py, src/helpers.py
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, firwin, oaconvolve, resample_poly, welch

logger = logging.getLogger(__name__)

# Authoritative constants (match DroneSecurity helpers.py)
NCARRIERS = 601           # active subcarriers (including DC)
NFFT = 1024               # base OFDM FFT size
BASE_RATE_HZ = 15.36e6    # LTE base rate for DroneID
CP_LENGTHS = [80, 72, 72, 72, 72, 72, 72, 72, 80]  # samples at base rate
BURST_SAMPLES_BASE = sum(CP_LENGTHS) + len(CP_LENGTHS) * NFFT  # ~9952 @ 15.36 Msps
ROOT_FINE = 147
ROOT_COARSE = 600


def _zc_freq_domain(root: int, seq_length: int = NCARRIERS) -> np.ndarray:
    """ZC sequence as used by DroneSecurity (frequency-domain values).

    Identical math to DroneSecurity.zcsequence.zcsequence_t but returned as
    complex64. The values are the modulation symbols placed on the 601 active
    subcarriers; the actual time-domain waveform is obtained by mapping these
    onto NFFT bins and IFFT-ing.
    """
    n = np.arange(seq_length)
    zc = np.exp(-1j * np.pi * root * n * (n + 1) / seq_length)
    return zc.astype(np.complex64)


def zc_time_waveform(
    root: int,
    sample_rate_hz: float,
    seq_length: int = NCARRIERS,
    nfft: int = NFFT,
) -> np.ndarray:
    """Generate the ZC OFDM symbol waveform (no CP) at ``sample_rate_hz``.

    Workflow: build the active-subcarrier vector → map onto an NFFT-bin
    spectrum with DC at index 0 (negative freqs at the top, positive at the
    bottom, matching helpers.itfft) → IFFT → resample to the target rate by
    complex linear interpolation (matches helpers.resample).
    """
    zc_f = _zc_freq_domain(root, seq_length)

    half = seq_length // 2
    spectrum = np.zeros(nfft, dtype=np.complex64)
    spectrum[-half:] = zc_f[:half]
    spectrum[: half + 1] = zc_f[half:]

    zc_t = np.fft.ifft(spectrum).astype(np.complex64)

    if abs(sample_rate_hz - BASE_RATE_HZ) < 1.0:
        return zc_t

    ratio = sample_rate_hz / BASE_RATE_HZ
    new_len = int(round(nfft * ratio))
    x_old = np.arange(nfft, dtype=np.float64)
    x_new = np.arange(new_len, dtype=np.float64) / ratio
    real = np.interp(x_new, x_old, zc_t.real)
    imag = np.interp(x_new, x_old, zc_t.imag)
    return (real + 1j * imag).astype(np.complex64)


def estimate_cfo_welch(
    iq: np.ndarray,
    sample_rate_hz: float,
    nfft_welch: int = 2048,
    bw_lo: float = 8e6,
    bw_hi: float = 11e6,
) -> tuple[float, bool]:
    """Estimate the DroneID carrier offset via Welch PSD band-search.

    Mirrors ``DroneSecurity.helpers.estimate_offset``: find a band wider
    than ``bw_lo`` MHz and narrower than ``bw_hi`` MHz above mean PSD;
    its center is the CFO. Returns ``(offset_hz, found)``.
    """
    if len(iq) < nfft_welch:
        return 0.0, False
    max_samples = nfft_welch * 256
    sample = iq if len(iq) <= max_samples else iq[:max_samples]
    f, pxx = welch(
        sample, fs=sample_rate_hz, nfft=nfft_welch,
        nperseg=nfft_welch, return_onesided=False,
    )
    pxx = np.fft.fftshift(pxx)
    f = np.fft.fftshift(f)
    pxx[nfft_welch // 2 - 10 : nfft_welch // 2 + 10] = 1.1 * pxx.mean()
    mask = pxx > pxx.mean()
    if not mask.any():
        return 0.0, False
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    for s, e in zip(starts, ends):
        bw = (f[e - 1] - f[s])
        if bw_lo < bw < bw_hi:
            return float((f[s] + f[e - 1]) / 2.0), True
    return 0.0, False


def analyze_spectrum_bands(
    iq: np.ndarray,
    sample_rate_hz: float,
    nfft_welch: int = 2048,
    min_bw_hz: float = 1e6,
) -> list[dict]:
    """Return every detected emission band above mean PSD (no width filter).

    Descriptive helper for content surveys: where the DroneID-specific
    estimator above only accepts 8–11 MHz-wide bands, this version
    reports anything wider than ``min_bw_hz``. Useful for documenting
    OcuSync video (~20 MHz), C2 (~1.4 MHz), beacons, etc.

    For 8–11 MHz wide candidates that match DroneID *by bandwidth*, a
    secondary burst-rate discriminator runs: real DroneID broadcasts at
    ~1.7 Hz, but Inspire 2 / OcuSync sub-frames in the same band can
    burst at 10–150 Hz with the same per-burst duration. The label
    distinguishes ``droneid`` (low burst rate) from ``lte_burst`` (high
    burst rate) so downstream tooling doesn't waste time on the latter.
    """
    if len(iq) < nfft_welch:
        return []
    # Welch converges quickly; using more than ~250 segments wastes memory.
    # scipy's spectrogram allocator scales O(nperseg * n_segments) and can
    # blow past 10 GB on multi-million-sample inputs.
    max_samples = nfft_welch * 256
    sample = iq if len(iq) <= max_samples else iq[:max_samples]
    f, pxx = welch(
        sample, fs=sample_rate_hz, nfft=nfft_welch,
        nperseg=nfft_welch, return_onesided=False,
    )
    pxx = np.fft.fftshift(pxx)
    f = np.fft.fftshift(f)
    # Suppress DC blip so it doesn't dominate the mean.
    pxx[nfft_welch // 2 - 10 : nfft_welch // 2 + 10] = pxx.mean()
    threshold = pxx.mean()
    mask = pxx > threshold
    if not mask.any():
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    bands: list[dict] = []
    for s, e in zip(starts, ends):
        bw = float(f[e - 1] - f[s])
        if bw < min_bw_hz:
            continue
        center_hz = float((f[s] + f[e - 1]) / 2.0)
        entry = {
            "f_center_hz": center_hz,
            "f_start_hz": float(f[s]),
            "f_end_hz": float(f[e - 1]),
            "bandwidth_hz": bw,
            "peak_psd_db_over_mean": float(
                10.0 * np.log10(pxx[s:e].max() / threshold)
            ),
        }
        # Burst-rate refinement only for the DroneID-shaped width band —
        # other widths get a pure-bandwidth label as before. The
        # discriminator is asymmetric on purpose: an 8–11 MHz band
        # *defaults* to ``droneid``, and is only demoted to ``lte_burst``
        # when we actively measure a high burst rate (> 5 Hz, i.e.
        # OcuSync sub-frames at 10–150 Hz). A zero-burst result is
        # ambiguous (detector may have missed bursts at high duty cycle,
        # or in dense-packet curated extracts where the median envelope
        # sits near the burst level) — leave the label as ``droneid``.
        if 8e6 <= bw <= 11e6:
            burst_count, burst_rate_hz = _burst_rate_in_band(
                iq, sample_rate_hz, center_hz, bw,
            )
            entry["burst_count"] = burst_count
            entry["burst_rate_hz"] = round(burst_rate_hz, 2)
            if burst_rate_hz > 5.0:
                entry["label"] = "lte_burst"
            else:
                entry["label"] = "droneid"
        else:
            entry["label"] = _classify_band(bw)
        bands.append(entry)
    bands.sort(key=lambda b: -b["peak_psd_db_over_mean"])
    return bands


def _classify_band(bw_hz: float) -> str:
    """Bandwidth-only label (used for non-DroneID-width bands)."""
    if 0.8e6 <= bw_hz <= 1.95e6:
        return "c2_control"     # DJI command/control channel
    if 17e6 <= bw_hz <= 25e6:
        return "ocusync_video"  # OcuSync 2.0 video (~20–22 MHz wide)
    return "unknown"


def _burst_rate_in_band(
    iq: np.ndarray,
    sample_rate_hz: float,
    f_center_hz: float,
    bandwidth_hz: float,
    max_window_s: float = 0.5,
    burst_duration_s: float = 650e-6,
) -> tuple[int, float]:
    """Count time-domain bursts in a specific frequency band of ``iq``.

    Workflow:
    1. Take at most ``max_window_s`` of samples (caps cost on big chunks).
    2. Shift the band to DC, lowpass to ``bandwidth_hz / 2``.
    3. Compute power envelope ``|y|^2`` smoothed over one burst length.
    4. Find peaks above a robust threshold (3× median, with a 95th-pct
       floor to handle continuous-emission cases).
    5. Return ``(burst_count, burst_rate_hz)``.

    Continuous-emission inputs typically return a burst count of 0 or
    1 (the envelope is too smooth to peak). Bursty inputs return as
    many peaks as bursts in the window.
    """
    n_window = min(len(iq), int(sample_rate_hz * max_window_s))
    if n_window < int(sample_rate_hz * 50e-3):
        # Need at least ~50 ms to see anything meaningful.
        return 0, 0.0
    window = iq[:n_window]

    # Shift the candidate band to DC.
    phase_step = np.float32(-2.0 * np.pi * f_center_hz / sample_rate_hz)
    phase = np.arange(n_window, dtype=np.float32) * phase_step
    rotator = np.empty(n_window, dtype=np.complex64)
    np.cos(phase, out=rotator.view(np.float32)[0::2])
    np.sin(phase, out=rotator.view(np.float32)[1::2])
    shifted = window * rotator
    del rotator, phase

    # Lowpass to half the band's measured bandwidth. firwin's cutoff
    # is the -6 dB point with default settings — set a bit above bw/2
    # so the passband is flat over the band.
    cutoff = min(bandwidth_hz * 0.55, sample_rate_hz / 2.05)
    fir_b = firwin(129, cutoff, fs=sample_rate_hz, window="hamming").astype(np.float32)
    filtered = oaconvolve(
        shifted, fir_b.astype(np.complex64), mode="same",
    )
    del shifted

    # Power envelope, smoothed over one burst length so peaks are flat-topped.
    power = (np.abs(filtered).astype(np.float32) ** 2)
    del filtered
    burst_len = max(1, int(sample_rate_hz * burst_duration_s))
    envelope = uniform_filter1d(power, size=burst_len, mode="constant")
    del power

    # Robust peak threshold. Use the higher of (3×median) and
    # (0.4×max) so continuous bands (median ≈ max) don't generate peaks.
    med = float(np.median(envelope))
    mx = float(envelope.max())
    threshold = max(3.0 * med, 0.4 * mx)
    if threshold <= 0 or not np.isfinite(threshold):
        return 0, 0.0

    peaks, _ = find_peaks(
        envelope,
        height=threshold,
        distance=burst_len * 2,
    )
    burst_count = int(len(peaks))
    duration_s = n_window / sample_rate_hz
    return burst_count, burst_count / duration_s


class ZadoffChuCorrelator:
    """FFT-based normalized cross-correlator against the DroneID ZC waveform.

    The correlator can run on either:
    - Raw IQ at the capture sample rate (with ``auto_cfo=True``, it
      estimates the carrier offset via Welch PSD and shifts to baseband
      before correlation — required when the signal is not centered).
    - Pre-corrected baseband IQ (set ``auto_cfo=False``).

    Parameters
    ----------
    sample_rate_hz : float
        Sample rate of the IQ stream.
    root_index : int
        ZC root index. Defaults to 147 (the deterministic fine-sync root —
        every DroneID burst contains it).
    detection_threshold : float
        Normalized correlation magnitude threshold (0..1). 0.15 picks up
        weak bursts; below ~0.08 false positives explode.
    min_spacing_samples : int | None
        Minimum gap between detected peaks. Defaults to one burst length
        (~9952 samples at 15.36 Msps, scaled to the input sample rate).
    auto_cfo : bool
        Whether to estimate and correct carrier offset before correlation.
    """

    def __init__(
        self,
        sample_rate_hz: float = BASE_RATE_HZ,
        root_index: int = ROOT_FINE,
        detection_threshold: float = 0.15,
        min_spacing_samples: int | None = None,
        auto_cfo: bool = True,
        decimate: bool = True,
    ) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        self.root_index = int(root_index)
        self.detection_threshold = float(detection_threshold)
        self.auto_cfo = bool(auto_cfo)
        self.last_cfo_hz: float = 0.0

        # Internal decimation: matched-filter cost scales with the *decimated*
        # sample rate, not the input rate. For 50 Msps -> q=3, 30.72 Msps ->
        # q=2, etc. The DroneID signal is ~9 MHz wide; an FIR-anti-aliased
        # downsample to >=15 Msps preserves it without aliasing.
        if decimate:
            self.decim_q = max(1, int(round(self.sample_rate_hz / BASE_RATE_HZ)))
        else:
            self.decim_q = 1
        self.decim_fs = self.sample_rate_hz / self.decim_q

        scale = self.decim_fs / BASE_RATE_HZ
        burst_at_decim = max(1, int(round(BURST_SAMPLES_BASE * scale * 0.9)))
        if min_spacing_samples is not None:
            # Caller-supplied spacing is in *input-rate* samples; convert.
            self.min_spacing_samples = max(1, int(min_spacing_samples) // self.decim_q)
        else:
            self.min_spacing_samples = burst_at_decim

        self._reference: np.ndarray | None = None

    def generate_reference(self) -> np.ndarray:
        """Generate (and cache) the time-domain ZC reference waveform.

        Generated at the *decimated* sample rate (``self.decim_fs``) so the
        matched filter operates on a smaller signal. For 50 Msps input with
        q=3 the reference is ~1111 samples instead of ~3333.
        """
        if self._reference is None:
            self._reference = zc_time_waveform(self.root_index, self.decim_fs)
            logger.debug(
                "ZC reference: root=%d, length=%d samples, decim_fs=%.3f MHz "
                "(input_fs=%.3f MHz, q=%d)",
                self.root_index,
                len(self._reference),
                self.decim_fs / 1e6,
                self.sample_rate_hz / 1e6,
                self.decim_q,
            )
        return self._reference

    def _maybe_shift_to_baseband(self, iq: np.ndarray) -> np.ndarray:
        """Estimate and remove the DroneID carrier offset if auto_cfo is on.

        Memory note: CFO is estimated on a downsampled window (Welch only
        needs a few seconds of signal), and the per-sample complex
        exponential is built directly in complex64 — never float64 — so
        memory stays bounded for multi-second IQ chunks.
        """
        if not self.auto_cfo:
            return iq
        # 0.1s is more than enough for a stable Welch PSD; full-chunk Welch
        # on a 25M-sample complex64 was the dominant cost previously.
        cfo_window = iq[: min(len(iq), int(self.sample_rate_hz * 0.1))]
        offset, ok = estimate_cfo_welch(cfo_window, self.sample_rate_hz)
        self.last_cfo_hz = offset if ok else 0.0
        if not ok or abs(offset) < 1e3:
            return iq

        # Build the rotator in float32 → complex64 directly (no float64).
        n = len(iq)
        phase_step = np.float32(-2.0 * np.pi * offset / self.sample_rate_hz)
        phase = (np.arange(n, dtype=np.float32) * phase_step)
        rotator = np.empty(n, dtype=np.complex64)
        np.cos(phase, out=rotator.view(np.float32)[0::2])
        np.sin(phase, out=rotator.view(np.float32)[1::2])
        del phase

        # In-place multiplication to avoid allocating another N-sample array.
        out = iq if iq.flags.writeable else iq.copy()
        np.multiply(out, rotator, out=out)
        del rotator
        logger.info("ZC correlator: detected CFO %.3f MHz, shifted to baseband",
                    offset / 1e6)
        return out

    def correlate(self, iq: np.ndarray) -> np.ndarray:
        """Normalized cross-correlation magnitude at the *decimated* rate.

        Workflow:
        1. (Optional) shift to baseband via Welch CFO estimate.
        2. Anti-alias FIR-decimate by ``self.decim_q``.
        3. Overlap-add matched filter against the ZC reference.
        4. Normalize by sliding envelope.

        The returned array length equals ``len(iq) // self.decim_q``. Peak
        indices must be multiplied by ``self.decim_q`` to map back to the
        input-rate sample index — :meth:`find_frame_starts` does this.
        """
        iq = np.ascontiguousarray(iq, dtype=np.complex64)
        iq = self._maybe_shift_to_baseband(iq)

        if self.decim_q > 1:
            # Polyphase decimation: O(N) and ~50x faster than filtfilt-based
            # ``scipy.signal.decimate`` on multi-million-sample arrays.
            taps = firwin(8 * self.decim_q + 1, 1.0 / self.decim_q,
                          window="hamming").astype(np.float32)
            iq = resample_poly(iq, up=1, down=self.decim_q, window=taps)
            iq = np.ascontiguousarray(iq, dtype=np.complex64)

        ref = self.generate_reference()
        n_ref = len(ref)

        matched = np.conj(ref[::-1]).astype(np.complex64)
        corr = oaconvolve(iq, matched, mode="same")

        power = (np.abs(iq).astype(np.float32) ** 2)
        env = uniform_filter1d(power, size=n_ref, mode="constant") * n_ref
        del power
        ref_energy = float(np.sum(np.abs(ref) ** 2))
        denom = np.sqrt(np.maximum(env * ref_energy, 1e-12))

        normalized = (np.abs(corr) / denom).astype(np.float32)
        np.clip(normalized, 0.0, 1.5, out=normalized)
        return normalized

    def find_frame_starts(
        self,
        iq: np.ndarray,
        threshold: float | None = None,
    ) -> list[int]:
        """Return *input-rate* sample indices where DroneID bursts likely begin.

        The reported indices point to the matched-filter peak position (i.e.
        roughly the centre of the ZC symbol within the burst), at the
        original input sample rate. They are NOT the start of the full
        9-symbol burst — downstream decoders that expect the burst start
        should rewind by ~3 OFDM symbols at the input rate.
        """
        thr = self.detection_threshold if threshold is None else float(threshold)
        corr = self.correlate(iq)
        peaks, _ = find_peaks(
            corr,
            height=thr,
            distance=self.min_spacing_samples,
        )
        starts = [int(p) * self.decim_q for p in peaks]
        logger.info(
            "ZC correlator: %d peak(s) >= %.3f (decim_q=%d, min spacing %d "
            "decimated samples = %d input samples)",
            len(starts),
            thr,
            self.decim_q,
            self.min_spacing_samples,
            self.min_spacing_samples * self.decim_q,
        )
        return starts

    def peak_score(self, iq: np.ndarray) -> tuple[float, int]:
        """Return (best peak value, input-rate sample index)."""
        corr = self.correlate(iq)
        idx = int(np.argmax(corr))
        return float(corr[idx]), idx * self.decim_q

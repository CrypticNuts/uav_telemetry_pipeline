"""
burst_detector.py — STFT-based energy burst detection in IQ streams.

Identifies candidate signal bursts by computing a spectrogram, collapsing
it to a per-frame power profile, and thresholding against a dynamically
estimated noise floor. This is the first-pass filter before Zadoff-Chu
correlation narrows down DroneID frame locations.

Algorithm:
    1. STFT → spectrogram (power spectral density per time frame)
    2. Collapse frequency axis → per-frame total power (dB)
    3. Noise floor = median of power profile (robust to sparse bursts)
    4. Threshold = noise_floor + threshold_db
    5. Label contiguous above-threshold regions
    6. Merge nearby regions (handles short dropouts mid-burst)
    7. Drop regions shorter than min_duration_s
    8. Map STFT frame indices back to sample indices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import stft

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BurstSegment:
    """A detected energy burst in the IQ stream."""

    start_sample: int
    end_sample: int
    duration_s: float
    peak_power_db: float
    mean_power_db: float
    snr_db: float

    @property
    def num_samples(self) -> int:
        return self.end_sample - self.start_sample


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class BurstDetector:
    """STFT-based energy-threshold burst detector.

    Parameters
    ----------
    threshold_db : float
        Power above the noise floor (dB) required to trigger detection.
    min_duration_s : float
        Minimum burst duration in seconds. Shorter detections are discarded.
    merge_gap_s : float
        Maximum gap in seconds between two detections before they are
        merged into a single burst. Prevents a brief dropout from
        splitting one transmission into two.
    nperseg : int
        STFT window length in samples. Controls frequency resolution.
        Shorter windows give better time resolution but coarser frequency
        resolution. 1024 is a good default for wideband captures.
    overlap_frac : float
        Fraction of nperseg used as overlap (0.0–1.0). Higher overlap
        gives smoother power profiles at the cost of more FFT frames.
    """

    def __init__(
        self,
        threshold_db: float = 10.0,
        min_duration_s: float = 100e-6,
        merge_gap_s: float = 500e-6,
        nperseg: int = 1024,
        overlap_frac: float = 0.5,
    ) -> None:
        if not 0.0 < overlap_frac < 1.0:
            raise ValueError(f"overlap_frac must be in (0, 1), got {overlap_frac}")
        self.threshold_db = threshold_db
        self.min_duration_s = min_duration_s
        self.merge_gap_s = merge_gap_s
        self.nperseg = nperseg
        self.overlap_frac = overlap_frac

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        iq: np.ndarray,
        sample_rate_hz: float,
        debug_plot: bool = False,
        plot_path: Path | None = None,
    ) -> list[BurstSegment]:
        """Detect energy bursts in a complex IQ array.

        Parameters
        ----------
        iq : np.ndarray
            Complex IQ samples (complex64 or complex128).
        sample_rate_hz : float
            Sample rate in Hz — required to convert frame indices to time.
        debug_plot : bool
            If True, generate a matplotlib diagnostic plot.
        plot_path : Path | None
            If provided (and debug_plot is True), save the plot to this
            path instead of displaying it.

        Returns
        -------
        list[BurstSegment]
            Detected bursts sorted by start sample.
        """
        if iq.size == 0:
            logger.warning("Empty IQ array — nothing to detect")
            return []

        if not np.iscomplexobj(iq):
            raise TypeError(
                f"Expected complex IQ array, got dtype={iq.dtype}"
            )

        noverlap = int(self.nperseg * self.overlap_frac)
        hop = self.nperseg - noverlap

        # Clamp nperseg to signal length
        effective_nperseg = min(self.nperseg, iq.size)
        effective_noverlap = min(noverlap, effective_nperseg - 1)

        logger.debug(
            "STFT params: nperseg=%d, noverlap=%d, hop=%d, N=%d",
            effective_nperseg, effective_noverlap,
            effective_nperseg - effective_noverlap, iq.size,
        )

        # --- Step 1: STFT ---
        freqs, times, Zxx = stft(
            iq,
            fs=sample_rate_hz,
            nperseg=effective_nperseg,
            noverlap=effective_noverlap,
            return_onesided=False,  # complex input → two-sided
        )

        # --- Step 2: per-frame power in dB ---
        power_linear = np.mean(np.abs(Zxx) ** 2, axis=0)  # collapse freq axis
        # Guard against log10(0)
        power_linear = np.maximum(power_linear, np.finfo(np.float64).tiny)
        power_db = 10.0 * np.log10(power_linear)

        # --- Step 3: noise floor ---
        noise_floor_db = self._estimate_noise_floor_stft(power_db)
        threshold = noise_floor_db + self.threshold_db
        logger.info(
            "Noise floor: %.1f dB | Threshold: %.1f dB (+%.1f dB)",
            noise_floor_db, threshold, self.threshold_db,
        )

        # --- Step 4: threshold mask ---
        above = power_db >= threshold

        # --- Step 5: label contiguous regions ---
        raw_regions = self._label_regions(above)

        # --- Step 6: merge nearby regions ---
        effective_hop = effective_nperseg - effective_noverlap
        merge_gap_frames = max(1, int(
            self.merge_gap_s * sample_rate_hz / effective_hop
        ))
        merged = self._merge_regions(raw_regions, merge_gap_frames)

        # --- Step 7: filter by minimum duration & build output ---
        min_frames = max(1, int(
            self.min_duration_s * sample_rate_hz / effective_hop
        ))
        bursts: list[BurstSegment] = []

        for start_frame, end_frame in merged:
            length_frames = end_frame - start_frame
            if length_frames < min_frames:
                continue

            start_sample = start_frame * effective_hop
            end_sample = min(end_frame * effective_hop + effective_nperseg, iq.size)

            region_power = power_db[start_frame:end_frame]
            peak = float(np.max(region_power))
            mean = float(np.mean(region_power))
            snr = peak - noise_floor_db

            bursts.append(BurstSegment(
                start_sample=int(start_sample),
                end_sample=int(end_sample),
                duration_s=(end_sample - start_sample) / sample_rate_hz,
                peak_power_db=peak,
                mean_power_db=mean,
                snr_db=snr,
            ))

        logger.info(
            "Detected %d burst(s) from %d raw region(s) "
            "(merged=%d, after min-duration filter=%d)",
            len(bursts), len(raw_regions), len(merged), len(bursts),
        )

        for i, b in enumerate(bursts):
            logger.debug(
                "  burst[%d]: samples %d–%d (%.3f ms), "
                "peak=%.1f dB, SNR=%.1f dB",
                i, b.start_sample, b.end_sample,
                b.duration_s * 1e3, b.peak_power_db, b.snr_db,
            )

        # --- Optional debug plot ---
        if debug_plot:
            self._plot(
                times, power_db, noise_floor_db, threshold,
                bursts, sample_rate_hz, plot_path,
            )

        return bursts

    # ------------------------------------------------------------------
    # Noise floor (static utility, kept from scaffold)
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_noise_floor_db(iq: np.ndarray) -> float:
        """Quick time-domain noise floor estimate using median magnitude.

        Parameters
        ----------
        iq : np.ndarray
            Complex IQ samples.

        Returns
        -------
        float
            Estimated noise floor in dB.
        """
        magnitude = np.abs(iq)
        median_mag = float(np.median(magnitude))
        noise_power = median_mag ** 2
        if noise_power == 0:
            return -np.inf
        return float(10.0 * np.log10(noise_power))

    @staticmethod
    def _estimate_noise_floor_stft(power_db: np.ndarray) -> float:
        """Robustly estimate noise floor from an STFT power profile.

        Uses a histogram-mode approach: bins the power values and picks
        the mode (most common power level) as the noise floor. This is
        robust even when the signal occupies >50% of the capture —
        the scenario where a simple median fails.

        Falls back to the 25th percentile if the histogram has fewer
        than 3 distinct bins (very short captures).

        Parameters
        ----------
        power_db : np.ndarray
            Per-frame power in dB from the STFT.

        Returns
        -------
        float
            Estimated noise floor in dB.
        """
        if power_db.size < 3:
            return float(np.min(power_db))

        # Sturges' rule for bin count, clamped to [10, 200]
        n_bins = min(200, max(10, int(np.ceil(np.log2(power_db.size)) + 1)))
        counts, bin_edges = np.histogram(power_db, bins=n_bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # The noise floor is the most populated bin (mode)
        mode_idx = int(np.argmax(counts))
        noise_est = float(bin_centers[mode_idx])

        # Sanity check: the mode should be below the median.
        # If not (e.g., entire capture is signal), fall back to 25th percentile.
        median_db = float(np.median(power_db))
        if noise_est > median_db:
            noise_est = float(np.percentile(power_db, 25))
            logger.debug(
                "Histogram mode (%.1f dB) above median (%.1f dB) — "
                "falling back to 25th percentile (%.1f dB)",
                bin_centers[mode_idx], median_db, noise_est,
            )

        return noise_est

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _label_regions(mask: np.ndarray) -> list[tuple[int, int]]:
        """Find contiguous True regions in a boolean array.

        Returns (start, end) pairs where end is exclusive.
        """
        if mask.size == 0:
            return []

        diff = np.diff(mask.astype(np.int8))
        # Rising edges: 0→1 means the region starts at index+1
        # Falling edges: 1→0 means the region ends at index+1
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        # Handle edge cases: signal starts or ends above threshold
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [mask.size]])

        return list(zip(starts.tolist(), ends.tolist()))

    @staticmethod
    def _merge_regions(
        regions: list[tuple[int, int]],
        max_gap: int,
    ) -> list[tuple[int, int]]:
        """Merge regions separated by ≤ max_gap frames."""
        if not regions:
            return []

        merged: list[tuple[int, int]] = [regions[0]]
        for start, end in regions[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= max_gap:
                # Extend the previous region
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        return merged

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    @staticmethod
    def _plot(
        times: np.ndarray,
        power_db: np.ndarray,
        noise_floor_db: float,
        threshold_db: float,
        bursts: list[BurstSegment],
        sample_rate_hz: float,
        save_path: Path | None,
    ) -> None:
        """Render a diagnostic plot of the detection result."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 4))
        times_ms = times * 1e3

        ax.plot(times_ms, power_db, linewidth=0.5, color="steelblue", label="Power")
        ax.axhline(
            noise_floor_db, color="gray", linestyle="--",
            linewidth=0.8, label=f"Noise floor ({noise_floor_db:.1f} dB)",
        )
        ax.axhline(
            threshold_db, color="red", linestyle="--",
            linewidth=0.8, label=f"Threshold ({threshold_db:.1f} dB)",
        )

        for i, b in enumerate(bursts):
            t_start = b.start_sample / sample_rate_hz * 1e3
            t_end = b.end_sample / sample_rate_hz * 1e3
            ax.axvspan(
                t_start, t_end, alpha=0.2, color="orange",
                label="Burst" if i == 0 else None,
            )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Power (dB)")
        ax.set_title("Burst Detection — STFT Power Profile")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
            logger.info("Debug plot saved to %s", save_path)
            plt.close(fig)
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Standalone test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    )

    # --- Synthesize a test signal ---
    # 50 MHz sample rate, 10 ms of noise with two embedded bursts
    fs = 50e6
    duration_s = 10e-3
    n_samples = int(fs * duration_s)

    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(
        np.complex64
    ) * 0.01  # low-power noise

    # Burst 1: 1 ms tone at 2 MHz offset, starting at 2 ms
    burst1_start = int(2e-3 * fs)
    burst1_len = int(1e-3 * fs)
    t1 = np.arange(burst1_len) / fs
    noise[burst1_start : burst1_start + burst1_len] += (
        0.5 * np.exp(2j * np.pi * 2e6 * t1)
    ).astype(np.complex64)

    # Burst 2: 0.5 ms tone at -5 MHz offset, starting at 6 ms
    burst2_start = int(6e-3 * fs)
    burst2_len = int(0.5e-3 * fs)
    t2 = np.arange(burst2_len) / fs
    noise[burst2_start : burst2_start + burst2_len] += (
        0.3 * np.exp(-2j * np.pi * 5e6 * t2)
    ).astype(np.complex64)

    # --- Run detector ---
    detector = BurstDetector(
        threshold_db=10.0,
        min_duration_s=100e-6,
        merge_gap_s=200e-6,
        nperseg=1024,
        overlap_frac=0.5,
    )

    results = detector.detect(noise, sample_rate_hz=fs, debug_plot=True)

    print(f"\n{'='*60}")
    print(f"Test signal: {n_samples} samples at {fs/1e6:.0f} MHz")
    print(f"Injected bursts:")
    print(f"  #1: sample {burst1_start}–{burst1_start+burst1_len} "
          f"({burst1_len/fs*1e3:.1f} ms)")
    print(f"  #2: sample {burst2_start}–{burst2_start+burst2_len} "
          f"({burst2_len/fs*1e3:.1f} ms)")
    print(f"{'='*60}")
    print(f"Detected {len(results)} burst(s):")
    for i, b in enumerate(results):
        print(f"  [{i}] samples {b.start_sample}–{b.end_sample} "
              f"| {b.duration_s*1e3:.2f} ms "
              f"| peak {b.peak_power_db:.1f} dB "
              f"| SNR {b.snr_db:.1f} dB")

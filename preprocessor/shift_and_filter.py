"""
shift_and_filter.py — Frequency-shift and bandlimit a wideband IQ capture.

When DroneID coexists with a wider OcuSync video link on the same capture,
DroneSecurity's per-burst ``helpers.estimate_offset`` finds a single 16–20
MHz blob above mean PSD (video + DroneID merged) instead of the 8–11 MHz
shape it requires, and rejects every burst with "cfo MISMATCH".

This preprocessor isolates the DroneID emission *before* the decoder sees
the file:

1. Mix the IQ down so the user-supplied centre frequency lands at DC.
2. Lowpass FIR-filter to keep only a ±bw/2 window around DC, killing the
   video and any other wideband emission.
3. Optionally decimate to a lower sample rate (the bandwidth allows it).
4. Write the cleaned signal to a new ``.fc32`` file so downstream
   decoders (subprocess or in-process) consume it unchanged.

Streaming: the file is read in fixed-size chunks; the mixer keeps a
continuous phase across chunks, and ``scipy.signal.lfilter`` preserves
the filter state — so memory is bounded regardless of input size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import math

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

logger = logging.getLogger(__name__)


@dataclass
class ShiftFilterResult:
    """Metadata returned by :func:`shift_and_filter`."""

    output_path: Path
    input_samples: int
    output_samples: int
    input_rate_hz: float
    output_rate_hz: float
    shift_hz: float
    keep_bandwidth_hz: float
    decim_q: int


def shift_and_filter(
    input_path: Path | str,
    output_path: Path | str,
    sample_rate_hz: float,
    shift_hz: float,
    keep_bandwidth_hz: float = 10e6,
    output_rate_hz: float | None = None,
    chunk_samples: int = 1 << 22,
    fir_taps: int = 257,
) -> ShiftFilterResult:
    """Stream-process an interleaved-float32 IQ file.

    Parameters
    ----------
    input_path, output_path : Path | str
        Input and output ``.fc32`` files (complex64 = interleaved float32).
    sample_rate_hz : float
        Sample rate of the input file.
    shift_hz : float
        Frequency of the band of interest **in the input spectrum** (e.g.
        ``-16.6e6`` if the signal is at -16.6 MHz from the capture
        centre). Internally the signal is multiplied by
        ``exp(-2j*pi*shift_hz*t)`` so the band lands at DC.
    keep_bandwidth_hz : float
        Total bandwidth (centred on DC after shift) preserved by the
        lowpass. The FIR cutoff is set to ``keep_bandwidth_hz / 2``.
    output_rate_hz : float | None
        If provided and lower than ``sample_rate_hz``, decimate to this
        rate by integer factor. Must divide ``sample_rate_hz`` evenly.
        If ``None`` (default), no decimation is performed.
    chunk_samples : int
        Stream chunk size in complex samples (≈4 MS by default → 32 MB
        complex64 working set per chunk).
    fir_taps : int
        Lowpass FIR length. 257 taps gives a ~30 dB transition over
        ~1% of fs, which is more than enough to suppress an adjacent
        20 MHz OcuSync emission when we're keeping a 10 MHz window.

    Returns
    -------
    ShiftFilterResult
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if keep_bandwidth_hz <= 0:
        raise ValueError("keep_bandwidth_hz must be > 0")
    if keep_bandwidth_hz >= sample_rate_hz:
        raise ValueError(
            "keep_bandwidth_hz must be smaller than sample_rate_hz"
        )

    # Output-rate handling has three regimes:
    #   - no rate change                               (decim_q=1, no poly)
    #   - integer ratio (rate divides evenly)          (integer decimate)
    #   - non-integer ratio (e.g. 50→15.36 Msps)       (integer decimate +
    #                                                   polyphase remainder)
    decim_q = 1
    fs_out = sample_rate_hz
    poly_up, poly_down = 1, 1     # final polyphase correction step
    if output_rate_hz is not None and output_rate_hz < sample_rate_hz:
        q_f = sample_rate_hz / output_rate_hz
        decim_q = int(round(q_f))
        if abs(q_f - decim_q) > 1e-6 or decim_q < 1:
            # Non-integer ratio. Pick the largest integer q such that
            # sample_rate_hz / q is still > output_rate_hz; the leftover
            # is done by resample_poly with small integer up:down so the
            # FIR filter remains tractable.
            decim_q = max(1, int(sample_rate_hz // output_rate_hz))
            intermediate_fs = sample_rate_hz / decim_q
            # Express the polyphase ratio in its lowest integer terms
            # using the *original* rates (not the rounded intermediate),
            # otherwise gcd reduction fails and we get huge filters.
            #   poly_up   output_rate         output_rate × decim_q
            #   ------- = ----------------- = ---------------------
            #   poly_down (sample_rate / q)        sample_rate
            num = int(round(output_rate_hz * decim_q))
            den = int(round(sample_rate_hz))
            g = math.gcd(num, den)
            poly_up = num // g
            poly_down = den // g
            fs_out = output_rate_hz
            logger.info(
                "shift_and_filter: non-integer rate ratio; integer q=%d "
                "to %.3f MHz, then polyphase %d:%d to %.3f MHz",
                decim_q, intermediate_fs / 1e6,
                poly_up, poly_down, fs_out / 1e6,
            )
        else:
            fs_out = sample_rate_hz / decim_q
        if keep_bandwidth_hz > fs_out * 0.9:
            raise ValueError(
                f"keep_bandwidth_hz {keep_bandwidth_hz/1e6:.1f} MHz too "
                f"wide for output rate {fs_out/1e6:.1f} MHz (aliasing)"
            )

    cutoff_hz = keep_bandwidth_hz / 2.0
    fir_b = firwin(
        fir_taps, cutoff_hz, fs=sample_rate_hz, window="hamming"
    ).astype(np.float32)
    fir_zi = np.zeros(fir_taps - 1, dtype=np.complex64)

    phase_step = np.float32(-2.0 * np.pi * shift_hz / sample_rate_hz)

    total_in_samples = input_path.stat().st_size // 8
    bytes_per_sample = 8  # complex64
    logger.info(
        "shift_and_filter: %s -> %s (fs=%.3f MHz, shift=%+.3f MHz, "
        "keep=±%.2f MHz, decim_q=%d, fs_out=%.3f MHz)",
        input_path.name, output_path.name,
        sample_rate_hz / 1e6, shift_hz / 1e6,
        cutoff_hz / 1e6, decim_q, fs_out / 1e6,
    )

    # If a polyphase correction step is needed we buffer the
    # integer-decimated output in memory and do the final resample
    # once. Otherwise we stream straight to disk.
    needs_poly = (poly_up, poly_down) != (1, 1)
    poly_buffer: list[np.ndarray] = [] if needs_poly else []
    written = 0
    cursor = 0
    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        while cursor < total_in_samples:
            end = min(cursor + chunk_samples, total_in_samples)
            n = end - cursor
            fin.seek(cursor * bytes_per_sample)
            raw = np.fromfile(fin, dtype=np.float32, count=n * 2)
            if raw.size != n * 2:
                logger.warning(
                    "short read at sample %d: got %d, expected %d",
                    cursor, raw.size, n * 2,
                )
                n = raw.size // 2
                raw = raw[: n * 2]
            iq = raw.view(np.complex64)

            # Continuous-phase rotator across chunks.
            idx = np.arange(cursor, cursor + n, dtype=np.float32)
            phase = idx * phase_step
            rotator = np.empty(n, dtype=np.complex64)
            np.cos(phase, out=rotator.view(np.float32)[0::2])
            np.sin(phase, out=rotator.view(np.float32)[1::2])
            shifted = iq * rotator
            del rotator, phase, idx, raw

            # Stateful lowpass — preserves continuity across chunks.
            filtered, fir_zi = lfilter(fir_b, 1.0, shifted, zi=fir_zi)
            filtered = filtered.astype(np.complex64, copy=False)
            del shifted

            if decim_q > 1:
                filtered = np.ascontiguousarray(filtered[::decim_q])

            if needs_poly:
                poly_buffer.append(filtered)
            else:
                filtered.view(np.float32).tofile(fout)
                written += len(filtered)
            cursor = end

        if needs_poly:
            # One-shot polyphase resample to land exactly on the target rate.
            buf = np.concatenate(poly_buffer) if len(poly_buffer) > 1 else poly_buffer[0]
            del poly_buffer
            logger.info(
                "shift_and_filter: polyphase resample %d samples @ %.3f MHz "
                "-> %d:%d -> %.3f MHz",
                len(buf), sample_rate_hz / decim_q / 1e6,
                poly_up, poly_down, fs_out / 1e6,
            )
            out_buf = resample_poly(buf, poly_up, poly_down).astype(np.complex64)
            del buf
            out_buf.view(np.float32).tofile(fout)
            written = len(out_buf)
            del out_buf

    logger.info(
        "shift_and_filter: wrote %d samples (%.3fs at %.3f MHz)",
        written, written / fs_out, fs_out / 1e6,
    )

    return ShiftFilterResult(
        output_path=output_path,
        input_samples=total_in_samples,
        output_samples=written,
        input_rate_hz=sample_rate_hz,
        output_rate_hz=fs_out,
        shift_hz=shift_hz,
        keep_bandwidth_hz=keep_bandwidth_hz,
        decim_q=decim_q,
    )

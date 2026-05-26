"""
plot_zc_correlator.py — Generate the ZC correlator figure for the PFE report.

Produces a single-panel matplotlib figure showing the normalized ZC
correlation magnitude versus sample index for a reference IQ capture,
with the detection threshold drawn as a dashed horizontal line and the
detected peaks marked. Intended for the "ZC Correlator Design"
subsection of the report.

Default capture is ``data/samples/mavic_air_2``. The plot is rendered at
publication DPI (300) and saved as PNG by default; pass ``--pdf`` to
also write a vector PDF for LaTeX inclusion.

Usage
-----
    python plot_zc_correlator.py
    python plot_zc_correlator.py --input data/samples/mavic_air_2 \\
        --sample-rate 50e6 --threshold 0.15 --pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(_PIPELINE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT.parent))

from uav_telemetry_pipeline.detection.zc_correlator import (  # noqa: E402
    BASE_RATE_HZ,
    ROOT_FINE,
    ZadoffChuCorrelator,
)


# ---------------------------------------------------------------- IQ loading


def load_iq(path: Path) -> np.ndarray:
    """Load an interleaved float32 IQ capture as a complex64 array."""
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 2:
        raw = raw[:-1]
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


# --------------------------------------------------------------- main plot


def plot_correlator_output(
    iq: np.ndarray,
    sample_rate_hz: float,
    out_png: Path,
    threshold: float = 0.15,
    root_index: int = ROOT_FINE,
    write_pdf: bool = False,
    time_axis_ms: bool = True,
) -> dict:
    """Run the correlator and save the figure. Returns a small summary dict."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    correlator = ZadoffChuCorrelator(
        sample_rate_hz=sample_rate_hz,
        root_index=root_index,
        detection_threshold=threshold,
        auto_cfo=True,
    )

    # `correlate()` returns the magnitude at the *decimated* sample rate.
    # Mapping decimated index -> input-rate sample index requires multiplying
    # by ``decim_q`` (same convention as ``find_frame_starts``).
    corr = correlator.correlate(iq)
    q = correlator.decim_q
    decim_fs = correlator.decim_fs

    # Detected peaks (decimated-rate indices, then expanded for plotting).
    from scipy.signal import find_peaks

    peak_idx_decim, _ = find_peaks(
        corr,
        height=threshold,
        distance=correlator.min_spacing_samples,
    )

    # X axis: choose between sample index (input rate) or time (ms).
    n_decim = len(corr)
    x_decim = np.arange(n_decim, dtype=np.float64)
    if time_axis_ms:
        x_full = x_decim / decim_fs * 1e3  # ms
        x_peaks = peak_idx_decim / decim_fs * 1e3
        xlabel = "Time (ms)"
    else:
        # Report input-rate sample indices so the figure is comparable to the
        # raw capture timeline rather than the internal decimated stream.
        x_full = x_decim * q
        x_peaks = peak_idx_decim * q
        xlabel = "Sample index (input rate)"

    # Figure.
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.plot(x_full, corr, color="#1f4e79", linewidth=0.9,
            label="Normalized correlation magnitude")
    ax.axhline(threshold, color="#b34a4a", linestyle="--", linewidth=1.2,
               label=f"Detection threshold ({threshold:.2f})")

    if len(peak_idx_decim) > 0:
        peak_vals = corr[peak_idx_decim]
        ax.scatter(x_peaks, peak_vals, s=42, facecolor="none",
                   edgecolor="#b34a4a", linewidths=1.5, zorder=5,
                   label=f"Detected peaks (n={len(peak_idx_decim)})")
        # Annotate each peak with its index value just above the marker.
        ymax = max(1.0, float(corr.max()) + 0.05)
        for xp, yp in zip(x_peaks, peak_vals):
            ax.annotate(
                f"{yp:.2f}",
                xy=(xp, yp),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#b34a4a",
            )
        ax.set_ylim(0.0, ymax)
    else:
        ax.set_ylim(0.0, max(1.0, float(corr.max()) + 0.05))

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized correlation magnitude")
    ax.set_title(
        f"ZC matched-filter output — root {root_index}, "
        f"$f_s$ = {sample_rate_hz/1e6:.2f} MHz "
        f"(decimated to {decim_fs/1e6:.2f} MHz, q={q})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    if write_pdf:
        fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)

    return {
        "n_samples_input": int(len(iq)),
        "n_samples_decimated": int(n_decim),
        "decim_q": int(q),
        "decim_fs_hz": float(decim_fs),
        "threshold": float(threshold),
        "n_peaks": int(len(peak_idx_decim)),
        "peaks_above_threshold": [
            {"sample_index_input": int(idx * q),
             "time_ms": float(idx / decim_fs * 1e3),
             "value": float(corr[idx])}
            for idx in peak_idx_decim
        ],
        "max_correlation": float(corr.max()),
        "cfo_hz_estimated": float(correlator.last_cfo_hz),
        "output_png": str(out_png),
    }


# ---------------------------------------------------------------------- CLI


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i",
                   default=str(_PIPELINE_ROOT / "data" / "samples" / "mavic_air_2"),
                   help="IQ capture (interleaved float32). "
                        "Default: data/samples/mavic_air_2")
    p.add_argument("--sample-rate", "-r", type=float, default=50e6,
                   help="IQ sample rate in Hz (default: 50e6)")
    p.add_argument("--threshold", "-t", type=float, default=0.15,
                   help="Detection threshold (default: 0.15)")
    p.add_argument("--root", type=int, default=ROOT_FINE,
                   help=f"ZC root index (default: {ROOT_FINE}, fine-sync)")
    p.add_argument("--output", "-o",
                   default=str(_PIPELINE_ROOT / "results" / "zc_correlator_output.png"),
                   help="Output PNG path (default: results/zc_correlator_output.png)")
    p.add_argument("--pdf", action="store_true",
                   help="Also write a vector PDF alongside the PNG")
    p.add_argument("--samples-axis", action="store_true",
                   help="Use input-rate sample index on the x-axis instead of ms")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress info-level logging")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")

    print(f"[plot_zc] loading {in_path}")
    iq = load_iq(in_path)
    print(f"[plot_zc] {len(iq):,} samples = {len(iq)/args.sample_rate*1e3:.2f} ms"
          f" at {args.sample_rate/1e6:.2f} Msps")

    summary = plot_correlator_output(
        iq=iq,
        sample_rate_hz=args.sample_rate,
        out_png=out_path,
        threshold=args.threshold,
        root_index=args.root,
        write_pdf=args.pdf,
        time_axis_ms=not args.samples_axis,
    )

    print(f"[plot_zc] wrote {summary['output_png']}")
    if args.pdf:
        print(f"[plot_zc] wrote {out_path.with_suffix('.pdf')}")
    print(f"[plot_zc] peaks >= {summary['threshold']:.2f}: "
          f"{summary['n_peaks']} (max corr = {summary['max_correlation']:.3f}, "
          f"CFO = {summary['cfo_hz_estimated']/1e6:.3f} MHz)")
    for p in summary["peaks_above_threshold"]:
        print(f"    peak: t={p['time_ms']:.3f} ms, "
              f"sample_index={p['sample_index_input']}, value={p['value']:.3f}")


if __name__ == "__main__":
    main()

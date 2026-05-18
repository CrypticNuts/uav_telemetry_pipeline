#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end DroneID telemetry recovery runner.

Tries decoders in order — DroneSecurity (primary) → proto17 (fallback A) →
Native (fallback B) — and stops at the first one that returns at least one
CRC-OK frame. Always writes a JSON result file under
``<results>/<input-stem>_telemetry.json``, even when every decoder fails,
so downstream tooling can rely on the file existing.

Usage::

    python run_pipeline.py --input data/samples/mini2_sm --sample-rate 50e6
    python run_pipeline.py --input data/samples/phantom.fc32 \\
        --sample-rate 30.72e6 --diagnose

The ``--diagnose`` flag runs the ZC correlator alone after all decoders
fail and reports the best correlation score / sample index — useful to
decide whether the capture contains signal at all.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

# Allow running as a script from the pipeline root
_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(_PIPELINE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT.parent))

from uav_telemetry_pipeline.config import PipelineConfig  # noqa: E402
from uav_telemetry_pipeline.decoders import (  # noqa: E402
    BaseDecoder,
    DroneSecurityDecoder,
    NativeDecoder,
    Proto17Decoder,
    TelemetryFrame,
)
from uav_telemetry_pipeline.detection.zc_correlator import (  # noqa: E402
    ZadoffChuCorrelator,
    analyze_spectrum_bands,
)
from uav_telemetry_pipeline.preprocessor.shift_and_filter import (  # noqa: E402
    shift_and_filter,
)

logger = logging.getLogger("pipeline")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_decoders(
    config: PipelineConfig,
    legacy: bool,
    timeout_s: float,
    enable_proto17: bool = False,
) -> list[BaseDecoder]:
    """Build the decoder fallback chain.

    Proto17 is **disabled by default** because its Octave implementation
    of ``find_zc.m`` is interpreter-bound and routinely times out on
    multi-second captures without ever returning a result. Pass
    ``enable_proto17=True`` (or ``--enable-proto17`` on the CLI) to opt
    back in for ZC-burst counting.
    """
    decoders: list[BaseDecoder] = [
        DroneSecurityDecoder(config=config, timeout_s=timeout_s,
                             extra_args=["-l"] if legacy else None),
    ]
    if enable_proto17:
        decoders.append(Proto17Decoder(config=config))
    decoders.append(NativeDecoder(config=config, legacy=legacy))
    return decoders


def _crc_ok_count(frames: list[TelemetryFrame]) -> int:
    return sum(1 for f in frames if f.get("crc_ok"))


def _first_gps_fix(frames: list[TelemetryFrame]) -> tuple[float, float] | None:
    for f in frames:
        lat = f.get("lat", 0.0) or 0.0
        lon = f.get("lon", 0.0) or 0.0
        if lat != 0.0 and lon != 0.0:
            return float(lat), float(lon)
    for f in frames:
        lat = f.get("app_lat", 0.0) or 0.0
        lon = f.get("app_lon", 0.0) or 0.0
        if lat != 0.0 and lon != 0.0:
            return float(lat), float(lon)
    return None


def _run_chain(
    decoders: list[BaseDecoder],
    iq_file: str,
    sample_rate: float,
) -> tuple[str, list[TelemetryFrame], list[dict]]:
    """Run decoders in order, return (winning_decoder, frames, attempt_log)."""
    attempts: list[dict] = []
    for dec in decoders:
        if not dec.is_available():
            attempts.append({"decoder": dec.name, "status": "unavailable",
                             "frames": 0, "crc_ok": 0})
            logger.info("Skipping %s (unavailable)", dec.name)
            continue
        logger.info("=== %s ===", dec.name)
        try:
            frames = dec.decode(iq_file, sample_rate)
        except Exception as exc:  # noqa: BLE001 - decoders are external
            logger.exception("%s crashed: %s", dec.name, exc)
            attempts.append({"decoder": dec.name, "status": "error",
                             "error": str(exc), "frames": 0, "crc_ok": 0})
            continue

        crc_ok = _crc_ok_count(frames)
        attempts.append({
            "decoder": dec.name,
            "status": "ran",
            "frames": len(frames),
            "crc_ok": crc_ok,
        })
        if crc_ok > 0:
            return dec.name, frames, attempts
        if frames:
            # Returned frames but no CRC pass — keep going but remember
            attempts[-1]["status"] = "no_crc_ok"
    return "", [], attempts


def _diagnose(
    iq_file: str,
    sample_rate: float,
    chunk_seconds: float = 1.5,
    threshold: float = 0.15,
) -> dict:
    """Run the ZC correlator alone for a diagnostic report.

    Processes the capture in non-overlapping chunks of ``chunk_seconds``
    each so memory usage stays bounded on multi-GB recordings. CFO is
    estimated per chunk (it can drift across captures). Returns peak
    statistics aggregated across all chunks plus the top correlation
    scores so the caller can judge whether signal is present at all.
    """
    iq_path = Path(iq_file)
    file_bytes = iq_path.stat().st_size
    total_samples = file_bytes // 8  # complex64 = 2 * float32
    total_seconds = total_samples / sample_rate

    chunk_samples = int(sample_rate * chunk_seconds)

    corr = ZadoffChuCorrelator(
        sample_rate_hz=sample_rate,
        detection_threshold=threshold,
    )
    ref_len = len(corr.generate_reference())

    all_peaks: list[tuple[int, float]] = []  # (absolute sample idx, score)
    chunk_reports: list[dict] = []
    best_score_global = 0.0
    best_idx_global = 0
    cursor = 0
    import time
    file_handle = open(iq_path, "rb")
    try:
        bytes_per_sample = 8  # complex64 = 2 * float32
        chunk_idx = 0
        while cursor < total_samples:
            end = min(cursor + chunk_samples, total_samples)
            n = end - cursor
            if n < ref_len * 4:
                break
            chunk_idx += 1
            t0 = time.monotonic()
            file_handle.seek(cursor * bytes_per_sample)
            raw = np.fromfile(
                file_handle, dtype=np.float32, count=n * 2,
            )
            iq = raw.view(np.complex64).copy()
            del raw
            t1 = time.monotonic()
            logger.info("chunk %d: load %.2fs (%d samples, start=%d)",
                        chunk_idx, t1 - t0, n, cursor)

            t2 = time.monotonic()
            corr_out = corr.correlate(iq)
            t3 = time.monotonic()
            logger.info("chunk %d: correlate %.2fs", chunk_idx, t3 - t2)
            peaks, props = find_peaks(
                corr_out, height=threshold, distance=corr.min_spacing_samples,
            )
            # corr_out is at the decimated rate; map peak indices back to the
            # input rate so all reported sample indices share a single axis.
            q = corr.decim_q
            for p_local, h in zip(peaks, props["peak_heights"]):
                all_peaks.append((int(cursor + p_local * q), float(h)))

            local_best_idx = int(np.argmax(corr_out))
            local_best_score = float(corr_out[local_best_idx])
            if local_best_score > best_score_global:
                best_score_global = local_best_score
                best_idx_global = cursor + local_best_idx * q

            chunk_reports.append({
                "sample_start": int(cursor),
                "samples": int(n),
                "cfo_mhz": round(corr.last_cfo_hz / 1e6, 3),
                "best_score": round(local_best_score, 3),
                "best_sample_index": int(cursor + local_best_idx),
                "peaks_above_threshold": int(len(peaks)),
            })

            del iq, corr_out
            cursor = end
            logger.info(
                "chunk %d done: peaks_above=%d, local_best=%.3f, total %.2fs",
                chunk_idx, len(peaks), local_best_score,
                time.monotonic() - t0,
            )
    finally:
        file_handle.close()

    all_peaks.sort(key=lambda t: -t[1])
    top_peaks = [
        {"sample_index": idx, "score": round(score, 3)}
        for idx, score in all_peaks[:20]
    ]
    return {
        "file_seconds": round(total_seconds, 3),
        "samples_analyzed": int(cursor),
        "threshold": threshold,
        "best_score": round(best_score_global, 3),
        "best_sample_index": int(best_idx_global),
        "burst_min_spacing_samples": int(corr.min_spacing_samples * corr.decim_q),
        "decim_q": int(corr.decim_q),
        "peaks_above_threshold": len(all_peaks),
        "top_peaks": top_peaks,
        "chunks": chunk_reports,
    }


def _spectrum_only(
    iq_file: str,
    sample_rate: float,
    chunk_seconds: float = 1.0,
) -> list[dict]:
    """Per-chunk Welch band analysis (no ZC correlation). Descriptive only."""
    import time as _time
    iq_path = Path(iq_file)
    file_bytes = iq_path.stat().st_size
    total_samples = file_bytes // 8
    chunk_samples = int(sample_rate * chunk_seconds)

    reports: list[dict] = []
    file_handle = open(iq_path, "rb")
    try:
        bytes_per_sample = 8
        cursor = 0
        chunk_idx = 0
        while cursor < total_samples:
            end = min(cursor + chunk_samples, total_samples)
            n = end - cursor
            if n < 2048:
                break
            chunk_idx += 1
            t0 = _time.monotonic()
            file_handle.seek(cursor * bytes_per_sample)
            raw = np.fromfile(file_handle, dtype=np.float32, count=n * 2)
            iq = raw.view(np.complex64).copy()
            del raw
            bands = analyze_spectrum_bands(iq, sample_rate)
            reports.append({
                "sample_start": int(cursor),
                "samples": int(n),
                "bands": bands,
            })
            logger.info(
                "spectrum chunk %d: %d band(s) %.2fs",
                chunk_idx, len(bands), _time.monotonic() - t0,
            )
            for b in bands:
                logger.info(
                    "    %-14s  center=%+7.2f MHz  bw=%5.2f MHz  +%4.1f dB",
                    b["label"], b["f_center_hz"] / 1e6,
                    b["bandwidth_hz"] / 1e6, b["peak_psd_db_over_mean"],
                )
            del iq
            cursor = end
    finally:
        file_handle.close()
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="DroneID telemetry pipeline runner")
    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="IQ capture file (interleaved float32)")
    parser.add_argument("--sample-rate", "-s", required=True, type=float,
                        help="Capture sample rate in Hz (e.g. 50e6)")
    parser.add_argument("--legacy", action="store_true",
                        help="Treat as legacy drone (Mavic Pro / Mavic 2)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output JSON path (default: results/<stem>_telemetry.json)")
    parser.add_argument("--decoder-timeout", type=float, default=1800.0,
                        help="Per-decoder subprocess timeout in seconds")
    parser.add_argument("--diagnose", action="store_true",
                        help="If all decoders fail, run ZC correlator diagnostic")
    parser.add_argument("--diagnose-only", action="store_true",
                        help="Skip decoders, only run the ZC correlator diagnostic")
    parser.add_argument("--diagnose-chunk-seconds", type=float, default=1.5,
                        help="Diagnostic chunk window in seconds (default: 1.5)")
    parser.add_argument("--diagnose-threshold", type=float, default=0.15,
                        help="ZC correlation threshold for diagnostic (default: 0.15)")
    parser.add_argument("--spectrum-only", action="store_true",
                        help="Skip decoders, run Welch PSD band-analysis. "
                             "Reports all emissions wider than 1 MHz with "
                             "heuristic labels (droneid / c2 / ocusync_video).")
    parser.add_argument("--center-shift-hz", type=float, default=None,
                        help="Pre-shift the IQ by this offset (Hz) so the "
                             "band of interest lands at DC before decoding. "
                             "Use the value from --spectrum-only's "
                             "bands[i].f_center_hz (typical: -16.6e6 for the "
                             "Inspire 2 corne capture).")
    parser.add_argument("--keep-bandwidth-hz", type=float, default=10e6,
                        help="Bandwidth (Hz) kept by the pre-filter after "
                             "--center-shift-hz. Default 10 MHz suits "
                             "DroneID. Set wider to keep more context.")
    parser.add_argument("--output-rate-hz", type=float, default=None,
                        help="Decimate the pre-shifted file to this rate "
                             "(must divide --sample-rate evenly). Default: "
                             "no decimation (keeps decoders simpler).")
    parser.add_argument("--keep-shifted-file", action="store_true",
                        help="Keep the temporary shift-filtered IQ file "
                             "after the run (useful for debugging).")
    parser.add_argument("--enable-proto17", action="store_true",
                        help="Include the Proto17 (Octave) decoder in the "
                             "fallback chain. Disabled by default because "
                             "find_zc.m in Octave is very slow and almost "
                             "always hits the 10-minute timeout without "
                             "producing useful output.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    if not args.input.is_file():
        logger.error("Input file not found: %s", args.input)
        return 2

    config = PipelineConfig.load()
    config.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or (
        config.results_dir / f"{args.input.stem}_telemetry.json"
    )

    logger.info("Input: %s (%.3f MHz)", args.input, args.sample_rate / 1e6)
    logger.info("DroneSecurity path: %s", config.dronesecurity_path)
    logger.info("Output: %s", output_path)

    if args.spectrum_only:
        spectrum_report = _spectrum_only(
            str(args.input), args.sample_rate,
            chunk_seconds=args.diagnose_chunk_seconds,
        )
        out_doc = {
            "input": str(args.input),
            "sample_rate_hz": args.sample_rate,
            "mode": "spectrum_only",
            "spectrum": spectrum_report,
        }
        output_path.write_text(json.dumps(out_doc, indent=2))
        print()
        print("=" * 60)
        print(f"  mode:          spectrum_only (descriptive)")
        print(f"  chunks:        {len(spectrum_report)}")
        labels: dict[str, int] = {}
        for ch in spectrum_report:
            for b in ch["bands"]:
                labels[b["label"]] = labels.get(b["label"], 0) + 1
        if labels:
            print(f"  emissions:     {labels}")
        print(f"  results:       {output_path}")
        print("=" * 60)
        return 0

    # Optional pre-shift / bandpass when the band of interest is not at DC.
    # Useful when DroneID coexists with wider OcuSync video and the per-burst
    # CFO estimator can't isolate the DroneID band-shape from the blob.
    decoder_input_path = str(args.input)
    decoder_input_rate = args.sample_rate
    shifted_artifact: dict | None = None
    shifted_temp_path: Path | None = None
    if args.center_shift_hz is not None:
        import tempfile
        suffix = f"{args.input.stem}_shifted_{int(args.center_shift_hz/1e3)}kHz.fc32"
        shifted_temp_path = Path(tempfile.gettempdir()) / f"uav_pipeline_{suffix}"
        logger.info(
            "Pre-shifting by %+.3f MHz, bandlimit ±%.2f MHz, output %s",
            args.center_shift_hz / 1e6,
            args.keep_bandwidth_hz / 2e6,
            shifted_temp_path,
        )
        shifted = shift_and_filter(
            input_path=args.input,
            output_path=shifted_temp_path,
            sample_rate_hz=args.sample_rate,
            shift_hz=args.center_shift_hz,
            keep_bandwidth_hz=args.keep_bandwidth_hz,
            output_rate_hz=args.output_rate_hz,
        )
        decoder_input_path = str(shifted.output_path)
        decoder_input_rate = shifted.output_rate_hz
        shifted_artifact = {
            "path": str(shifted.output_path),
            "shift_hz": shifted.shift_hz,
            "keep_bandwidth_hz": shifted.keep_bandwidth_hz,
            "input_rate_hz": shifted.input_rate_hz,
            "output_rate_hz": shifted.output_rate_hz,
            "input_samples": shifted.input_samples,
            "output_samples": shifted.output_samples,
            "decim_q": shifted.decim_q,
        }

    if args.diagnose_only:
        winner, frames, attempts = "", [], []
    else:
        decoders = _build_decoders(
            config,
            legacy=args.legacy,
            timeout_s=args.decoder_timeout,
            enable_proto17=args.enable_proto17,
        )
        winner, frames, attempts = _run_chain(
            decoders, decoder_input_path, decoder_input_rate,
        )

    diagnostic: dict | None = None
    if (not frames and args.diagnose) or args.diagnose_only:
        logger.info("Running ZC diagnostic …")
        try:
            diagnostic = _diagnose(
                str(args.input), args.sample_rate,
                chunk_seconds=args.diagnose_chunk_seconds,
                threshold=args.diagnose_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Diagnostic failed: %s", exc)
            diagnostic = {"error": str(exc)}

    crc_ok = _crc_ok_count(frames)
    fix = _first_gps_fix(frames)
    result_doc = {
        "input": str(args.input),
        "sample_rate_hz": args.sample_rate,
        "preprocessing": shifted_artifact,
        "decoder_used": winner or None,
        "frames_decoded": len(frames),
        "crc_ok_frames": crc_ok,
        "crc_pass_rate": (crc_ok / len(frames)) if frames else 0.0,
        "first_gps_fix": {"lat": fix[0], "lon": fix[1]} if fix else None,
        "attempts": attempts,
        "diagnostic": diagnostic,
        "frames": frames,
    }
    output_path.write_text(json.dumps(result_doc, indent=2))

    if shifted_temp_path is not None and not args.keep_shifted_file:
        try:
            shifted_temp_path.unlink(missing_ok=True)
            logger.info("Cleaned up temp file: %s", shifted_temp_path)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", shifted_temp_path, exc)

    print()
    print("=" * 60)
    print(f"  decoder used:  {winner or '(none — all decoders failed)'}")
    print(f"  frames:        {len(frames)}")
    print(f"  CRC OK:        {crc_ok}")
    print(
        f"  CRC pass rate: "
        f"{(100*crc_ok/len(frames)) if frames else 0:.1f}%"
    )
    if fix:
        print(f"  first fix:     lat={fix[0]:.6f}, lon={fix[1]:.6f}")
    else:
        print("  first fix:     (no GPS position recovered)")
    if diagnostic and "error" not in diagnostic:
        print(
            f"  ZC diag:       best_score={diagnostic.get('best_score', 0):.3f} "
            f"@ sample {diagnostic.get('best_sample_index')}, "
            f"peaks>={diagnostic.get('threshold')}: "
            f"{diagnostic.get('peaks_above_threshold')} "
            f"(over {diagnostic.get('file_seconds')}s)"
        )
        top = diagnostic.get("top_peaks") or []
        for p in top[:5]:
            print(f"     peak: sample={p['sample_index']}, score={p['score']}")
    print(f"  results:       {output_path}")
    print("=" * 60)

    return 0 if frames else 1


if __name__ == "__main__":
    sys.exit(main())

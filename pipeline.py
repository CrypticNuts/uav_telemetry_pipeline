"""
pipeline.py — Main entry point for the UAV telemetry pipeline.

Orchestrates the full processing chain:
  1. Load IQ data (Case A or Case B)
  2. Detect signal bursts
  3. Correlate with Zadoff-Chu to find DroneID frames
  4. Decode frames via external reference decoder
  5. Parse and output structured telemetry

Usage:
    python pipeline.py --input capture.fc32 --sample-rate 50e6 --center-freq 2.4e9
    python pipeline.py --input data.csv --csv-mode
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from preprocessor.csv_to_iq import convert_csv_to_iq
from preprocessor.load_verified_iq import load_verified_iq
from detection.burst_detector import BurstDetector, BurstSegment
from detection.zc_correlator import ZadoffChuCorrelator
from decoding.reference_decoder import ReferenceDecoder
from telemetry.droneid_parser import DroneIDParser
from telemetry.models import DecodeResult, InputClassification, DroneIDFrame

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """Set up structured logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="UAV Telemetry Pipeline — DJI DroneID Recovery",
    )

    # --- Input ---
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input file (IQ capture or CSV).",
    )
    parser.add_argument(
        "--csv-mode",
        action="store_true",
        help="Treat input as CSV (Case B / exploratory data).",
    )
    parser.add_argument(
        "--fmt",
        type=str,
        default=None,
        help="Explicit IQ format (e.g. fc32, sc16, sc8). "
             "Required for files without an extension.",
    )

    # --- RF parameters ---
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=50e6,
        help="Sample rate in Hz (default: 50 MHz).",
    )
    parser.add_argument(
        "--center-freq",
        type=float,
        default=2.4e9,
        help="Center frequency in Hz (default: 2.4 GHz).",
    )

    # --- Decoder ---
    parser.add_argument(
        "--decoder-path",
        type=Path,
        default=None,
        help="Root path of the external decoder repository.",
    )
    parser.add_argument(
        "--decoder-backend",
        type=str,
        default=None,
        help="Explicit decoder script/binary path (absolute, or relative "
             "to --decoder-path). Auto-discovered if omitted.",
    )
    parser.add_argument(
        "--decoder-args",
        type=str,
        default="",
        help="Extra arguments passed to the decoder (quoted string).",
    )
    parser.add_argument(
        "--decoder-timeout",
        type=float,
        default=60.0,
        help="Decoder subprocess timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary decoder I/O files for debugging.",
    )

    # --- Output ---
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON file for decoded frames.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> list[DecodeResult]:
    """Execute the full pipeline with the given arguments.

    Returns
    -------
    list[DecodeResult]
        One DecodeResult per burst that was sent to the decoder.
    """
    # --- Stage 1: Load IQ data ---
    if args.csv_mode:
        logger.info("=== Stage 1: Loading CSV data (Case B) ===")
        iq, classification = convert_csv_to_iq(args.input)
        metadata = {
            "sample_rate_hz": args.sample_rate,
            "center_freq_hz": args.center_freq,
        }
        logger.warning(
            "CSV input classified as CASE_B — decoding results may be unreliable"
        )
    else:
        logger.info("=== Stage 1: Loading verified IQ data (Case A) ===")
        iq, classification, metadata = load_verified_iq(
            args.input,
            sample_rate_hz=args.sample_rate,
            center_freq_hz=args.center_freq,
            fmt=args.fmt,
        )

    sample_rate = metadata["sample_rate_hz"]
    center_freq = metadata["center_freq_hz"]
    logger.info("Loaded %d samples, classification=%s", len(iq), classification.value)

    # --- Stage 2: Burst detection ---
    logger.info("=== Stage 2: Burst detection ===")
    detector = BurstDetector()
    bursts = detector.detect(iq, sample_rate_hz=sample_rate)
    logger.info("Detected %d burst(s)", len(bursts))

    if not bursts:
        logger.warning("No bursts detected — nothing to decode")
        return []

    # --- Stage 3: ZC correlation ---
    logger.info("=== Stage 3: Zadoff-Chu correlation ===")
    correlator = ZadoffChuCorrelator()
    zc_ref = correlator.generate_reference()
    logger.info("Generated ZC reference sequence (length=%d)", len(zc_ref))
    # TODO: narrow burst boundaries with ZC correlation

    # --- Stage 4: Decoding ---
    logger.info("=== Stage 4: Decoding ===")
    decode_results = _run_decode_stage(
        iq=iq,
        bursts=bursts,
        classification=classification,
        sample_rate_hz=sample_rate,
        center_freq_hz=center_freq,
        args=args,
    )

    # --- Stage 5: Summary ---
    logger.info("=== Stage 5: Results ===")
    total_frames = sum(r.num_frames for r in decode_results)
    succeeded = sum(1 for r in decode_results if r.success)
    logger.info(
        "Pipeline complete — %d burst(s) processed, %d succeeded, %d frame(s) decoded",
        len(decode_results), succeeded, total_frames,
    )

    for r in decode_results:
        logger.info("  %s", r.summary())

    return decode_results


def _run_decode_stage(
    iq: np.ndarray,
    bursts: list[BurstSegment],
    classification: InputClassification,
    sample_rate_hz: float,
    center_freq_hz: float,
    args: argparse.Namespace,
) -> list[DecodeResult]:
    """Run Stage 4: extract each burst and pass to the decoder.

    Parameters
    ----------
    iq : np.ndarray
        Full IQ capture array.
    bursts : list[BurstSegment]
        Detected burst segments from Stage 2.
    classification : InputClassification
        Input data classification (Case A / Case B).
    sample_rate_hz : float
        Sample rate.
    center_freq_hz : float
        Center frequency.
    args : argparse.Namespace
        CLI arguments (decoder config).

    Returns
    -------
    list[DecodeResult]
    """
    # Gate: only Case A data should be decoded
    if classification == InputClassification.CASE_B:
        logger.warning(
            "Input is CASE_B (exploratory) — skipping decode stage. "
            "Only verified IQ captures (Case A) are sent to the decoder."
        )
        return [
            DecodeResult(
                burst_index=i,
                backend_name="skipped",
                success=False,
                error_message="Decode skipped: input classified as CASE_B",
            )
            for i in range(len(bursts))
        ]

    if not args.decoder_path:
        logger.warning("No --decoder-path specified — skipping decode stage")
        return [
            DecodeResult(
                burst_index=i,
                backend_name="none",
                success=False,
                error_message="No decoder configured (use --decoder-path)",
            )
            for i in range(len(bursts))
        ]

    # Initialize decoder
    decoder = ReferenceDecoder(
        decoder_path=args.decoder_path,
        backend=args.decoder_backend,
        extra_args=args.decoder_args,
        timeout_s=args.decoder_timeout,
        keep_temp=args.keep_temp,
    )

    if not decoder.is_available():
        logger.error("Decoder not available — aborting decode stage")
        return [
            DecodeResult(
                burst_index=i,
                backend_name="unavailable",
                success=False,
                error_message=f"Decoder not found at {args.decoder_path}",
            )
            for i in range(len(bursts))
        ]

    # Decode each burst and parse telemetry from stdout
    parser = DroneIDParser(input_classification=classification)
    results: list[DecodeResult] = []

    for i, burst in enumerate(bursts):
        logger.info(
            "--- Burst %d/%d: samples %d–%d (%.3f ms) ---",
            i + 1, len(bursts),
            burst.start_sample, burst.end_sample,
            burst.duration_s * 1e3,
        )

        # Extract burst IQ segment from the full capture
        iq_burst = iq[burst.start_sample : burst.end_sample]

        result = decoder.decode_burst(
            iq_burst=iq_burst,
            sample_rate_hz=sample_rate_hz,
            center_freq_hz=center_freq_hz,
            burst_index=i,
        )

        # Parse telemetry from decoder stdout
        if result.success and result.stdout:
            frames = parser.parse_stdout(result.stdout)
            result.frames = frames

        results.append(result)

    return results


def _serialize_results(results: list[DecodeResult]) -> list[dict]:
    """Convert DecodeResults to JSON-serializable dicts."""
    output: list[dict] = []
    for r in results:
        entry: dict = {
            "burst_index": r.burst_index,
            "backend": r.backend_name,
            "success": r.success,
            "exit_code": r.exit_code,
            "duration_s": round(r.duration_s, 3),
            "command": r.command,
            "error_message": r.error_message,
            "num_frames": r.num_frames,
            "artifacts": [str(p) for p in r.artifact_paths],
        }

        if r.stdout.strip():
            entry["stdout"] = r.stdout.strip()
        if r.stderr.strip():
            entry["stderr"] = r.stderr.strip()

        # Serialize parsed frames
        frames_out: list[dict] = []
        for f in r.frames:
            frame_dict: dict = {
                "serial_number": f.serial_number,
                "manufacturer": f.manufacturer,
                "classification": f.input_classification.value,
                "decode_confidence": f.decode_confidence,
            }
            if f.drone_position:
                frame_dict["drone_lat"] = f.drone_position.latitude
                frame_dict["drone_lon"] = f.drone_position.longitude
                frame_dict["drone_alt"] = f.drone_position.altitude_m
            if f.pilot_position:
                frame_dict["pilot_lat"] = f.pilot_position.latitude
                frame_dict["pilot_lon"] = f.pilot_position.longitude
            if f.home_position:
                frame_dict["home_lat"] = f.home_position.latitude
                frame_dict["home_lon"] = f.home_position.longitude
            if f.speed_horizontal_ms is not None:
                frame_dict["speed_horizontal_ms"] = round(f.speed_horizontal_ms, 2)
            if f.speed_vertical_ms is not None:
                frame_dict["speed_vertical_ms"] = round(f.speed_vertical_ms, 2)
            if f.heading_deg is not None:
                frame_dict["heading_deg"] = round(f.heading_deg, 1)
            if f.height_agl_m is not None:
                frame_dict["height_agl_m"] = round(f.height_agl_m, 2)
            if f.timestamp is not None:
                frame_dict["timestamp"] = f.timestamp.isoformat()
            frames_out.append(frame_dict)
        entry["frames"] = frames_out

        output.append(entry)
    return output


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    logger.info("UAV Telemetry Pipeline starting")
    logger.info("Input: %s", args.input)

    results = run_pipeline(args)

    if args.output:
        output_data = _serialize_results(results)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_data, indent=2))
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()

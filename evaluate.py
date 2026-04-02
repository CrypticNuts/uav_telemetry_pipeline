"""
evaluate.py — Evaluation and reporting for the UAV telemetry pipeline.

Reads one or more pipeline output JSON files (produced by pipeline.py -o)
and generates summary statistics suitable for the Evaluation & Results
section of a final-year project report.

Output formats:
  - Terminal table (human-readable)
  - JSON summary  (machine-readable, for further analysis)
  - CSV summary   (spreadsheet-ready, for tables/charts in the report)

Usage:
    python evaluate.py data/results/mavic_full_parse.json
    python evaluate.py data/results/*.json -o data/results/summary
    python evaluate.py data/results/*.json --format all -o data/results/summary
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ======================================================================
# Per-file metrics
# ======================================================================

@dataclass
class FileReport:
    """Aggregated metrics for one pipeline output file."""

    file_name: str
    source_case: str                          # "verified_iq", "exploratory", or "unknown"

    # Counts
    bursts_detected: int = 0
    decode_attempts: int = 0
    decode_successes: int = 0
    decode_failures: int = 0
    frames_parsed: int = 0
    crc_valid_frames: int = 0
    plausible_coordinates: int = 0

    # Timing
    total_decode_time_s: float = 0.0
    avg_decode_time_s: float = 0.0

    # Failure breakdown
    failure_reasons: dict[str, int] = field(default_factory=dict)

    # Unique identifiers seen
    serial_numbers: list[str] = field(default_factory=list)
    device_types: list[str] = field(default_factory=list)


def analyse_file(path: Path) -> FileReport:
    """Analyse a single pipeline output JSON file.

    Parameters
    ----------
    path : Path
        Path to a JSON file produced by ``pipeline.py -o``.

    Returns
    -------
    FileReport
    """
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        data = [data]

    report = FileReport(file_name=path.name, source_case="unknown")

    decode_times: list[float] = []
    serials: set[str] = set()
    devices: set[str] = set()

    for burst in data:
        report.bursts_detected += 1

        # Decode outcome
        success = burst.get("success", False)
        report.decode_attempts += 1
        if success:
            report.decode_successes += 1
        else:
            report.decode_failures += 1
            reason = burst.get("error_message") or f"exit_code={burst.get('exit_code')}"
            report.failure_reasons[reason] = report.failure_reasons.get(reason, 0) + 1

        # Timing
        dt = burst.get("duration_s", 0.0)
        if dt > 0:
            decode_times.append(dt)
        report.total_decode_time_s += dt

        # Frames
        frames = burst.get("frames", [])
        report.frames_parsed += len(frames)

        for frame in frames:
            # Source case (take from first frame seen)
            classification = frame.get("classification", "unknown")
            if report.source_case == "unknown":
                report.source_case = classification

            # CRC
            confidence = frame.get("decode_confidence")
            if confidence is not None and confidence >= 1.0:
                report.crc_valid_frames += 1

            # Plausible coordinates: drone_lat/lon present and non-zero
            d_lat = frame.get("drone_lat")
            d_lon = frame.get("drone_lon")
            if (
                d_lat is not None
                and d_lon is not None
                and not (d_lat == 0.0 and d_lon == 0.0)
                and -90 <= d_lat <= 90
                and -180 <= d_lon <= 180
            ):
                report.plausible_coordinates += 1

            # Identity
            sn = frame.get("serial_number", "")
            if sn:
                serials.add(sn)
            mfr = frame.get("manufacturer", "")
            if mfr:
                devices.add(mfr)

    report.serial_numbers = sorted(serials)
    report.device_types = sorted(devices)

    if decode_times:
        report.avg_decode_time_s = round(sum(decode_times) / len(decode_times), 3)
    report.total_decode_time_s = round(report.total_decode_time_s, 3)

    return report


# ======================================================================
# Aggregate across files
# ======================================================================

def aggregate(reports: list[FileReport]) -> FileReport:
    """Compute totals across multiple file reports."""
    agg = FileReport(file_name="TOTAL", source_case="mixed")

    all_times: list[float] = []
    all_serials: set[str] = set()
    all_devices: set[str] = set()
    cases: set[str] = set()

    for r in reports:
        agg.bursts_detected += r.bursts_detected
        agg.decode_attempts += r.decode_attempts
        agg.decode_successes += r.decode_successes
        agg.decode_failures += r.decode_failures
        agg.frames_parsed += r.frames_parsed
        agg.crc_valid_frames += r.crc_valid_frames
        agg.plausible_coordinates += r.plausible_coordinates
        agg.total_decode_time_s += r.total_decode_time_s

        if r.avg_decode_time_s > 0:
            all_times.append(r.avg_decode_time_s)

        for reason, count in r.failure_reasons.items():
            agg.failure_reasons[reason] = agg.failure_reasons.get(reason, 0) + count

        all_serials.update(r.serial_numbers)
        all_devices.update(r.device_types)
        cases.add(r.source_case)

    agg.serial_numbers = sorted(all_serials)
    agg.device_types = sorted(all_devices)
    agg.total_decode_time_s = round(agg.total_decode_time_s, 3)
    agg.source_case = cases.pop() if len(cases) == 1 else "mixed"

    if all_times:
        agg.avg_decode_time_s = round(sum(all_times) / len(all_times), 3)

    return agg


# ======================================================================
# Output formatters
# ======================================================================

def print_terminal(reports: list[FileReport], agg: FileReport) -> None:
    """Print a human-readable summary table to stdout."""

    # Header
    print()
    print("=" * 90)
    print("  UAV Telemetry Pipeline — Evaluation Report")
    print("=" * 90)
    print()

    # Column definitions
    headers = [
        "File", "Case", "Bursts", "Attempts", "Success",
        "Frames", "CRC OK", "Coords", "Avg Time",
    ]
    col_w = [28, 12, 7, 9, 8, 7, 7, 7, 10]

    # Header row
    header_line = ""
    for h, w in zip(headers, col_w):
        header_line += h.ljust(w)
    print(header_line)
    print("-" * sum(col_w))

    # Data rows
    all_rows = reports + [agg]
    for r in all_rows:
        if r.file_name == "TOTAL":
            print("-" * sum(col_w))

        name = r.file_name
        if len(name) > col_w[0] - 2:
            name = name[: col_w[0] - 5] + "..."

        vals = [
            name,
            r.source_case[:col_w[1] - 1],
            str(r.bursts_detected),
            str(r.decode_attempts),
            str(r.decode_successes),
            str(r.frames_parsed),
            str(r.crc_valid_frames),
            str(r.plausible_coordinates),
            f"{r.avg_decode_time_s:.2f}s" if r.avg_decode_time_s > 0 else "-",
        ]
        row = ""
        for v, w in zip(vals, col_w):
            row += v.ljust(w)
        print(row)

    print()

    # Decode success rate
    if agg.decode_attempts > 0:
        rate = agg.decode_successes / agg.decode_attempts * 100
        print(f"  Decode success rate:  {agg.decode_successes}/{agg.decode_attempts} ({rate:.1f}%)")
    # Frame yield
    if agg.decode_successes > 0:
        yield_rate = agg.frames_parsed / agg.decode_successes
        print(f"  Frames per success:   {yield_rate:.1f}")
    # CRC rate
    if agg.frames_parsed > 0:
        crc_rate = agg.crc_valid_frames / agg.frames_parsed * 100
        print(f"  CRC-valid frames:     {agg.crc_valid_frames}/{agg.frames_parsed} ({crc_rate:.1f}%)")
    # Coordinate rate
    if agg.frames_parsed > 0:
        coord_rate = agg.plausible_coordinates / agg.frames_parsed * 100
        print(f"  Plausible coords:     {agg.plausible_coordinates}/{agg.frames_parsed} ({coord_rate:.1f}%)")

    # Timing
    print(f"  Total decode time:    {agg.total_decode_time_s:.2f}s")

    # Identifiers
    if agg.serial_numbers:
        print(f"  Unique serials:       {', '.join(agg.serial_numbers)}")
    if agg.device_types:
        print(f"  Device types:         {', '.join(agg.device_types)}")

    # Failure reasons
    if agg.failure_reasons:
        print()
        print("  Failure reasons:")
        for reason, count in sorted(agg.failure_reasons.items(), key=lambda x: -x[1]):
            # Truncate very long reasons
            short = reason[:70] + "..." if len(reason) > 70 else reason
            print(f"    [{count}x] {short}")

    print()
    print("=" * 90)
    print()


def write_json(reports: list[FileReport], agg: FileReport, path: Path) -> None:
    """Write the full report as a JSON file."""
    output = {
        "files": [_report_to_dict(r) for r in reports],
        "aggregate": _report_to_dict(agg),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    logger.info("JSON report written to %s", path)


def write_csv(reports: list[FileReport], agg: FileReport, path: Path) -> None:
    """Write the summary table as a CSV file."""
    fieldnames = [
        "file_name", "source_case",
        "bursts_detected", "decode_attempts", "decode_successes",
        "decode_failures", "frames_parsed", "crc_valid_frames",
        "plausible_coordinates", "avg_decode_time_s", "total_decode_time_s",
        "serial_numbers", "device_types",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reports + [agg]:
            row = {
                "file_name": r.file_name,
                "source_case": r.source_case,
                "bursts_detected": r.bursts_detected,
                "decode_attempts": r.decode_attempts,
                "decode_successes": r.decode_successes,
                "decode_failures": r.decode_failures,
                "frames_parsed": r.frames_parsed,
                "crc_valid_frames": r.crc_valid_frames,
                "plausible_coordinates": r.plausible_coordinates,
                "avg_decode_time_s": r.avg_decode_time_s,
                "total_decode_time_s": r.total_decode_time_s,
                "serial_numbers": "; ".join(r.serial_numbers),
                "device_types": "; ".join(r.device_types),
            }
            writer.writerow(row)
    logger.info("CSV report written to %s", path)


def _report_to_dict(r: FileReport) -> dict:
    """Convert a FileReport to a JSON-serializable dict."""
    d = asdict(r)
    return d


# ======================================================================
# CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UAV Telemetry Pipeline — Evaluation Report Generator",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Pipeline output JSON file(s) to analyse.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path prefix (without extension). Generates .json and/or .csv.",
    )
    parser.add_argument(
        "--format",
        choices=["terminal", "json", "csv", "all"],
        default="all",
        help="Output format (default: all).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate inputs
    valid_paths: list[Path] = []
    for p in args.inputs:
        if not p.exists():
            logger.warning("File not found, skipping: %s", p)
            continue
        if not p.suffix == ".json":
            logger.warning("Not a JSON file, skipping: %s", p)
            continue
        valid_paths.append(p)

    if not valid_paths:
        logger.error("No valid input files found.")
        sys.exit(1)

    # Analyse
    reports: list[FileReport] = []
    for p in sorted(valid_paths):
        try:
            report = analyse_file(p)
            reports.append(report)
            logger.debug("Analysed %s: %d frames", p.name, report.frames_parsed)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse %s: %s", p.name, exc)

    if not reports:
        logger.error("No files could be analysed.")
        sys.exit(1)

    agg = aggregate(reports)

    # Output
    fmt = args.format
    if fmt in ("terminal", "all"):
        print_terminal(reports, agg)

    if args.output:
        prefix = Path(args.output)
        if fmt in ("json", "all"):
            write_json(reports, agg, prefix.with_suffix(".json"))
        if fmt in ("csv", "all"):
            write_csv(reports, agg, prefix.with_suffix(".csv"))
    elif fmt in ("json", "csv"):
        logger.warning("--output is required for %s format; use -o <prefix>", fmt)


if __name__ == "__main__":
    main()

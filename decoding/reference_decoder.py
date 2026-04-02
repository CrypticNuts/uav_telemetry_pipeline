"""
reference_decoder.py — Subprocess-based adapter for external DroneID decoders.

Wraps any command-line decoder tool (e.g., from RUB-SysSec/DroneSecurity or
anarkiwi/samples2djidroneid) into the pipeline's ReceiverAdapter interface.

Workflow:
    1. Export the burst IQ segment to a temp .fc32 file
    2. Build a command line from the configured entrypoint + user args
    3. Run the decoder as a subprocess with a timeout
    4. Capture stdout, stderr, exit code
    5. Scan the working directory for output artifacts
    6. Return a structured DecodeResult

This module does NOT contain any decoding logic — it is purely a
subprocess orchestrator.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from decoding.receiver_adapter import ReceiverAdapter
from telemetry.models import DecodeResult

logger = logging.getLogger(__name__)

# Scripts we look for when auto-discovering the decoder entrypoint.
# Ordered by priority — first match wins.
_KNOWN_ENTRYPOINTS = [
    "droneid_receiver_offline.py",
    "decode_droneid.py",
    "droneid_receiver.py",
    "receiver.py",
    "main.py",
]

# File extensions we consider decoder output artifacts.
_ARTIFACT_EXTENSIONS = {".json", ".csv", ".txt", ".log", ".bin", ".out"}


class ReferenceDecoder(ReceiverAdapter):
    """Subprocess adapter for an external DroneID decoder.

    Parameters
    ----------
    decoder_path : Path
        Root directory of the decoder repository (e.g., the cloned
        DroneSecurity repo). Used for entrypoint discovery and as
        the subprocess working directory.
    backend : str | None
        Explicit path to the decoder script or binary, relative to
        decoder_path or absolute. If None, the adapter auto-discovers
        the entrypoint by scanning for known script names.
    extra_args : str
        Additional arguments appended to every decoder invocation.
        Passed as a single string and split with shlex.
    timeout_s : float
        Maximum time in seconds to wait for the decoder subprocess.
    keep_temp : bool
        If True, do not delete temp files after decoding. Useful for
        debugging.
    """

    def __init__(
        self,
        decoder_path: Path,
        backend: str | None = None,
        extra_args: str = "",
        timeout_s: float = 60.0,
        keep_temp: bool = False,
    ) -> None:
        self.decoder_path = Path(decoder_path).resolve()
        self._backend_override = backend
        self.extra_args = extra_args
        self.timeout_s = timeout_s
        self.keep_temp = keep_temp

        self._entrypoint: Path | None = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if the decoder is installed and an entrypoint can be found."""
        if not self.decoder_path.exists():
            logger.warning(
                "Decoder path does not exist: %s", self.decoder_path
            )
            return False

        entrypoint = self._resolve_entrypoint()
        if entrypoint is None:
            logger.warning(
                "No decoder entrypoint found in %s. "
                "Searched for: %s. "
                "Use --decoder-backend to specify one explicitly.",
                self.decoder_path,
                ", ".join(_KNOWN_ENTRYPOINTS),
            )
            return False

        self._entrypoint = entrypoint
        logger.info("Decoder entrypoint: %s", self._entrypoint)
        return True

    # ------------------------------------------------------------------
    # Decode from in-memory burst
    # ------------------------------------------------------------------

    def decode_burst(
        self,
        iq_burst: np.ndarray,
        sample_rate_hz: float,
        center_freq_hz: float,
        burst_index: int = 0,
    ) -> DecodeResult:
        """Export burst to a temp file and invoke the decoder.

        Parameters
        ----------
        iq_burst : np.ndarray
            Complex64 IQ samples for one detected burst.
        sample_rate_hz : float
            Sample rate in Hz.
        center_freq_hz : float
            Center frequency in Hz.
        burst_index : int
            Index of this burst (for logging and result tagging).

        Returns
        -------
        DecodeResult
        """
        backend_name = self._entrypoint.name if self._entrypoint else "unknown"
        logger.info(
            "Decoding burst %d: %d samples (%.3f ms) via %s",
            burst_index, iq_burst.size,
            iq_burst.size / sample_rate_hz * 1e3,
            backend_name,
        )

        # Create temp directory and export the burst
        tmp_dir = self.create_temp_dir(prefix=f"burst{burst_index}_")
        burst_file = tmp_dir / f"burst_{burst_index}.fc32"
        self.export_burst_fc32(iq_burst, burst_file)

        # Run the decoder (use entrypoint's parent as cwd so local
        # imports like SpectrumCapture, Packet, etc. resolve correctly)
        entrypoint = self._resolve_entrypoint()
        work_dir = entrypoint.parent if entrypoint else tmp_dir
        result = self._run_decoder(
            input_file=burst_file,
            sample_rate_hz=sample_rate_hz,
            center_freq_hz=center_freq_hz,
            burst_index=burst_index,
            work_dir=work_dir,
        )

        # Cleanup unless debugging
        if not self.keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.debug("Cleaned up temp dir: %s", tmp_dir)
        else:
            logger.info("Temp files kept at: %s", tmp_dir)

        return result

    # ------------------------------------------------------------------
    # Decode from file on disk
    # ------------------------------------------------------------------

    def decode_file(
        self,
        file_path: Path,
        sample_rate_hz: float,
        center_freq_hz: float,
    ) -> DecodeResult:
        """Invoke the decoder directly on an existing capture file.

        Parameters
        ----------
        file_path : Path
            Path to the IQ capture file.
        sample_rate_hz : float
            Sample rate in Hz.
        center_freq_hz : float
            Center frequency in Hz.

        Returns
        -------
        DecodeResult
        """
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            return DecodeResult(
                burst_index=0,
                backend_name=self._entrypoint.name if self._entrypoint else "unknown",
                success=False,
                error_message=f"Input file not found: {file_path}",
            )

        logger.info("Decoding file: %s", file_path)
        return self._run_decoder(
            input_file=file_path,
            sample_rate_hz=sample_rate_hz,
            center_freq_hz=center_freq_hz,
            burst_index=0,
            work_dir=file_path.parent,
        )

    # ------------------------------------------------------------------
    # Core subprocess runner
    # ------------------------------------------------------------------

    def _run_decoder(
        self,
        input_file: Path,
        sample_rate_hz: float,
        center_freq_hz: float,
        burst_index: int,
        work_dir: Path,
    ) -> DecodeResult:
        """Build and execute the decoder command, capture all output.

        Parameters
        ----------
        input_file : Path
            The IQ file to decode.
        sample_rate_hz : float
            Sample rate in Hz.
        center_freq_hz : float
            Center frequency in Hz.
        burst_index : int
            Burst index for result tagging.
        work_dir : Path
            Working directory for the subprocess.

        Returns
        -------
        DecodeResult
        """
        entrypoint = self._resolve_entrypoint()
        backend_name = entrypoint.name if entrypoint else "unknown"

        if entrypoint is None:
            return DecodeResult(
                burst_index=burst_index,
                backend_name=backend_name,
                success=False,
                error_message="No decoder entrypoint found. Use --decoder-backend.",
            )

        # Build command
        cmd = self._build_command(
            entrypoint=entrypoint,
            input_file=input_file,
            sample_rate_hz=sample_rate_hz,
            center_freq_hz=center_freq_hz,
        )
        cmd_str = " ".join(str(c) for c in cmd)
        logger.info("Running: %s", cmd_str)
        logger.debug("Working directory: %s", work_dir)

        # Execute
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            duration = time.monotonic() - t0

            # Log output
            if proc.stdout.strip():
                logger.debug("STDOUT:\n%s", proc.stdout.strip())
            if proc.stderr.strip():
                logger.debug("STDERR:\n%s", proc.stderr.strip())

            success = proc.returncode == 0

            if not success:
                logger.warning(
                    "Decoder exited with code %d for burst %d",
                    proc.returncode, burst_index,
                )

            # Scan for output artifacts
            artifacts = self._find_artifacts(work_dir, exclude={input_file})

            result = DecodeResult(
                burst_index=burst_index,
                backend_name=backend_name,
                success=success,
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                command=cmd_str,
                duration_s=duration,
                input_file=input_file,
                artifact_paths=artifacts,
            )

        except subprocess.TimeoutExpired:
            duration = time.monotonic() - t0
            logger.error(
                "Decoder timed out after %.1fs for burst %d",
                self.timeout_s, burst_index,
            )
            result = DecodeResult(
                burst_index=burst_index,
                backend_name=backend_name,
                success=False,
                command=cmd_str,
                duration_s=duration,
                input_file=input_file,
                error_message=f"Timeout after {self.timeout_s:.0f}s",
            )

        except FileNotFoundError as exc:
            logger.error("Failed to execute decoder: %s", exc)
            result = DecodeResult(
                burst_index=burst_index,
                backend_name=backend_name,
                success=False,
                command=cmd_str,
                input_file=input_file,
                error_message=f"Decoder executable not found: {exc}",
            )

        except OSError as exc:
            logger.error("OS error running decoder: %s", exc)
            result = DecodeResult(
                burst_index=burst_index,
                backend_name=backend_name,
                success=False,
                command=cmd_str,
                input_file=input_file,
                error_message=f"OS error: {exc}",
            )

        logger.info("Decode result: %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _build_command(
        self,
        entrypoint: Path,
        input_file: Path,
        sample_rate_hz: float,
        center_freq_hz: float,
    ) -> list[str]:
        """Build the subprocess command list.

        Detects the DroneSecurity CLI convention (``-i FILE -s RATE``)
        by checking the entrypoint filename.  Falls back to a generic
        positional-argument style for other backends.

        Subclass and override this method to customize for a specific
        decoder's CLI interface.
        """
        # DroneSecurity-style CLI: -i <file> -s <rate> [-f]
        if entrypoint.name in (
            "droneid_receiver_offline.py",
            "droneid_receiver_live.py",
        ):
            cmd: list[str] = [
                "python3", str(entrypoint),
                "-i", str(input_file),
                "-s", str(sample_rate_hz),
            ]
        else:
            # Generic: positional input file
            cmd = ["python3", str(entrypoint), str(input_file)]

        # Append extra user-supplied arguments
        if self.extra_args:
            cmd.extend(shlex.split(self.extra_args))

        return cmd

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_entrypoint(self) -> Path | None:
        """Find the decoder entrypoint script.

        Priority: explicit --decoder-backend > auto-discovery.

        Returns
        -------
        Path | None
            Absolute path to the entrypoint, or None if not found.
        """
        if self._backend_override:
            candidate = Path(self._backend_override)
            # Try as absolute path first
            if candidate.is_absolute() and candidate.is_file():
                return candidate
            # Try relative to decoder_path
            relative = self.decoder_path / candidate
            if relative.is_file():
                return relative.resolve()
            logger.warning(
                "Explicit backend '%s' not found (tried absolute and "
                "relative to %s)",
                self._backend_override, self.decoder_path,
            )
            return None

        # Auto-discovery: walk known entrypoint names
        for name in _KNOWN_ENTRYPOINTS:
            # Search up to 2 levels deep
            for pattern in [name, f"*/{name}", f"*/*/{name}"]:
                matches = list(self.decoder_path.glob(pattern))
                if matches:
                    return matches[0].resolve()

        return None

    @staticmethod
    def _find_artifacts(
        directory: Path,
        exclude: set[Path] | None = None,
    ) -> list[Path]:
        """Scan a directory for output files produced by the decoder.

        Parameters
        ----------
        directory : Path
            Directory to scan.
        exclude : set[Path] | None
            Paths to exclude (e.g., the input file we created).

        Returns
        -------
        list[Path]
            Artifact file paths, sorted by name.
        """
        exclude = exclude or set()
        artifacts: list[Path] = []
        for f in directory.iterdir():
            if f.is_file() and f.suffix in _ARTIFACT_EXTENSIONS and f not in exclude:
                artifacts.append(f)
        artifacts.sort(key=lambda p: p.name)
        if artifacts:
            logger.info(
                "Found %d artifact(s): %s",
                len(artifacts),
                ", ".join(a.name for a in artifacts),
            )
        return artifacts

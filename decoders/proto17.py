"""
proto17.py — Fallback decoder via proto17/dji_droneid Octave scripts.

The proto17 toolkit is a collection of Matlab/Octave scripts that detect
ZC bursts but do **not** produce a finished telemetry payload — they are
primarily a sync/extraction tool. Without a turbo decoder it cannot yield
the structured DroneID payload that DroneSecurity does.

Therefore this fallback is implemented as a **lightweight burst-counter**:
it runs ``find_zc.m`` over the capture (downsampled to 15.36 Msps as that
script requires) and reports how many burst candidates it found. Each
candidate is emitted as a TelemetryFrame with ``crc_ok=False`` and the
sample index in ``timestamp_sample`` so the runner records that proto17
saw signal without claiming it decoded telemetry.

If Octave is not installed this decoder reports unavailable and returns
``[]`` silently — per the brief.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..config import PipelineConfig
from ..preprocessor.shift_and_filter import shift_and_filter
from .base import BaseDecoder, TelemetryFrame

logger = logging.getLogger(__name__)

# proto17/find_zc.m requires fs as a multiple of 15 kHz (it uses
# fft_size = fs/15e3 as an integer array index). 15.36 Msps = 1024×15 kHz
# is the LTE base rate the script was designed for. From 50 Msps no
# integer decimation reaches it, so shift_and_filter does the leftover
# polyphase pass internally.
_PROTO17_TARGET_FS_HZ = 15.36e6
_PROTO17_KEEP_BW_HZ = 12e6              # >= DroneID 10 MHz + margin
_PROTO17_FS_TOLERANCE_HZ = 1e3


class Proto17Decoder(BaseDecoder):
    """Octave-driven ZC burst counter.

    Parameters
    ----------
    config : PipelineConfig | None
        Resolved configuration. Used to locate ``matlab/find_zc.m``.
    timeout_s : float
        Wall-clock timeout for the Octave invocation.
    """

    name = "proto17"

    def __init__(
        self,
        config: PipelineConfig | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.config = config or PipelineConfig.load()
        self.timeout_s = float(timeout_s)
        self._octave: str | None = shutil.which("octave")

    def is_available(self) -> bool:
        if self._octave is None:
            logger.info("proto17 fallback unavailable: 'octave' not on PATH")
            return False
        matlab_dir = self.config.proto17_matlab_dir
        if not (matlab_dir / "find_zc.m").is_file():
            logger.info(
                "proto17 fallback unavailable: find_zc.m not found in %s",
                matlab_dir,
            )
            return False
        return True

    def decode(self, iq_file_path: str, sample_rate: float) -> list[TelemetryFrame]:
        if not self.is_available():
            return []

        iq_path = Path(iq_file_path).expanduser().resolve()
        if not iq_path.is_file():
            logger.error("IQ file not found: %s", iq_path)
            return []

        matlab_dir = self.config.proto17_matlab_dir
        with tempfile.TemporaryDirectory(prefix="proto17_") as tmp:
            tmp_dir = Path(tmp)
            # Pre-decimate in Python BEFORE handing the file to Octave.
            # Octave reads the whole file as float64 (16 B/sample); a
            # 2.6 GB input becomes >5 GB in RAM and the OS kills it. By
            # decimating to ~16.67 Msps first (q=3 from 50 Msps), the
            # working set drops below 2 GB and Octave's internal
            # resample step has almost nothing left to do.
            iq_for_octave, fs_for_octave = self._pre_decimate(
                iq_path, sample_rate, tmp_dir,
            )

            script = tmp_dir / "run.m"
            script.write_text(
                self._octave_script(iq_for_octave, fs_for_octave, matlab_dir)
            )

            cmd = [self._octave or "octave", "--no-gui", "--quiet", str(script)]
            logger.info(
                "Running proto17 (fs=%.3f MHz): %s",
                fs_for_octave / 1e6, " ".join(cmd),
            )
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired:
                logger.error("proto17 timed out after %.0fs", self.timeout_s)
                return []
            except OSError as exc:
                logger.error("Failed to launch octave: %s", exc)
                return []

        if proc.returncode != 0:
            logger.warning(
                "octave exited %d; stderr tail: %s",
                proc.returncode,
                proc.stderr.strip().splitlines()[-3:] if proc.stderr else "(empty)",
            )

        frames = self._parse_indices(proc.stdout)
        # Indices reported by Octave are at the decimated rate. Map them
        # back to input-rate sample indices so all downstream tooling
        # uses one consistent sample axis.
        if frames and fs_for_octave < sample_rate - _PROTO17_FS_TOLERANCE_HZ:
            ratio = sample_rate / fs_for_octave
            for f in frames:
                f["timestamp_sample"] = int(f["timestamp_sample"] * ratio)
        return frames

    def _pre_decimate(
        self,
        iq_path: Path,
        sample_rate: float,
        tmp_dir: Path,
    ) -> tuple[Path, float]:
        """Return (file_path, fs) for Octave: cleaned copy or original.

        Asks ``shift_and_filter`` for the proto17 target rate exactly
        (15.36 Msps = 1024 × 15 kHz). If the input is already there,
        skip the pass; otherwise the preprocessor does integer decimate
        + a polyphase remainder so the output rate matches exactly.
        """
        if abs(sample_rate - _PROTO17_TARGET_FS_HZ) < _PROTO17_FS_TOLERANCE_HZ:
            return iq_path, sample_rate

        if sample_rate < _PROTO17_TARGET_FS_HZ:
            # Below target: skip (we'd be upsampling, not the proto17
            # use case).
            return iq_path, sample_rate

        decimated_path = tmp_dir / "decimated.fc32"
        try:
            shift_and_filter(
                input_path=iq_path,
                output_path=decimated_path,
                sample_rate_hz=sample_rate,
                shift_hz=0.0,                  # no frequency shift, just LPF + resample
                keep_bandwidth_hz=_PROTO17_KEEP_BW_HZ,
                output_rate_hz=_PROTO17_TARGET_FS_HZ,
            )
        except Exception as exc:  # noqa: BLE001 - fall back to raw file
            logger.warning(
                "proto17 pre-decimation failed (%s); falling back to raw file",
                exc,
            )
            return iq_path, sample_rate

        in_mb = iq_path.stat().st_size / (1 << 20)
        out_mb = decimated_path.stat().st_size / (1 << 20)
        logger.info(
            "proto17 pre-decimated %.0f MB -> %.0f MB at fs=%.3f MHz "
            "(%.3f -> %.3f MHz)",
            in_mb, out_mb, _PROTO17_TARGET_FS_HZ / 1e6,
            sample_rate / 1e6, _PROTO17_TARGET_FS_HZ / 1e6,
        )
        return decimated_path, _PROTO17_TARGET_FS_HZ

    @staticmethod
    def _octave_script(iq_path: Path, sample_rate: float, matlab_dir: Path) -> str:
        # ``resample`` and (typically) ``find_zc`` both come from the
        # Octave signal-processing package, which has to be loaded
        # explicitly. Pre-decimation in Python brings us to ~16.67 MHz;
        # if we're already within 10% of the 15.36 MHz target we skip
        # the in-Octave resample altogether (find_zc.m tolerates a
        # small rate mismatch, and resample() of a 100M-sample vector
        # is expensive in Octave).
        # find_zc.m signature: ``scores = find_zc(samples, sample_rate)``.
        # It returns a *complex score vector* with one entry per starting
        # offset, not a list of peak indices. We take ``abs(scores).^2``,
        # then keep peaks above a fraction of the global maximum with a
        # minimum-separation equal to one ZC symbol (= fft_size, i.e.
        # sample_rate / 15 kHz). One ZC peak per real DroneID burst.
        target_fs = 15.36e6
        return (
            f"pkg load signal;\n"
            f"addpath('{matlab_dir}');\n"
            f"fid = fopen('{iq_path}', 'rb');\n"
            f"raw = fread(fid, 'float32');\n"
            f"fclose(fid);\n"
            f"iq = raw(1:2:end) + 1i*raw(2:2:end);\n"
            f"fs = {float(sample_rate)};\n"
            f"target_fs = {target_fs};\n"
            # If the input is far from the LTE base rate, downsample to it
            # first (proto17 expects ~15.36 Msps).
            f"if abs(fs - target_fs) / target_fs > 0.10;\n"
            f"  iq = resample(iq, round(target_fs), round(fs));\n"
            f"  fs = target_fs;\n"
            f"end;\n"
            # find_zc.m uses fft_size = sample_rate/15e3 as an array
            # index — so fs must be an integer multiple of 15 kHz.
            # Snap to the nearest multiple and apply the small (often
            # <0.01%) resample needed to land exactly.
            f"fs_snap = round(fs / 15e3) * 15e3;\n"
            f"if abs(fs - fs_snap) > 1;\n"
            f"  iq = resample(iq, round(fs_snap), round(fs));\n"
            f"  fs = fs_snap;\n"
            f"else;\n"
            f"  fs = fs_snap;\n"
            f"end;\n"
            f"try;\n"
            f"  scores = find_zc(iq, fs);\n"
            f"  mag = abs(scores).^2;\n"
            f"  peak_thr = 0.30 * max(mag);\n"
            f"  fft_size = round(fs / 15e3);\n"
            f"  min_sep = fft_size;\n"
            f"  if peak_thr > 0;\n"
            f"    [pks, locs] = findpeaks(mag, 'MinPeakHeight', peak_thr, "
            f"'MinPeakDistance', min_sep);\n"
            f"    for k = 1:length(locs);\n"
            f"      printf('ZC_PEAK %d %g\\n', locs(k), pks(k));\n"
            f"    end;\n"
            f"  end;\n"
            f"  printf('ZC_BEST %g\\n', max(mag));\n"
            f"catch err;\n"
            f"  fprintf(stderr, 'proto17 error: %s\\n', err.message);\n"
            f"  exit(2);\n"
            f"end;\n"
        )

    @staticmethod
    def _parse_indices(stdout: str) -> list[TelemetryFrame]:
        frames: list[TelemetryFrame] = []
        for line in stdout.splitlines():
            if not line.startswith("ZC_PEAK "):
                continue
            try:
                idx = int(line.split()[1])
            except (IndexError, ValueError):
                continue
            frames.append({
                "lat": 0.0,
                "lon": 0.0,
                "altitude_m": 0.0,
                "serial": "",
                "crc_ok": False,
                "decoder": "proto17",
                "timestamp_sample": idx,
            })
        logger.info("proto17 reported %d ZC candidate(s)", len(frames))
        return frames

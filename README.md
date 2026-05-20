# UAV Telemetry Pipeline

Passive, offline pipeline for recovering DJI DroneID telemetry from raw IQ
captures. Wraps the RUB-SysSec/DroneSecurity decoder, the proto17 Octave
toolkit, and an in-process Python decoder behind a single command-line
runner with a clean JSON output format.

Built as a final-year cybersecurity engineering project (PFE).

---

## Quick start

The pipeline expects the DroneSecurity repository to exist at
`~/projects/PFE/DroneSecurity` with its venv populated. Override with
`$DRONESECURITY_PATH` or `config.yaml` if needed.

```bash
# Activate the DroneSecurity venv (provides distutils shim for Python 3.12+)
source ~/projects/PFE/DroneSecurity/.venv/bin/activate

# End-to-end run on the reference sample
python -u uav_telemetry_pipeline/run_pipeline.py \
    --input ~/projects/PFE/DroneSecurity/samples/mini2_sm \
    --sample-rate 50e6

# Diagnose-only (skip decoders, report ZC correlation peaks)
python -u uav_telemetry_pipeline/run_pipeline.py \
    --input <capture.fc32> --sample-rate 50e6 --diagnose-only

# Descriptive band analysis (no decoding, no ZC matching)
python -u uav_telemetry_pipeline/run_pipeline.py \
    --input <capture.fc32> --sample-rate 50e6 --spectrum-only
```

On a known-good capture (e.g. `mini2_sm` at 50 Msps) the runner produces:

```
============================================================
  decoder used:  dronesecurity
  frames:        9
  CRC OK:        7
  CRC pass rate: 77.8%
  first fix:     lat=51.447176, lon=7.266528
  results:       results/mini2_sm_telemetry.json
============================================================
```

---

## Pipeline architecture

```
┌─────────────────────────────────────────────────────────────┐
│  run_pipeline.py                                            │
│  CLI orchestrator: tries decoders in priority order until   │
│  one returns ≥1 CRC-OK frame. Always writes a JSON result.  │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌──────────────┐ ┌─────────┐ ┌───────────┐    decoders/
│DroneSecurity │ │ Proto17 │ │  Native   │    (BaseDecoder
│  (primary)   │ │ (fallA) │ │ (fallB)   │     subclasses)
└──────────────┘ └─────────┘ └───────────┘
        │            │            │
        └────────────┴────────────┘
                     │
                     ▼  (no decoder produced CRC OK)
            ┌──────────────────┐
            │ZadoffChuCorrelat.│    detection/zc_correlator.py
            │ + spectrum bands │    (only via --diagnose / --spectrum)
            └──────────────────┘
```

---

## Decoders

All three implement the same `BaseDecoder` interface in
`decoders/base.py`:

```python
def decode(iq_file_path: str, sample_rate: float) -> list[TelemetryFrame]
```

They return a list of `TelemetryFrame` (TypedDict) — empty on failure,
never raising.

### 1. `DroneSecurityDecoder` — primary

- Invokes `DroneSecurity/src/droneid_receiver_offline.py` as a subprocess.
- Always uses the **DroneSecurity venv interpreter**
  (`.venv/bin/python`), never bare `python3`. `SpectrumCapture.py`
  imports `distutils.log`, which Python 3.12 removed from stdlib; the
  venv has `setuptools` installed which re-provides the shim.
- Parses stdout for the `## Drone-ID Payload ##` JSON blocks emitted by
  the receiver and maps them to the canonical `TelemetryFrame` shape.
- Default subprocess timeout: 30 minutes (configurable via
  `--decoder-timeout`).

### 2. `Proto17Decoder` — fallback A (Octave)

- Invokes `dji_droneid/matlab/find_zc.m` via the `octave` CLI on a
  resampled (15.36 Msps) copy of the input.
- The proto17 toolkit only finds ZC burst positions; it does **not**
  decode payload, so this fallback emits one entry per detected burst
  with `crc_ok=False` and `timestamp_sample=<idx>`. It is a signal-
  presence proof, not a decoder.
- Becomes unavailable (silently skipped) if the `octave` binary is not
  on `$PATH`.

### 3. `NativeDecoder` — fallback B (in-process Python)

- Imports `SpectrumCapture` / `Packet` / `qpsk.Decoder` /
  `DroneIDPacket` from DroneSecurity and drives them in-process. No
  subprocess.
- Same decode chain as the primary, but useful when the subprocess path
  is broken (venv missing, distutils shim unavailable).
- Same `TelemetryFrame` mapping as `DroneSecurityDecoder`.

### 4. `MMSEDecoder` — opt-in variant of Native

- Subclass of `NativeDecoder` that substitutes DroneSecurity's
  LS / Zero-Forcing channel equalization for **MMSE / Wiener-shrinkage**
  weights `W = conj(H_ls) / (|H_ls|² + σ²)`.
- Activated via `--use-mmse`. Replaces `NativeDecoder` in the last slot
  of the fallback chain.
- See [Channel equalization](#channel-equalization-ls-vs-mmse) below.

### Fallback ordering

The runner tries decoders in the order listed above and **stops at the
first decoder that returns at least one CRC-OK frame**. If a decoder
returns frames but none with `crc_ok=True`, it logs `no_crc_ok` and
moves on to the next one. Proto17 is **disabled by default**
(opt-in via `--enable-proto17`) because its Octave `find_zc.m` routinely
times out.

---

## CLI reference

```
run_pipeline.py --input <FILE> --sample-rate <Hz> [options]

  --input, -i              IQ capture (interleaved float32)        [required]
  --sample-rate, -s        Capture sample rate in Hz                [required]
  --legacy                 Treat as Mavic Pro / Mavic 2 (legacy CP layout)
  --output, -o             Output JSON path (default: results/<stem>_telemetry.json)
  --decoder-timeout        Per-decoder subprocess timeout, seconds (default 1800)

  --diagnose               If all decoders fail, run ZC correlator diagnostic
  --diagnose-only          Skip decoders, only run ZC correlator
  --spectrum-only          Skip decoders, run Welch band analysis
  --diagnose-chunk-seconds Diagnostic window in seconds (default 1.5)
  --diagnose-threshold     ZC correlation threshold (default 0.15)

  --center-shift-hz        Pre-shift IQ by this carrier offset (Hz) before decode
  --keep-bandwidth-hz      Bandpass bandwidth after shift (default 10 MHz)
  --output-rate-hz         Decimate to this rate post-shift (default = input)
  --keep-shifted-file      Don't delete the temp shifted .fc32 (debug)

  --enable-proto17         Include Octave-based Proto17 in the chain (off by default)
  --use-mmse               Swap NativeDecoder for MMSEDecoder (LS → MMSE equalizer)
  --kalman                 Smooth decoded frames with a 6D constant-velocity Kalman

  -v, --verbose            Enable DEBUG logging
```

### Recommended flags for large captures (multi-GB)

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 python -u uav_telemetry_pipeline/run_pipeline.py ...
```

Single-threaded scipy/numpy avoids WSL2 thread-thrash; unbuffered Python
lets you `tail -f` the log during long runs.

---

## Configuration

Resolution order (highest priority first):

1. CLI flag, e.g. an explicit path argument
2. Environment variable: `$DRONESECURITY_PATH`, `$PROTO17_PATH`,
   `$PIPELINE_RESULTS_DIR`
3. `uav_telemetry_pipeline/config.yaml` (optional, key
   `dronesecurity_path` / `proto17_path` / `results_dir`)
4. Defaults under `~/projects/PFE/`

Example `config.yaml`:

```yaml
dronesecurity_path: /opt/DroneSecurity
proto17_path: /opt/dji_droneid
results_dir: /var/lib/uav-telemetry/results
```

---

## Output JSON

```json
{
  "input": "<path>",
  "sample_rate_hz": 50000000.0,
  "decoder_used": "dronesecurity",        // null if all failed
  "frames_decoded": 9,
  "crc_ok_frames": 7,
  "crc_pass_rate": 0.778,
  "first_gps_fix": {"lat": 51.447, "lon": 7.266},  // null if none
  "attempts": [                            // per-decoder log
    {"decoder": "dronesecurity", "status": "ran",
     "frames": 9, "crc_ok": 7}
  ],
  "diagnostic": null,                      // present only with --diagnose
  "smoothed_track": [...],                 // present only with --kalman
  "track_metrics": {                       // present only with --kalman
    "rmse_raw_m": 12.3, "rmse_smoothed_m": 1.4,
    "smoothness_raw": 14.7, "smoothness_smoothed": 2.1,
    "outliers_rejected": 2
  },
  "frames": [
    {
      "lat": 0.0,                          // drone latitude (0.0 = no GPS lock)
      "lon": 0.0,
      "altitude_m": 0.0,
      "height_m": 0.0,
      "vel_north": 0.0,                    // raw drone units (decimeters/s)
      "vel_east": 0.0,
      "vel_up": 0.0,
      "yaw": 9575,                         // raw drone heading
      "serial": "SysSecWasHere",
      "device_type": "Mini 2",
      "app_lat": 51.447176178716916,       // pilot phone/app GPS
      "app_lon": 7.266528392911369,
      "home_lat": 0.0,
      "home_lon": 0.0,
      "sequence_number": 786,
      "gps_time_ms": 1648558221652,
      "crc_ok": true,
      "decoder": "dronesecurity",
      "timestamp_sample": -1               // sample index, -1 if unknown
    }
  ]
}
```

### Field semantics

- `lat / lon = 0.0` means the drone had no GPS lock at broadcast time
  (still-on-ground, indoors, etc.). The pilot (`app_lat / app_lon`)
  may still be valid in this case.
- `crc_ok` is the only authoritative validity flag. Frames with
  `crc_ok=false` are *kept* in the output (useful for debugging) but
  don't count toward `crc_ok_frames` or `first_gps_fix`.
- `timestamp_sample = -1` indicates the decoder didn't pin the frame
  to a specific sample offset. The `NativeDecoder` reports the chunk
  start; `DroneSecurityDecoder` leaves it at -1 because the receiver
  doesn't emit per-frame offsets.

### `--diagnose-only` output

```json
"diagnostic": {
  "file_seconds": 4.841,
  "threshold": 0.15,
  "best_score": 0.141,
  "best_sample_index": 148111420,
  "peaks_above_threshold": 0,
  "decim_q": 3,
  "burst_min_spacing_samples": 28944,
  "top_peaks": [...],                      // up to 20, sorted by score
  "chunks": [                              // per-chunk breakdown
    {"sample_start": 0, "samples": 50000000,
     "cfo_mhz": 0.0, "best_score": 0.135,
     "best_sample_index": 2732205,
     "peaks_above_threshold": 0}
  ]
}
```

### `--spectrum-only` output

```json
"spectrum": [
  {"sample_start": 0, "samples": 75000000,
   "bands": [
     {"f_center_hz": 280761.7,
      "f_start_hz": -10742187.5, "f_end_hz": 11303710.9,
      "bandwidth_hz": 22045898.4,
      "peak_psd_db_over_mean": 4.57,
      "label": "ocusync_video"}
   ]}
]
```

Heuristic labels emitted by `analyze_spectrum_bands`:

| Label | Bandwidth range | Notes |
|---|---|---|
| `c2_control` | 0.8 – 1.95 MHz | DJI command/control narrowband |
| `droneid` | 8 – 11 MHz, burst rate ≤ 5 Hz | LTE-derived DroneID broadcast (~1.7 Hz spec) |
| `lte_burst` | 8 – 11 MHz, burst rate > 5 Hz | LTE-shaped emission with the right width but the wrong rate — e.g. OcuSync sub-frame bursts. Looks like DroneID by spectrum alone but isn't. |
| `ocusync_video` | 17 – 25 MHz | OcuSync 2.0 video |
| `unknown` | anything else > 1 MHz | descriptive only |

The 8–11 MHz band defaults to `droneid` and is only demoted to `lte_burst`
when a per-band burst-rate detector actively measures a high (>5 Hz)
inter-burst rate. Zero-burst detector results stay `droneid`: real
DroneID at high duty cycle (e.g. curated multi-packet extracts) can
defeat the detector, so the conservative label is preferred over a
false negative.

---

## ZC correlator internals

`detection/zc_correlator.py` provides `ZadoffChuCorrelator` — the
matched filter used by `--diagnose`.

- **Reference:** root 147 (the deterministic fine-sync ZC root in every
  DroneID burst), seq length 601, IFFT'd to a time-domain waveform.
- **CFO correction:** Welch PSD band-search (mirrors
  DroneSecurity.helpers.estimate_offset). If a band of width 8–11 MHz is
  found above mean PSD, its center is removed via a complex rotator.
- **Decimation:** when the capture sample rate is much higher than the
  LTE base rate (15.36 Msps), the correlator decimates internally
  (polyphase, FIR-anti-aliased) so the matched filter operates on a
  smaller signal. At 50 Msps the correlator decimates by 3 → 16.67
  Msps, a ~10× speedup with no detectable quality loss.
- **Matched filter:** overlap-add convolution against the time-reversed
  conjugate reference, normalized by a sliding power envelope so
  scores land in [0, ~1].
- **Peaks:** `scipy.signal.find_peaks` with `height = threshold` and
  `distance = 0.9 × burst_length`.

Calibration: on `mini2_sm`, the correlator returns all 10 ground-truth
bursts at any threshold from 0.05 up to 0.30 (best peak score 0.78).
The default threshold of 0.15 was selected as a safety margin.

---

## Three-regime interpretation

The runner is designed to differentiate three regimes cleanly:

| Capture | OcuSync band | DroneID 8–11 MHz band | Best ZC score | Runner outcome |
|---|---|---|---|---|
| `DroneSecurity/samples/mini2_sm` (reference extract) | n/a | yes (CFO ≈ +9.6 MHz) | **0.78** | 9 frames, 7 CRC OK |
| `Recording 3.0 / mini2` (4.8 s field) | 19 MHz | **no** | 0.14 | 0 frames, diagnostic only |
| `Recording 3.0 / phantom` (3.7 s field) | 22 MHz | **no** | 0.14 | 0 frames, diagnostic only |

- The **reference extract** is a curated ZC-aligned slice → clean
  decode, multiple CRC-OK frames, drone serial recovered.
- The **field captures** contain active drone RF (OcuSync video clearly
  present at +4 dB PSD) but no DroneID-shaped band anywhere in the
  3.7–4.8 s span. ZC scores stay at the noise floor (~0.13–0.14, below
  the 0.15 threshold).

In the mini2 field capture, video drops out at t≈3.7 s and two narrow
auxiliary bands appear at -11.3 MHz (~4 MHz wide, +8–11 dB) and
-17.9 MHz (~3.8 MHz wide, +4–7 dB). These are continuous (not bursty),
appear simultaneously with video drop-out, and are too wide for DJI C2
and too narrow for DroneID — most likely DJI's OcuSync auxiliary
downlink (residual telemetry/control after the main video link
degrades). They are recorded in `results/band_probe.json` but are not
DroneID and are not decoded here.

---

## Channel equalization (LS vs MMSE)

DroneSecurity's `Packet.estimate_channel` uses **Least-Squares /
Zero-Forcing**: `H_ls = Y / X` then `X_est = Y / H_ls`. This is unbiased
but amplifies noise on subcarriers where the true channel response is
small (frequency-selective nulls). `decoders/mmse_equalizer.py` adds an
MMSE alternative:

```
W_mmse = conj(H_ls) / (|H_ls|² + σ²)
X_est  = W_mmse · Y                      (multiplicative, not divisive)
```

Noise variance `σ²` is estimated from the LS residual against a smoothed
channel; a guard-bin estimator (`estimate_noise_var`) is also available
for callers that retain the full 1024-bin FFT. `MMSEDecoder` replaces
`packet.channel` with `1 / W_mmse` so DroneSecurity's
`Y / packet.channel` pipeline yields the MMSE result without touching
DS source.

### Why CRC outcomes are identical to LS in practice

For per-subcarrier MMSE on the DroneID waveform, `X_mmse` and `X_ls`
differ only by a **real positive scalar per bin**
(`α = |H|² / (|H|² + σ²) ∈ (0, 1]`). DJI's DroneID frames use **hard-
decision QPSK without forward error correction**, and QPSK hard
decisions only look at the sign of `Re(X)` and `Im(X)` — invariant to
positive real scaling. The two equalizers therefore produce **identical
bit decisions**, hence identical CRC outcomes, on every realisation.

This is confirmed by `evaluate_snr.py`, a synthetic-SNR sweep harness:

```bash
python -u uav_telemetry_pipeline/evaluate_snr.py \
    -i uav_telemetry_pipeline/data/samples/mavic_air_2 \
    -s 50e6 \
    --snr-min -5 --snr-max 25 --snr-step 2 --trials 10 \
    --multipath-taps 3 --multipath-delay-spread-ns 500 \
    --rician-k-db 8.0 --show-channel-response \
    -o uav_telemetry_pipeline/results/fer_vs_snr.csv
```

Produces three figures: CRC-pass-rate vs SNR for flat AWGN
(`*_flat.png`), for AWGN + Rician multipath (`*_multipath.png`), and
post-equalization MSE on synthetic channels (`*_evm.png`). On the
mavic_air_2 reference, the CRC curves overlap exactly — the analytical
equivalence — while the EVM curve shows MMSE has measurably lower MSE.
The MMSE advantage would only translate to CRC on a system using
**soft-decision** QPSK with FEC, which DroneID lacks.

---

## Track smoothing (Kalman)

`tracking/kalman_tracker.py` adds a 6D constant-velocity Kalman filter
over the decoded frames:

| State | `[lat, lon, alt, v_lat, v_lon, v_alt]` |
|---|---|
| Observation | `[lat, lon, alt]` from each frame |
| Process noise | discrete white-acceleration, `σ_p = 1e-5` (deg/m units) |
| Measurement noise | `σ_meas_ok = 1e-4` (≈11 m), inflated 10× when `crc_ok = False` |

CRC-failed frames are **soft-rejected** (R inflated 100× in variance)
rather than dropped — preserving information when the tracker is short
on samples. Enable via `--kalman`:

```bash
python -u uav_telemetry_pipeline/run_pipeline.py \
    -i <capture.fc32> -s 50e6 --kalman \
    -o results/track.json
```

The output JSON gains `smoothed_track` (list of `SmoothedState` dicts)
and `track_metrics`:

| Metric | Meaning |
|---|---|
| `rmse_raw_m` | RMS haversine distance from each raw frame to its smoothed point |
| `rmse_smoothed_m` | RMS residual of smoothed track against its own 5-point moving average |
| `smoothness_raw` / `smoothness_smoothed` | Mean consecutive-sample displacement (m); lower = smoother |
| `outliers_rejected` | Count of raw frames > 10 m from smoothed track |

`tracking/track_evaluator.py` also exposes `plot()` for a raw-vs-smoothed
scatter (grey dots / blue line). Unit tests in
`tests/test_kalman_tracker.py` cover init, CRC inflation, outlier
suppression (≥50% of a +0.01° lat spike is rejected after 5 stable
samples), straight-line variance reduction, and reset semantics.

---

## Tests

All unit tests run under pytest from the project root:

```bash
source ~/projects/PFE/DroneSecurity/.venv/bin/activate
python -m pytest uav_telemetry_pipeline/tests/ -v
```

| File | Coverage |
|---|---|
| `tests/test_mmse_equalizer.py` | 13 tests — MMSE channel estimate, multiplicative weights, guard-bin noise estimation, packet-level averaging, shape contracts |
| `tests/test_kalman_tracker.py` | 7 tests — Kalman bootstrap, CRC-based R inflation, outlier suppression, variance reduction, reset, evaluator sanity |

---

## Layout

```
uav_telemetry_pipeline/
├── run_pipeline.py        ← CLI entry point (use this)
├── evaluate_snr.py        ← LS vs MMSE SNR sweep + Rician multipath
├── config.py              ← path/env resolution
├── decoders/
│   ├── base.py            ← BaseDecoder + TelemetryFrame
│   ├── dronesecurity.py   ← primary subprocess decoder
│   ├── proto17.py         ← Octave fallback (opt-in)
│   ├── native.py          ← in-process Python decoder
│   ├── mmse_equalizer.py  ← Wiener-shrinkage + multiplicative MMSE weights
│   └── mmse_decoder.py    ← NativeDecoder variant using MMSE equalization
├── detection/
│   ├── zc_correlator.py   ← matched filter + spectrum analysis
│   └── burst_detector.py  ← STFT-based energy detection (legacy)
├── preprocessor/
│   └── shift_and_filter.py ← carrier shift + bandpass + polyphase decimation
├── tracking/
│   ├── kalman_tracker.py  ← 6D constant-velocity Kalman filter
│   └── track_evaluator.py ← RMSE / smoothness / outlier metrics + plot
├── tests/                 ← pytest unit tests
├── telemetry/             ← (legacy) stdout parser + dataclasses
├── pipeline.py            ← (legacy) prior CLI; superseded by run_pipeline.py
├── results/               ← JSON outputs land here
└── data/samples/          ← capture files
```

`pipeline.py` and the modules under `telemetry/` predate this work and
are kept for the burst-detector / stdout-parser experiments documented
in the project notes. New work should target `run_pipeline.py` and the
`decoders/` package.

---

## Requirements

```
numpy>=1.24, scipy>=1.10, pandas>=2.0
pyyaml>=6.0           # optional, for config.yaml
matplotlib>=3.7       # debug plots + evaluate_snr.py + Kalman track plot
bitarray>=2.4, crcmod>=1.7, setuptools>=70.0    # for NativeDecoder / MMSEDecoder
pytest>=8.0           # to run uav_telemetry_pipeline/tests/
```

Octave (`/usr/bin/octave`) is optional — Proto17Decoder skips itself
silently if absent.

---

## Reproducing the validation results

```bash
source ~/projects/PFE/DroneSecurity/.venv/bin/activate

# Validation Step 1 — DroneSecurity directly on the reference
python ~/projects/PFE/DroneSecurity/src/droneid_receiver_offline.py \
    -i ~/projects/PFE/DroneSecurity/samples/mini2_sm -s 50e6

# Validation Step 2 — full pipeline on the reference
python -u uav_telemetry_pipeline/run_pipeline.py \
    -i ~/projects/PFE/DroneSecurity/samples/mini2_sm -s 50e6

# Validation Step 3 — diagnose on a field capture
python -u uav_telemetry_pipeline/run_pipeline.py \
    -i "data/samples/Recording 3.0/x310_rawIQ_2435MHz_50Msps_continuous_corne_phantom.fc32" \
    -s 50e6 --diagnose-only

# Validation Step 4 — descriptive spectrum of a field capture
python -u uav_telemetry_pipeline/run_pipeline.py \
    -i "data/samples/Recording 3.0/x310_rawIQ_2435MHz_50Msps_continuous_corne_phantom.fc32" \
    -s 50e6 --spectrum-only
```

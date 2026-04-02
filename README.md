# UAV Telemetry Pipeline

**Passive Analysis and Decoding of UAV Radio Links for Telemetry Recovery and Counter-Drone Applications**

A modular Python pipeline that processes recorded IQ radio captures to detect, decode, and extract structured DJI DroneID telemetry — including drone serial number, GPS coordinates, altitude, speed, heading, and pilot position.

Built as a final-year cybersecurity project (PFE).

---

## Problem Statement

Commercial drones broadcast identification and telemetry data via the DJI DroneID protocol embedded within their OcuSync 2.0 radio link. This broadcast is mandated for Remote ID compliance but is transmitted over the air in a format that requires specialized signal processing to recover.

This project provides an **offline, passive, receive-only** pipeline that takes raw IQ radio captures (recorded with an SDR) and extracts structured telemetry without any interaction with the drone or its controller.

## Goals

- Load and classify IQ captures from various SDR formats
- Detect DroneID signal bursts within wideband captures
- Integrate with existing reference decoders (no reimplementation of OFDM/PHY-layer math)
- Parse decoder output into clean, structured Python objects
- Produce JSON/CSV reports suitable for analysis and academic evaluation

---

## Features

| Stage | Module | Description | Status |
|-------|--------|-------------|--------|
| 1. Load IQ | `preprocessor/` | Load `.fc32`, `.cs8`, `.cs16`, `.npy` (Case A) or CSV (Case B) | Complete |
| 2. Burst Detection | `detection/burst_detector.py` | STFT-based energy detection with adaptive noise floor | Complete |
| 3. ZC Correlation | `detection/zc_correlator.py` | Zadoff-Chu reference generation | Partial (generation only) |
| 4. Decode | `decoding/` | Subprocess adapter for external decoders (DroneSecurity) | Complete |
| 5. Parse | `telemetry/` | Stdout JSON extraction into `DroneIDFrame` objects | Complete |
| Evaluate | `evaluate.py` | Aggregate metrics, terminal/JSON/CSV reports | Complete |

---

## Architecture

```
                         ┌──────────────────────────────────────────┐
                         │              pipeline.py                 │
                         │         (CLI orchestrator)               │
                         └──────┬───────┬───────┬───────┬──────────┘
                                │       │       │       │
                     Stage 1    │  St.2 │  St.3 │  St.4 │  Stage 5
                                ▼       ▼       ▼       ▼
IQ File (.fc32)  ──►  preprocessor/  detection/  decoding/  telemetry/
  or CSV (.csv)       load_verified  burst_      reference_  droneid_
                      _iq.py         detector.py decoder.py  parser.py
                                │               │           │
                                │               ▼           ▼
                                │        ┌──────────┐  DroneIDFrame
                                │        │DroneSec. │  (dataclass)
                                │        │subprocess│       │
                                │        └──────────┘       ▼
                                │                      DecodeResult
                                │                           │
                                └───────────────────────────┘
                                                            │
                                              output.json ◄─┘
                                                  │
                                            evaluate.py
                                                  │
                                     ┌────────────┼────────────┐
                                     ▼            ▼            ▼
                                 terminal     summary.json  summary.csv
```

**Key design decisions:**

- The pipeline does **not** reimplement OFDM demodulation or QPSK decoding. It wraps external reference decoders (primarily [RUB-SysSec/DroneSecurity](https://github.com/RUB-SysSec/DroneSecurity)) as subprocess calls.
- Input data is classified at load time as **Case A** (verified, decodable) or **Case B** (exploratory, best-effort). Only Case A data is sent to the decoder.
- All inter-stage communication uses typed Python dataclasses (`BurstSegment`, `DecodeResult`, `DroneIDFrame`).

---

## Case A vs Case B Inputs

| | Case A — Verified IQ | Case B — Exploratory |
|---|---|---|
| **Source** | SDR capture (USRP, HackRF, etc.) | CSV export (DroneDetect, DroneRF) |
| **Format** | `.fc32`, `.cs8`, `.cs16`, `.npy` | `.csv` with I/Q columns |
| **Sample rate** | Known (e.g., 50 MHz) | May be unknown |
| **Decodable** | Yes | Not guaranteed |
| **Pipeline behavior** | Full pipeline through decode | Load + detect only, decode skipped |

> Case B data is **never silently treated as decodable**. The pipeline logs a warning and tags all outputs with the classification.

---

## Repository Structure

```
uav_telemetry_pipeline/
├── pipeline.py              # Main CLI — runs the 5-stage pipeline
├── evaluate.py              # Evaluation report generator
├── requirements.txt         # Python dependencies
├── .gitignore
│
├── preprocessor/            # Stage 1: IQ loading and format conversion
│   ├── load_verified_iq.py  #   Case A: binary IQ formats → complex64
│   └── csv_to_iq.py         #   Case B: CSV → complex64
│
├── detection/               # Stage 2–3: Signal detection
│   ├── burst_detector.py    #   STFT energy detection + noise floor estimation
│   └── zc_correlator.py     #   Zadoff-Chu reference sequence generation
│
├── decoding/                # Stage 4: External decoder integration
│   ├── receiver_adapter.py  #   Abstract adapter interface (ABC)
│   └── reference_decoder.py #   Subprocess wrapper for DroneSecurity et al.
│
├── telemetry/               # Stage 5: Telemetry parsing and data models
│   ├── models.py            #   DroneIDFrame, DecodeResult, GPSCoordinate
│   └── droneid_parser.py    #   Stdout JSON extraction → DroneIDFrame
│
├── data/
│   ├── samples/             #   Input IQ captures (not committed to git)
│   └── results/             #   Pipeline outputs (not committed to git)
│
└── tests/                   #   Unit tests (placeholder)
```

---

## Quick Start

Minimal steps from clone to first successful decode:

```bash
# 1. Clone this repository
git clone https://github.com/CrypticNuts/uav_telemetry_pipeline.git
cd uav_telemetry_pipeline

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Clone and patch the DroneSecurity decoder
git clone https://github.com/RUB-SysSec/DroneSecurity.git ../DroneSecurity
pip install bitarray crcmod setuptools

# Apply compatibility patches for Python 3.12+ (see Troubleshooting)
sed -i 's/np\.complex,/complex,/' ../DroneSecurity/src/qpsk.py
sed -i 's/np\.complex(/complex(/' ../DroneSecurity/src/qpsk.py

# 4. Place an IQ capture in data/samples/
cp /path/to/your/capture.fc32 data/samples/

# 5. Run the pipeline
python3 pipeline.py \
  -i data/samples/capture.fc32 \
  --sample-rate 50e6 \
  --center-freq 2.4e9 \
  --decoder-path ../DroneSecurity \
  -o data/results/output.json \
  -v

# 6. Generate evaluation report
python3 evaluate.py data/results/output.json -o data/results/summary
```

---

## Installation

### Prerequisites

- **Python 3.10+** (tested on 3.13)
- **Linux** (tested on Kali 2024/WSL2 and Ubuntu 22.04)
- **pip** for package management

### Step 1: Install pipeline dependencies

```bash
pip install -r requirements.txt
```

This installs: `numpy`, `scipy`, `pandas`, `matplotlib`.

### Step 2: Set up the DroneSecurity decoder

The pipeline wraps [RUB-SysSec/DroneSecurity](https://github.com/RUB-SysSec/DroneSecurity) as its primary decoding backend. Clone it adjacent to the pipeline:

```bash
git clone https://github.com/RUB-SysSec/DroneSecurity.git ../DroneSecurity
```

Install its additional dependencies:

```bash
pip install bitarray crcmod setuptools
```

> `setuptools` provides the `distutils` module removed in Python 3.12+. See [Troubleshooting](#troubleshooting).

### Step 3: Apply compatibility patches

DroneSecurity was written for Python 3.9 / NumPy 1.22. Two changes are needed for modern environments:

```bash
# Fix np.complex removal (NumPy 1.24+)
sed -i 's/np\.complex,/complex,/' ../DroneSecurity/src/qpsk.py
sed -i 's/np\.complex(/complex(/' ../DroneSecurity/src/qpsk.py
```

Verify the decoder works standalone:

```bash
cd ../DroneSecurity/src
python3 droneid_receiver_offline.py -i ../../uav_telemetry_pipeline/data/samples/mavic_air_2 -s 50e6
```

You should see `## Drone-ID Payload ##` followed by a JSON block with coordinates.

---

## Usage

### Running the pipeline

**Basic usage with a verified IQ capture:**

```bash
python3 pipeline.py \
  -i data/samples/capture.fc32 \
  --sample-rate 50e6 \
  --center-freq 2.4e9 \
  --decoder-path ../DroneSecurity \
  -o data/results/output.json \
  -v
```

**Extensionless file (e.g., DroneSecurity samples):**

```bash
python3 pipeline.py \
  -i data/samples/mavic_air_2 \
  --fmt fc32 \
  --sample-rate 50e6 \
  --center-freq 2.4e9 \
  --decoder-path ../DroneSecurity \
  -o data/results/output.json \
  -v
```

**With explicit decoder script and extra arguments:**

```bash
python3 pipeline.py \
  -i data/samples/capture.fc32 \
  --decoder-path ../DroneSecurity \
  --decoder-backend src/droneid_receiver_offline.py \
  --decoder-args "-d" \
  --decoder-timeout 120 \
  -o data/results/output.json \
  -v
```

**CSV input (Case B — exploratory):**

```bash
python3 pipeline.py \
  -i data/samples/dronerf_export.csv \
  --csv-mode \
  -o data/results/output.json \
  -v
```

> Case B runs stages 1–2 only. The decoder is not invoked.

**Keep temp files for debugging:**

```bash
python3 pipeline.py \
  -i data/samples/mavic_air_2 \
  --fmt fc32 \
  --decoder-path ../DroneSecurity \
  --keep-temp \
  -o data/results/output.json \
  -v
```

The exported burst `.fc32` file will be kept in `/tmp/burst*_*/` so you can run the decoder on it manually.

### CLI reference

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Input file path (required) | — |
| `--fmt` | Explicit IQ format (`fc32`, `sc16`, `sc8`) | inferred from extension |
| `--csv-mode` | Treat input as CSV (Case B) | `false` |
| `--sample-rate` | Sample rate in Hz | `50e6` |
| `--center-freq` | Center frequency in Hz | `2.4e9` |
| `--decoder-path` | Path to decoder repository root | — |
| `--decoder-backend` | Explicit decoder script path | auto-discovered |
| `--decoder-args` | Extra args passed to the decoder | `""` |
| `--decoder-timeout` | Subprocess timeout (seconds) | `60` |
| `--keep-temp` | Keep temp burst files | `false` |
| `-o, --output` | Output JSON path | — |
| `-v, --verbose` | Debug logging | `false` |

### Running the evaluation report

```bash
# Single file
python3 evaluate.py data/results/output.json

# Multiple files with export
python3 evaluate.py data/results/*.json -o data/results/summary

# JSON only
python3 evaluate.py data/results/*.json --format json -o data/results/summary

# CSV only (for spreadsheet import)
python3 evaluate.py data/results/*.json --format csv -o data/results/summary
```

**Terminal output example:**

```
==========================================================================================
  UAV Telemetry Pipeline — Evaluation Report
==========================================================================================

File                        Case        Bursts Attempts Success Frames CRC OK Coords Avg Time
-----------------------------------------------------------------------------------------------
mavic_full_parse.json       verified_iq 1      1        1       1      1      1      3.84s
-----------------------------------------------------------------------------------------------
TOTAL                       verified_iq 1      1        1       1      1      1      3.84s

  Decode success rate:  1/1 (100.0%)
  Frames per success:   1.0
  CRC-valid frames:     1/1 (100.0%)
  Plausible coords:     1/1 (100.0%)
  Total decode time:    3.84s
  Unique serials:       1WNBH3900201N1
  Device types:         DJI Mavic Air 2

==========================================================================================
```

### Batch processing

Process multiple samples and generate a combined report:

```bash
for sample in data/samples/*; do
    [ -f "$sample" ] || continue
    name=$(basename "$sample")
    python3 pipeline.py \
      -i "$sample" \
      --fmt fc32 \
      --sample-rate 50e6 \
      --center-freq 2.4e9 \
      --decoder-path ../DroneSecurity \
      -o "data/results/${name}.json" \
      -v
done

python3 evaluate.py data/results/*.json -o data/results/evaluation
```

---

## Output Format

The pipeline produces a JSON array with one entry per detected burst:

```json
[
  {
    "burst_index": 0,
    "backend": "droneid_receiver_offline.py",
    "success": true,
    "exit_code": 0,
    "duration_s": 3.837,
    "command": "python3 .../droneid_receiver_offline.py -i /tmp/.../burst_0.fc32 -s 50000000.0",
    "error_message": "",
    "num_frames": 1,
    "frames": [
      {
        "serial_number": "1WNBH3900201N1",
        "manufacturer": "DJI Mavic Air 2",
        "classification": "verified_iq",
        "decode_confidence": 1.0,
        "drone_lat": 51.4463,
        "drone_lon": 7.2672,
        "drone_alt": 42.97,
        "pilot_lat": 51.4462,
        "pilot_lon": 7.2671,
        "home_lat": 51.4463,
        "home_lon": 7.2674,
        "speed_horizontal_ms": 2.83,
        "speed_vertical_ms": 26.0,
        "heading_deg": 25.0,
        "height_agl_m": 12.8,
        "timestamp": "2022-04-21T11:53:46.258000+00:00"
      }
    ]
  }
]
```

| Field | Description |
|-------|-------------|
| `serial_number` | DJI drone serial number |
| `manufacturer` | Device type (e.g., "DJI Mavic Air 2") |
| `decode_confidence` | 1.0 if CRC matches, 0.5 if CRC mismatch |
| `drone_lat/lon/alt` | Drone GPS position (WGS84) |
| `pilot_lat/lon` | Pilot/controller app position |
| `home_lat/lon` | Home point position |
| `speed_horizontal_ms` | Horizontal speed in m/s |
| `speed_vertical_ms` | Vertical speed in m/s |
| `heading_deg` | Heading in degrees (0–360) |
| `height_agl_m` | Height above ground level in meters |
| `timestamp` | GPS timestamp (UTC ISO 8601) |

---

## Using Your Own Data


To use this pipeline with new IQ captures:

1. **Record an IQ capture** at the DroneID frequency (around 2.4 GHz) using an SDR such as a USRP, HackRF, or RTL-SDR at **50 MSPS** or higher.

2. **Save the capture** in one of the supported formats:
   - `.fc32` — interleaved float32 I/Q (most common, default for GNU Radio)
   - `.cs16` — interleaved signed int16 I/Q
   - `.cs8` — interleaved signed int8 I/Q
   - `.npy` — NumPy complex64 array

3. **Place the file** in `data/samples/`.

4. **Run the pipeline:**
   ```bash
   python3 pipeline.py \
     -i data/samples/your_capture.fc32 \
     --sample-rate <your_sample_rate> \
     --center-freq <your_center_freq> \
     --decoder-path ../DroneSecurity \
     -o data/results/your_capture.json \
     -v
   ```

5. **Check the output** in `data/results/your_capture.json`. The `frames` array contains decoded telemetry. If `frames` is empty, check:
   - Is the sample rate correct?
   - Does the capture contain a DroneID signal? (Check the burst detection log)
   - Did the decoder produce output? (Check `stdout` and `stderr` in the JSON)

### If your file has no extension

Use `--fmt` to specify the format explicitly:

```bash
python3 pipeline.py -i data/samples/my_capture --fmt fc32 --sample-rate 50e6 ...
```

### If using a different sample rate

DroneSecurity expects 50 MSPS captures. If your SDR records at a different rate, specify it:

```bash
python3 pipeline.py -i data/samples/capture.fc32 --sample-rate 30.72e6 ...
```

> Note: The DroneSecurity decoder may not decode correctly at sample rates other than 50 MSPS. The burst detector itself works at any sample rate.

---

## Reproducibility

Exact steps to reproduce the successful decode verified during development:

```bash
# Environment
# Python 3.13.12, Kali Linux (WSL2), April 2026

# 1. Install dependencies
pip install numpy scipy pandas matplotlib bitarray crcmod setuptools

# 2. Clone decoder and apply patches
git clone https://github.com/RUB-SysSec/DroneSecurity.git ../DroneSecurity
sed -i 's/np\.complex,/complex,/' ../DroneSecurity/src/qpsk.py
sed -i 's/np\.complex(/complex(/' ../DroneSecurity/src/qpsk.py

# 3. Obtain the mavic_air_2 sample from DroneSecurity
cp ../DroneSecurity/samples/mavic_air_2 data/samples/

# 4. Run the pipeline
python3 pipeline.py \
  -i data/samples/mavic_air_2 \
  --fmt fc32 \
  --sample-rate 50e6 \
  --center-freq 2.4e9 \
  --decoder-path ../DroneSecurity \
  -o data/results/mavic_full_parse.json \
  -v

# Expected output:
#   Stage 2: 1 burst detected (samples 93184–225280, 2.64 ms, SNR 17.8 dB)
#   Stage 4: exit_code=0, CRC OK
#   Stage 5: 1 frame decoded
#     Serial: 1WNBH3900201N1
#     Model:  DJI Mavic Air 2
#     Drone:  51.4463°N, 7.2672°E, 42.97m
#     Pilot:  51.4462°N, 7.2671°E
#     Time:   2022-04-21T11:53:46 UTC

# 5. Generate evaluation report
python3 evaluate.py data/results/mavic_full_parse.json -o data/results/summary
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'distutils'`

**Cause:** Python 3.12+ removed the `distutils` module. DroneSecurity's `SpectrumCapture.py` imports it.

**Fix:**
```bash
pip install setuptools
```

### `AttributeError: module 'numpy' has no attribute 'complex'`

**Cause:** NumPy 1.24+ removed the deprecated `np.complex` alias. DroneSecurity's `qpsk.py` uses it.

**Fix:**
```bash
sed -i 's/np\.complex,/complex,/' ../DroneSecurity/src/qpsk.py
sed -i 's/np\.complex(/complex(/' ../DroneSecurity/src/qpsk.py
```

### `Detected 0 burst(s)` on a file that should contain a signal

The burst detector uses a histogram-based noise floor estimator. If the signal occupies most of the capture, try lowering the threshold:

```bash
# In your Python code or by editing BurstDetector defaults:
detector = BurstDetector(threshold_db=6.0, nperseg=256, overlap_frac=0.75)
```

### `Unsupported format ''` for extensionless files

Use `--fmt` to specify the format:
```bash
python3 pipeline.py -i data/samples/my_file --fmt fc32 ...
```

### Decoder exits with code 1 but no useful stderr

Run the decoder standalone to isolate the issue:
```bash
cd ../DroneSecurity/src
python3 droneid_receiver_offline.py -i /path/to/burst.fc32 -s 50e6 -d
```

The `-d` flag enables DroneSecurity's debug output.

### `frames: []` despite `success: true`

The decoder ran successfully but either:
- No DroneID payload was found in the signal (common for frame 1 of 2 in the DroneSecurity output — this is expected)
- The stdout parser didn't find a `## Drone-ID Payload ##` marker

Check the `stdout` field in the output JSON for the raw decoder output.

---

## Limitations

- **Verified IQ only for decoding.** Only Case A inputs (known sample rate, center frequency, and sufficient bandwidth) produce reliable decode results. Case B (CSV/exploratory) data is loaded and detected but not decoded.
- **External decoder dependency.** The OFDM demodulation and QPSK decoding are performed by the [DroneSecurity](https://github.com/RUB-SysSec/DroneSecurity) reference implementation. This pipeline does not contain its own PHY-layer decoder.
- **50 MSPS expected.** The DroneSecurity decoder expects captures at 50 MSPS. Other sample rates may cause decode failures.
- **DJI protocol only.** This pipeline targets the DJI DroneID protocol. Other manufacturers' Remote ID implementations are not supported.
- **Proprietary protocol.** DJI DroneID field semantics are based on reverse-engineering research. Some fields (e.g., `d_1_angle`, `state_info`) may not be fully documented.
- **Passive and offline only.** No real-time processing, no transmission, no active interaction with drones.
- **ZC correlation not yet integrated.** The Zadoff-Chu correlator generates the reference sequence but the cross-correlation and frame-start detection are not yet implemented. Burst detection relies on STFT energy alone.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.24 | Array operations, IQ data handling |
| `scipy` | >= 1.10 | STFT computation for burst detection |
| `pandas` | >= 2.0 | CSV input loading (Case B) |
| `matplotlib` | >= 3.7 | Debug plots for burst detection |
| `bitarray` | >= 2.4 | Required by DroneSecurity decoder |
| `crcmod` | >= 1.7 | Required by DroneSecurity decoder |
| `setuptools` | any | Provides `distutils` on Python 3.12+ |

---

## References

- Schiller, M. et al. — *"Drone Security and the Mystery of DJI's DroneID"* (RUB-SysSec), USENIX Security 2023
- [RUB-SysSec/DroneSecurity](https://github.com/RUB-SysSec/DroneSecurity) — Reference decoder implementation
- ASTM F3411 — Standard Specification for Remote ID and Tracking
- 3GPP TS 36.211 — Zadoff-Chu sequence definitions
- [anarkiwi/samples2djidroneid](https://github.com/anarkiwi/samples2djidroneid) — Alternative decoder reference

---

## License

This project is developed as an academic PFE (Projet de Fin d'Etudes).

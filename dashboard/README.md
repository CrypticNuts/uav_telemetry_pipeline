# UAV Telemetry Dashboard

Lightweight Flask + Leaflet dashboard for visualising decoded DJI DroneID
frames produced by the `uav_telemetry_pipeline`. Streams `.jsonl` frames
over Server-Sent Events with no database, no JS build step, and a single
runtime dependency (`flask`).

## Install

```bash
pip install -r requirements.txt
```

## Expected frame schema

The dashboard reads one JSON object per line. Each frame should follow:

```json
{
  "timestamp":     "2026-05-21T12:34:56Z",
  "decoder":       "droneid",
  "drone_lat":     36.8090,
  "drone_lon":     10.1330,
  "drone_alt":     80.0,
  "drone_speed":   8.0,
  "pilot_lat":     36.8090,
  "pilot_lon":     10.1330,
  "serial_number": "1WNBH3900201N1",
  "crc_valid":     true
}
```

## Live mode

Point the dashboard at the file the pipeline is currently appending to.
The server tails it line by line — only frames written **after** the page
opens are shown.

```bash
# Option A: CLI flag
python app.py --file /path/to/telemetry_stream.jsonl

# Option B: env var
TELEMETRY_FILE=/path/to/telemetry_stream.jsonl python app.py
```

Then open <http://127.0.0.1:5000/>. The header pill turns green (`LIVE`)
once the SSE stream connects.

To exercise live mode without an SDR, run the demo generator in another
terminal:

```bash
python generate_demo.py --out telemetry_stream.jsonl --live --frames 300
```

## Replay mode (demo / jury presentation)

Replay an existing recording at adjustable speed without touching the live
file. Append `?replay=<path>&speed=<x>` to the dashboard URL:

```bash
# 1. generate a 300-frame recording
python generate_demo.py --out demo.jsonl --frames 300

# 2. start the server (any --file is fine, replay overrides it)
python app.py --file demo.jsonl

# 3. open the replay URL
#    http://127.0.0.1:5000/?replay=demo.jsonl&speed=4
```

`speed=1.0` re-emits at 1 Hz, `speed=4.0` at 4 Hz, `speed=0.5` at half
speed. The header pill turns amber (`REPLAY`) and flips to `REPLAY DONE`
when the file is exhausted.

## Replaying real recordings

The pipeline emits a single JSON document per recording (see
`results/mavic_air_2_DECODE.json`). The adapter `pipeline_to_jsonl.py`
converts that to the dashboard's `.jsonl` schema, mapping the pipeline
fields and skipping zero-fix or out-of-range positions so the drone
marker never snaps to (0, 0) or to garbage coordinates from a
CRC-failed frame.

```bash
# Convert a pipeline output to dashboard jsonl
python pipeline_to_jsonl.py ../results/mavic_air_2_DECODE.json -o mavic_air_2.jsonl
python pipeline_to_jsonl.py ../results/mini2_sm_telemetry.json  -o mini2_sm.jsonl

# Start the server (any --file is fine; replay overrides it)
python app.py --file mavic_air_2.jsonl

# Open one of:
#   http://127.0.0.1:5000/?replay=mavic_air_2.jsonl&speed=1
#   http://127.0.0.1:5000/?replay=mini2_sm.jsonl&speed=2
```

What you'll see on the included captures:

| Recording      | Frames | CRC OK | Real drone GPS | Notes |
|----------------|--------|--------|----------------|-------|
| `mavic_air_2`  | 1      | 1      | yes            | Single point in Bochum, DE (51.4463, 7.2672). Capture too short for a track. |
| `mini2_sm`     | 9      | 7      | no             | Drone hadn't acquired GPS lock — frames carry `lat=lon=0`; pilot/app coords are valid. Last frame is a CRC-failed corrupt one and turns the CRC indicator red. |

Both demonstrate that the dashboard is purely offline: the SDR plays no
part once the IQ has been decoded into `.jsonl`.

Adapter flags:

| Flag             | Effect                                                    |
|------------------|-----------------------------------------------------------|
| `--skip-no-fix`  | Drop frames with no drone GPS fix entirely (smaller jsonl) |

## Status endpoint

```bash
curl http://127.0.0.1:5000/status
# {"frame_count": 42, "last_serial": "1WNBH3900201N1", ...}
```

## Files

```
dashboard/
  app.py                 # Flask server: /, /stream, /replay, /status
  templates/index.html   # Single-file Leaflet UI (CDN, no build step)
  generate_demo.py       # Synthetic circular-patrol generator (offline demo)
  pipeline_to_jsonl.py   # Pipeline JSON -> dashboard .jsonl adapter
  requirements.txt       # flask
  README.md              # this file
```

## Offline operation

The dashboard never talks to an SDR. Everything it renders comes from
`.jsonl` text files on disk, which means three offline workflows are
available without any radio hardware:

1. **Synthetic replay** — `generate_demo.py` writes a deterministic
   circular-patrol recording, perfect for jury demos that must survive a
   flaky Wi-Fi room.
2. **Synthetic live tail** — run `generate_demo.py --live` in one
   terminal to drip-feed frames at 1 Hz while the dashboard tails the
   file in another, visually identical to a real SDR feed.
3. **Real-recording replay** — convert a pipeline output JSON with
   `pipeline_to_jsonl.py` and serve it through `/replay`.

## Notes

- The SSE generator tails the file with periodic short sleeps and never
  loads it into memory.
- If the SSE connection drops, the browser auto-reconnects every 3 s.
- The drone marker turns **red** when `crc_valid` is `false`.
- The map auto-centres on the first valid drone coordinate, then stays
  put so the operator can pan without losing focus.
- `null` drone coordinates (no GPS fix) are honoured: the sidebar still
  updates with the frame, but the drone marker is **not** moved — this
  is what mini2_sm looks like before the drone acquires GPS lock.

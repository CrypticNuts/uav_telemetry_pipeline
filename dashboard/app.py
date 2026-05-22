"""
app.py — Flask backend for the UAV telemetry dashboard.

Serves a Leaflet map at ``/`` and streams decoded DroneID frames over
Server-Sent Events (``/stream``). Designed for two modes:

* **Live mode** — tails ``telemetry_stream.jsonl`` as the pipeline appends
  lines. Each new line is forwarded as one SSE event.
* **Replay mode** — re-emits the lines of an existing ``.jsonl`` file at a
  configurable speed for offline demos.

Input file path can be supplied via the ``--file`` CLI flag or the
``TELEMETRY_FILE`` env var. The server holds no database and never loads
the whole file into memory.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from flask import Flask, Response, render_template, request, jsonify


# -------------------------------------------------------------- configuration

DEFAULT_FILE = "telemetry_stream.jsonl"
POLL_INTERVAL_S = 0.25  # tail loop sleep when at EOF
HEARTBEAT_INTERVAL_S = 15.0  # comment events to keep the SSE connection warm


# ------------------------------------------------------------------ app state

app = Flask(__name__)


class DashboardState:
    """Thread-safe counters surfaced by ``/status``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.frame_count: int = 0
        self.last_serial: Optional[str] = None
        self.last_timestamp: Optional[str] = None
        self.telemetry_file: Path = Path(DEFAULT_FILE)

    def record(self, frame: dict) -> None:
        with self._lock:
            self.frame_count += 1
            sn = frame.get("serial_number")
            if isinstance(sn, str) and sn:
                self.last_serial = sn
            ts = frame.get("timestamp")
            if isinstance(ts, str) and ts:
                self.last_timestamp = ts

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "frame_count": self.frame_count,
                "last_serial": self.last_serial,
                "last_timestamp": self.last_timestamp,
                "telemetry_file": str(self.telemetry_file),
            }


state = DashboardState()


# ----------------------------------------------------------- tail / replay IO


def _tail_lines(path: Path) -> Iterator[str]:
    """Yield new lines appended to ``path``.

    Blocks (with short sleeps) when at EOF and resumes once more data is
    written. If the file does not yet exist, polls until it appears.
    """
    while not path.exists():
        time.sleep(POLL_INTERVAL_S)

    with path.open("r", encoding="utf-8") as fh:
        # Start at end-of-file: the dashboard should only show frames that
        # arrive after the user opens it (typical "live tail" behaviour).
        fh.seek(0, os.SEEK_END)
        last_beat = time.monotonic()
        while True:
            line = fh.readline()
            if not line:
                now = time.monotonic()
                if now - last_beat >= HEARTBEAT_INTERVAL_S:
                    yield ""  # heartbeat sentinel
                    last_beat = now
                time.sleep(POLL_INTERVAL_S)
                continue
            line = line.strip()
            if line:
                yield line
                last_beat = time.monotonic()


def _replay_lines(path: Path, speed: float) -> Iterator[str]:
    """Yield lines of ``path`` with inter-line delay scaled by ``speed``."""
    speed = max(0.01, float(speed))
    delay = 1.0 / speed
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            yield line
            time.sleep(delay)


def _format_sse(payload: str, event: Optional[str] = None) -> str:
    """Wrap a payload as a single SSE ``data:`` block."""
    if payload == "":
        # Comment lines are valid SSE keep-alives and ignored by EventSource.
        return ": heartbeat\n\n"
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"


# ----------------------------------------------------------------- endpoints


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/status")
def status() -> Response:
    return jsonify(state.snapshot())


@app.route("/stream")
def stream() -> Response:
    """Live-tail SSE endpoint."""

    def _gen() -> Iterator[str]:
        yield _format_sse(json.dumps({"mode": "live"}), event="mode")
        for line in _tail_lines(state.telemetry_file):
            if line == "":
                yield _format_sse("")
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            state.record(frame)
            yield _format_sse(json.dumps(frame))

    return Response(_gen(), mimetype="text/event-stream")


@app.route("/replay")
def replay() -> Response:
    """Re-emit a recorded ``.jsonl`` file as SSE."""
    file_arg = request.args.get("file") or str(state.telemetry_file)
    speed = float(request.args.get("speed", "1.0"))
    target = Path(file_arg).expanduser()

    if not target.exists() or not target.is_file():
        return jsonify({"error": f"file not found: {target}"}), 404

    def _gen() -> Iterator[str]:
        yield _format_sse(
            json.dumps({"mode": "replay", "file": str(target), "speed": speed}),
            event="mode",
        )
        for line in _replay_lines(target, speed):
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            state.record(frame)
            yield _format_sse(json.dumps(frame))
        yield _format_sse(json.dumps({"done": True}), event="end")

    return Response(_gen(), mimetype="text/event-stream")


# --------------------------------------------------------------------- entry


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--file", "-f",
        help="Path to the telemetry .jsonl file (default: $TELEMETRY_FILE or telemetry_stream.jsonl)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    chosen = args.file or os.environ.get("TELEMETRY_FILE") or DEFAULT_FILE
    state.telemetry_file = Path(chosen).expanduser().resolve()
    print(f"[dashboard] telemetry file: {state.telemetry_file}")
    print(f"[dashboard] serving on http://{args.host}:{args.port}/")
    # threaded=True lets each SSE generator run on its own thread without
    # blocking the index page or status endpoint.
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()

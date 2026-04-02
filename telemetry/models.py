"""
models.py — Data models for the UAV telemetry pipeline.

Defines the structured output types used throughout the pipeline.
All models use dataclasses for clarity and immutability where possible.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class InputClassification(Enum):
    """Classification of input data provenance.

    CASE_A: Verified complex IQ captures with known sample rate,
            center frequency, and bandwidth — suitable for decoding.

    CASE_B: Exploratory RF files (e.g., DroneDetect/DroneRF CSV exports)
            that may lack the properties needed for DroneID decoding.
            Downstream stages should treat these as best-effort.
    """

    CASE_A = "verified_iq"
    CASE_B = "exploratory"


@dataclass(frozen=True)
class GPSCoordinate:
    """A WGS84 GPS coordinate."""

    latitude: float
    longitude: float
    altitude_m: float

    def __post_init__(self) -> None:
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Invalid longitude: {self.longitude}")


@dataclass(frozen=True)
class DroneIDFrame:
    """A decoded DJI DroneID telemetry frame.

    Fields correspond to the DroneID specification as documented
    in RUB-SysSec/DroneSecurity and ASTM F3411 Remote ID.
    """

    # Serial / identification
    serial_number: str
    manufacturer: str = "DJI"

    # Position
    drone_position: GPSCoordinate | None = None
    pilot_position: GPSCoordinate | None = None
    home_position: GPSCoordinate | None = None

    # Flight dynamics
    speed_horizontal_ms: float | None = None
    speed_vertical_ms: float | None = None
    heading_deg: float | None = None
    height_agl_m: float | None = None

    # Timestamps
    timestamp: datetime | None = None

    # Decode metadata
    input_classification: InputClassification = InputClassification.CASE_A
    decode_confidence: float | None = None

    def is_position_valid(self) -> bool:
        """Check if the drone position is present and non-zero."""
        if self.drone_position is None:
            return False
        return not (
            self.drone_position.latitude == 0.0
            and self.drone_position.longitude == 0.0
        )


@dataclass
class DecodeResult:
    """Structured result from an external decoder invocation.

    Captures everything needed to diagnose success or failure:
    the subprocess output, timing, artifact paths, and any
    parsed telemetry frames.
    """

    # Identity
    burst_index: int
    backend_name: str

    # Subprocess outcome
    success: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    command: str = ""

    # Timing
    duration_s: float = 0.0

    # File I/O
    input_file: Path | None = None
    artifact_paths: list[Path] = field(default_factory=list)

    # Parsed output
    frames: list[DroneIDFrame] = field(default_factory=list)
    error_message: str = ""

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def summary(self) -> str:
        """One-line human-readable summary."""
        status = "OK" if self.success else "FAIL"
        return (
            f"[{status}] burst={self.burst_index} "
            f"backend={self.backend_name} "
            f"exit={self.exit_code} "
            f"frames={self.num_frames} "
            f"duration={self.duration_s:.2f}s"
        )

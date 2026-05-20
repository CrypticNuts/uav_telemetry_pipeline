"""
kalman_tracker.py — 6D constant-velocity Kalman tracker for DroneID frames.

State (6D):    x = [lat, lon, alt, v_lat, v_lon, v_alt]
Observation:   z = [lat, lon, alt]   (extracted from each decoded frame)
Model:         constant-velocity with discrete white-acceleration noise.
Measurement:   per-frame inflation when ``crc_ok`` is False.

Units are kept consistent with the upstream pipeline: lat/lon in **degrees**,
altitude in **metres**, velocities in **deg/s** (lat, lon) and **m/s** (alt).
The exposed ``uncertainty_m`` field collapses the position covariance back to
a single metric distance for ease of reporting.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


# 1 degree of latitude is approximately 111 320 m on the WGS-84 ellipsoid.
# Longitude metres-per-degree depend on latitude, but for the small per-track
# extent of DroneID captures (< 1 km), using a constant pre-multiplier is
# accurate to fractions of a percent and avoids a state-dependent Jacobian.
_METRES_PER_DEG = 111_320.0


class SmoothedState(TypedDict, total=False):
    """One smoothed track sample, produced by ``KalmanTracker.update``."""

    lat: float
    lon: float
    alt: float
    v_lat: float
    v_lon: float
    v_alt: float
    uncertainty_m: float
    crc_ok: bool
    sequence_number: int


class KalmanTracker:
    """Linear 6D constant-velocity Kalman filter for DroneID position frames.

    Parameters
    ----------
    sigma_process : float
        Standard deviation of the discrete white-acceleration process noise.
        Drives Q. Larger values let the filter track manoeuvres at the cost
        of noisier output.
    sigma_meas_ok : float
        Position measurement noise std (in degrees / metres) for frames with
        ``crc_ok == True``. Sets the diagonal of R for trusted frames.
    sigma_meas_bad : float
        Position measurement noise std for frames with ``crc_ok == False``.
        Typically ~10x larger than ``sigma_meas_ok`` to down-weight the
        outlier without ignoring it entirely.
    dt : float
        Nominal time step between frames (s). DroneID broadcasts at ~1 Hz,
        so ``dt=1.0`` matches the default cadence.
    """

    def __init__(
        self,
        sigma_process: float = 1e-5,
        sigma_meas_ok: float = 1e-4,
        sigma_meas_bad: float = 1e-3,
        dt: float = 1.0,
    ) -> None:
        self.sigma_process = float(sigma_process)
        self.sigma_meas_ok = float(sigma_meas_ok)
        self.sigma_meas_bad = float(sigma_meas_bad)
        self.dt = float(dt)
        self._track: list[SmoothedState] = []
        self.reset()

    # ------------------------------------------------------------------ matrices

    @property
    def F(self) -> np.ndarray:
        """State transition (constant velocity)."""
        dt = self.dt
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    @property
    def H(self) -> np.ndarray:
        """Observation matrix — picks (lat, lon, alt) from the 6D state."""
        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        return H

    @property
    def Q(self) -> np.ndarray:
        """Discrete white-acceleration process noise covariance (6x6)."""
        dt, s = self.dt, self.sigma_process
        s2 = s * s
        q_pp = (dt ** 4) / 4.0
        q_pv = (dt ** 3) / 2.0
        q_vv = dt ** 2
        Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            Q[i, i] = q_pp
            Q[i + 3, i + 3] = q_vv
            Q[i, i + 3] = q_pv
            Q[i + 3, i] = q_pv
        return s2 * Q

    def _R(self, crc_ok: bool) -> np.ndarray:
        s = self.sigma_meas_ok if crc_ok else self.sigma_meas_bad
        return (s * s) * np.eye(3, dtype=np.float64)

    # ----------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Clear state, covariance, and accumulated track."""
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)  # large prior; first update will dominate
        self._initialized = False
        self._track.clear()

    def get_track(self) -> list[SmoothedState]:
        """Return the list of smoothed states produced so far (chronological)."""
        return list(self._track)

    # ------------------------------------------------------------------- update

    def update(self, frame: dict) -> SmoothedState:
        """Ingest one decoded telemetry frame; return the smoothed state."""
        z = np.array([
            float(frame.get("lat", 0.0)),
            float(frame.get("lon", 0.0)),
            float(frame.get("altitude_m", 0.0)),
        ], dtype=np.float64)
        crc_ok = bool(frame.get("crc_ok", False))

        if not self._initialized:
            # Bootstrap the state from the first observation.
            self.x[:3] = z
            self.x[3:] = 0.0
            # Initialise position variance to the trusted measurement noise so
            # later updates can shrink it; velocity stays at the prior.
            R0 = self._R(crc_ok)
            self.P[:3, :3] = R0
            self.P[3:, 3:] = np.eye(3, dtype=np.float64) * (self.sigma_process ** 2)
            self.P[:3, 3:] = 0.0
            self.P[3:, :3] = 0.0
            self._initialized = True
        else:
            # Predict
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            # Update
            R = self._R(crc_ok)
            S = self.H @ self.P @ self.H.T + R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            y = z - self.H @ self.x
            self.x = self.x + K @ y
            self.P = (np.eye(6) - K @ self.H) @ self.P

        smoothed: SmoothedState = {
            "lat": float(self.x[0]),
            "lon": float(self.x[1]),
            "alt": float(self.x[2]),
            "v_lat": float(self.x[3]),
            "v_lon": float(self.x[4]),
            "v_alt": float(self.x[5]),
            "uncertainty_m": self._position_uncertainty_m(),
            "crc_ok": crc_ok,
            "sequence_number": int(frame.get("sequence_number", -1)),
        }
        self._track.append(smoothed)
        return smoothed

    # ---------------------------------------------------------------- internals

    def _position_uncertainty_m(self) -> float:
        """Collapse the 3x3 position covariance into a single scalar metre value.

        Uses ``sqrt(trace(P_xy in m^2) + P_alt)`` — the 1-sigma equivalent
        radius of the position uncertainty ellipsoid.
        """
        var_lat_m2 = self.P[0, 0] * (_METRES_PER_DEG ** 2)
        var_lon_m2 = self.P[1, 1] * (_METRES_PER_DEG ** 2)
        var_alt_m2 = self.P[2, 2]
        return float(np.sqrt(var_lat_m2 + var_lon_m2 + var_alt_m2))

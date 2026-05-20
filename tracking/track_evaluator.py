"""
track_evaluator.py — Quality metrics + plotting for KalmanTracker output.

Treats the smoothed track as the reference trajectory and quantifies how
much noise/outliers the filter rejected from the raw frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np


_EARTH_RADIUS_M = 6_371_008.8  # mean Earth radius (WGS-84 sphere approx)


class TrackMetrics(TypedDict, total=False):
    rmse_raw_m: float
    rmse_smoothed_m: float
    smoothness_raw: float
    smoothness_smoothed: float
    outliers_rejected: int


def _haversine_m(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance between two lat/lon points, in metres."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a)))


class TrackEvaluator:
    """Compute residual-noise and smoothness metrics for a Kalman track."""

    def __init__(self, outlier_threshold_m: float = 10.0) -> None:
        self.outlier_threshold_m = float(outlier_threshold_m)

    # ------------------------------------------------------------------ metrics

    def evaluate(
        self,
        raw_frames: list[dict],
        smoothed_track: list[dict],
    ) -> TrackMetrics:
        """Return RMSE, smoothness and outlier stats for the two tracks.

        Conventions
        -----------
        - The smoothed track is treated as the reference trajectory.
        - ``rmse_raw_m`` is the RMS great-circle distance from each raw
          frame to its corresponding smoothed point — measures the raw
          noise the Kalman filter removed.
        - ``rmse_smoothed_m`` is the RMS distance from each smoothed point
          to a 5-point moving-average of the smoothed track — measures
          residual noise the filter could not absorb.
        - ``smoothness_*`` is the mean consecutive-sample displacement.
          Lower = smoother. The filter should reduce it substantially
          versus the raw input.
        - ``outliers_rejected`` counts raw frames more than
          ``outlier_threshold_m`` metres from the smoothed track.
        """
        n = min(len(raw_frames), len(smoothed_track))
        if n == 0:
            return TrackMetrics(
                rmse_raw_m=0.0, rmse_smoothed_m=0.0,
                smoothness_raw=0.0, smoothness_smoothed=0.0,
                outliers_rejected=0,
            )

        raw_lat = np.array([f.get("lat", 0.0) for f in raw_frames[:n]])
        raw_lon = np.array([f.get("lon", 0.0) for f in raw_frames[:n]])
        sm_lat = np.array([s.get("lat", 0.0) for s in smoothed_track[:n]])
        sm_lon = np.array([s.get("lon", 0.0) for s in smoothed_track[:n]])

        # Per-sample distance raw <-> smoothed.
        d_pair = np.array([
            _haversine_m(raw_lat[i], raw_lon[i], sm_lat[i], sm_lon[i])
            for i in range(n)
        ])
        rmse_raw_m = float(np.sqrt(np.mean(d_pair ** 2)))
        outliers = int(np.sum(d_pair > self.outlier_threshold_m))

        # Smoothed residuals against a moving-average reference.
        # Use boundary-aware averaging: divide by the count of actual samples
        # under the window at each position, not by the window size. Without
        # this correction, np.convolve(mode='same') zero-pads beyond the
        # edges and the "average" near the boundaries collapses toward 0,0.
        win = min(5, max(1, n))
        ones_k = np.ones(win)
        counts = np.convolve(np.ones(n), ones_k, mode="same")
        avg_lat = np.convolve(sm_lat, ones_k, mode="same") / counts
        avg_lon = np.convolve(sm_lon, ones_k, mode="same") / counts
        d_resid = np.array([
            _haversine_m(sm_lat[i], sm_lon[i], avg_lat[i], avg_lon[i])
            for i in range(n)
        ])
        rmse_smoothed_m = float(np.sqrt(np.mean(d_resid ** 2)))

        # Consecutive-step smoothness (mean displacement between samples).
        def _step_mean(lat: np.ndarray, lon: np.ndarray) -> float:
            if len(lat) < 2:
                return 0.0
            return float(np.mean([
                _haversine_m(lat[i], lon[i], lat[i + 1], lon[i + 1])
                for i in range(len(lat) - 1)
            ]))

        smoothness_raw = _step_mean(raw_lat, raw_lon)
        smoothness_smoothed = _step_mean(sm_lat, sm_lon)

        return TrackMetrics(
            rmse_raw_m=rmse_raw_m,
            rmse_smoothed_m=rmse_smoothed_m,
            smoothness_raw=smoothness_raw,
            smoothness_smoothed=smoothness_smoothed,
            outliers_rejected=outliers,
        )

    # -------------------------------------------------------------------- plot

    def plot(
        self,
        raw_frames: list[dict],
        smoothed_track: list[dict],
        save_path: Path | str | None = None,
    ) -> None:
        """Scatter raw positions vs the smoothed connected track.

        Imports matplotlib lazily so the rest of the module stays usable
        in headless / minimal-deps environments.
        """
        import matplotlib  # noqa: WPS433
        if save_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433

        raw_lat = [f.get("lat", 0.0) for f in raw_frames]
        raw_lon = [f.get("lon", 0.0) for f in raw_frames]
        sm_lat = [s.get("lat", 0.0) for s in smoothed_track]
        sm_lon = [s.get("lon", 0.0) for s in smoothed_track]

        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax.scatter(raw_lon, raw_lat, s=30, color="gray", alpha=0.6,
                   label=f"Raw frames (n={len(raw_lat)})")
        ax.plot(sm_lon, sm_lat, color="tab:blue", linewidth=2.0,
                marker="o", markersize=3,
                label=f"Smoothed track (n={len(sm_lat)})")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_title("Raw decoded positions vs Kalman-smoothed track")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.legend(loc="best")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
        else:
            plt.show()

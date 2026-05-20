"""Unit tests for `uav_telemetry_pipeline.tracking.kalman_tracker`."""

from __future__ import annotations

import numpy as np
import pytest

from uav_telemetry_pipeline.tracking import KalmanTracker


# ------------------------------------------------------------------- helpers


def _frame(lat: float, lon: float, alt: float = 100.0,
           crc_ok: bool = True, seq: int = 0) -> dict:
    return {
        "lat": lat,
        "lon": lon,
        "altitude_m": alt,
        "crc_ok": crc_ok,
        "sequence_number": seq,
    }


# --------------------------------------------------------------------- tests


def test_single_update_initialises_state_from_first_frame():
    tracker = KalmanTracker()
    out = tracker.update(_frame(51.4463, 7.2672, alt=42.97))

    assert out["lat"] == pytest.approx(51.4463, abs=1e-9)
    assert out["lon"] == pytest.approx(7.2672, abs=1e-9)
    assert out["alt"] == pytest.approx(42.97, abs=1e-9)
    # Velocity components default to zero on bootstrap.
    assert out["v_lat"] == 0.0
    assert out["v_lon"] == 0.0
    assert out["v_alt"] == 0.0
    assert tracker.get_track() == [out]


def test_crc_failed_frame_uses_inflated_measurement_noise():
    """A CRC-bad frame must produce a smaller correction than a CRC-OK
    frame at the same measurement offset, because R_bad >> R_ok shrinks
    the Kalman gain."""
    base = _frame(51.4463, 7.2672)

    # Warm up two filters on identical CRC-OK history to equalise P.
    t_ok = KalmanTracker(); t_bad = KalmanTracker()
    for _ in range(5):
        t_ok.update(base); t_bad.update(base)

    spike = _frame(51.4463 + 0.001, 7.2672)  # ~111 m jump
    out_ok = t_ok.update({**spike, "crc_ok": True})
    out_bad = t_bad.update({**spike, "crc_ok": False})

    delta_ok = abs(out_ok["lat"] - 51.4463)
    delta_bad = abs(out_bad["lat"] - 51.4463)

    assert delta_bad < delta_ok, (
        f"CRC-failed frame should produce a smaller state update than "
        f"a CRC-OK frame (got delta_bad={delta_bad}, delta_ok={delta_ok})"
    )


def test_synthetic_outlier_is_suppressed_below_50_percent_of_spike():
    """After 5 stable observations, a single +0.01 deg lat spike should
    move the smoothed state by less than 50% of the spike magnitude."""
    tracker = KalmanTracker()
    for _ in range(5):
        tracker.update(_frame(51.4463, 7.2672))

    spike_size = 0.01
    out = tracker.update(_frame(51.4463 + spike_size, 7.2672))
    moved = abs(out["lat"] - 51.4463)
    assert moved < 0.5 * spike_size, (
        f"Outlier suppressed by only {moved/spike_size:.0%} of spike; "
        f"expected <50%"
    )


def test_straight_line_trajectory_lowers_position_variance():
    """A noisy straight-line trajectory should come out smoother than
    its raw input (variance of position residuals goes down)."""
    rng = np.random.default_rng(42)
    n = 30
    true_lat = 51.0 + 1e-5 * np.arange(n)  # ~1.1 m / step
    noise = rng.normal(0.0, 1e-4, size=n)  # ~11 m std measurement noise
    raw_lat = true_lat + noise

    tracker = KalmanTracker()
    smoothed_lat = []
    for i in range(n):
        out = tracker.update(_frame(raw_lat[i], 7.2672, alt=100.0))
        smoothed_lat.append(out["lat"])
    smoothed_lat = np.array(smoothed_lat)

    raw_resid = raw_lat - true_lat
    smooth_resid = smoothed_lat - true_lat

    assert smooth_resid.var() < raw_resid.var(), (
        f"Smoothed variance {smooth_resid.var():.3e} not lower than "
        f"raw variance {raw_resid.var():.3e}"
    )


def test_get_track_length_matches_update_count():
    tracker = KalmanTracker()
    for i in range(7):
        tracker.update(_frame(51.0 + i * 1e-5, 7.0 + i * 1e-5))
    assert len(tracker.get_track()) == 7


def test_reset_clears_state_and_track():
    tracker = KalmanTracker()
    tracker.update(_frame(51.4463, 7.2672))
    tracker.update(_frame(51.4464, 7.2672))
    assert len(tracker.get_track()) == 2

    tracker.reset()
    assert tracker.get_track() == []
    # After reset, the next update must bootstrap from a fresh observation.
    out = tracker.update(_frame(48.0, 2.0))
    assert out["lat"] == pytest.approx(48.0, abs=1e-9)
    assert out["lon"] == pytest.approx(2.0, abs=1e-9)


def test_evaluator_metrics_drop_after_smoothing():
    """Sanity check the evaluator: raw smoothness should exceed smoothed
    smoothness on a noisy straight-line trajectory."""
    from uav_telemetry_pipeline.tracking import TrackEvaluator

    rng = np.random.default_rng(1)
    n = 40
    true_lat = 51.0 + 1e-5 * np.arange(n)
    raw_frames = [
        _frame(true_lat[i] + rng.normal(0.0, 1e-4), 7.0, alt=100.0, seq=i)
        for i in range(n)
    ]
    tracker = KalmanTracker()
    for f in raw_frames:
        tracker.update(f)

    metrics = TrackEvaluator().evaluate(raw_frames, tracker.get_track())
    assert metrics["smoothness_raw"] > metrics["smoothness_smoothed"]
    assert metrics["rmse_smoothed_m"] < metrics["rmse_raw_m"]

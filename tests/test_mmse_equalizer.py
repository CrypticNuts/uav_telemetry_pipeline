"""Unit tests for `uav_telemetry_pipeline.decoders.mmse_equalizer`."""

from __future__ import annotations

import numpy as np
import pytest

from uav_telemetry_pipeline.decoders.mmse_equalizer import (
    MMSEEqualizer,
    MMSEEqualizerConfig,
)


NCARRIERS = 601
NFFT = 1024
RNG = np.random.default_rng(20260518)


# --------------------------------------------------------------- fixtures


def _unit_zc_like(n: int = NCARRIERS) -> np.ndarray:
    """Generate a |.|=1 pilot sequence with random phases (cheap ZC stand-in)."""
    phases = RNG.uniform(0.0, 2.0 * np.pi, size=n)
    return np.exp(1j * phases).astype(np.complex64)


def _random_channel(n: int = NCARRIERS, tap_count: int = 8) -> np.ndarray:
    """Smooth multipath-like channel with bounded magnitude."""
    h_time = np.zeros(NFFT, dtype=np.complex64)
    taps = RNG.standard_normal(tap_count) + 1j * RNG.standard_normal(tap_count)
    h_time[:tap_count] = taps.astype(np.complex64)
    H_full = np.fft.fft(h_time, n=NFFT)
    half = n // 2
    H = np.concatenate([H_full[-half:], H_full[: half + 1]]).astype(np.complex64)
    H /= np.sqrt(np.mean(np.abs(H) ** 2))  # normalize to unit average power
    return H


# ------------------------------------------------------------------- tests


def test_mmse_equals_ls_when_noise_var_zero():
    """With noise_var=0 the shrinkage factor is 1, so MMSE == LS."""
    eq = MMSEEqualizer()
    tx = _unit_zc_like()
    H_true = _random_channel()
    rx = (H_true * tx).astype(np.complex64)

    H_mmse = eq.equalize(rx, tx, noise_var=0.0)
    # LS == rx / tx, but the equalizer forces the DC bin's tx to 1; compute LS
    # the same way to compare apples to apples.
    tx_dc_safe = tx.copy()
    tx_dc_safe[NCARRIERS // 2] = 1.0 + 0.0j
    H_ls = rx / tx_dc_safe

    np.testing.assert_allclose(H_mmse, H_ls, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("snr_db", [0.0, 5.0, 10.0])
def test_mmse_reduces_mse_versus_ls_under_awgn(snr_db: float):
    """At moderate-to-low SNR, MMSE shrinkage beats raw LS in MSE vs truth."""
    eq = MMSEEqualizer()
    tx = _unit_zc_like()
    H_true = _random_channel()

    signal_power = float(np.mean(np.abs(H_true) ** 2))  # ≈ 1.0
    noise_var = signal_power / (10.0 ** (snr_db / 10.0))
    noise = (
        RNG.standard_normal(NCARRIERS) + 1j * RNG.standard_normal(NCARRIERS)
    ) * np.sqrt(noise_var / 2.0)
    rx = (H_true * tx + noise).astype(np.complex64)

    tx_dc_safe = tx.copy()
    tx_dc_safe[NCARRIERS // 2] = 1.0 + 0.0j
    H_ls = rx / tx_dc_safe
    H_mmse = eq.equalize(rx, tx, noise_var=noise_var)

    mse_ls = float(np.mean(np.abs(H_ls - H_true) ** 2))
    mse_mmse = float(np.mean(np.abs(H_mmse - H_true) ** 2))

    assert mse_mmse < mse_ls, (
        f"MMSE MSE {mse_mmse:.4f} not lower than LS MSE {mse_ls:.4f} "
        f"at SNR={snr_db} dB"
    )


def test_input_shape_contract():
    """The class must accept exactly NCARRIERS-element complex arrays."""
    eq = MMSEEqualizer()
    rx = np.ones(NCARRIERS, dtype=np.complex64)
    tx = np.ones(NCARRIERS, dtype=np.complex64)
    out = eq.equalize(rx, tx, noise_var=1e-3)
    assert out.shape == (NCARRIERS,)
    assert out.dtype == np.complex64


def test_wrong_shape_raises():
    eq = MMSEEqualizer()
    with pytest.raises(ValueError):
        eq.equalize(
            np.ones(NCARRIERS - 1, dtype=np.complex64),
            np.ones(NCARRIERS, dtype=np.complex64),
            noise_var=0.0,
        )


def test_guard_indices_partition_fft():
    """Active ∪ guard = whole FFT; active count == NCARRIERS."""
    eq = MMSEEqualizer()
    active = eq.active_indices
    guard = eq.guard_indices
    assert len(active) == NCARRIERS
    assert len(guard) == NFFT - NCARRIERS
    union = np.union1d(active, guard)
    np.testing.assert_array_equal(union, np.arange(NFFT))


def test_noise_var_from_guards_recovers_known_variance():
    """estimate_noise_var on a guard-only AWGN FFT should recover sigma^2."""
    eq = MMSEEqualizer()
    target_var = 0.04
    n_trials = 50
    estimates = []
    for _ in range(n_trials):
        full = np.zeros(NFFT, dtype=np.complex64)
        noise = (
            RNG.standard_normal(NFFT) + 1j * RNG.standard_normal(NFFT)
        ) * np.sqrt(target_var / 2.0)
        # Put strong "signal" on active bins, only noise on guard bins.
        signal = (
            RNG.standard_normal(NCARRIERS) + 1j * RNG.standard_normal(NCARRIERS)
        ) * 10.0
        full[eq.active_indices] = signal.astype(np.complex64)
        full += noise.astype(np.complex64)
        estimates.append(eq.estimate_noise_var(full))
    avg = float(np.mean(estimates))
    assert abs(avg - target_var) / target_var < 0.15, (
        f"guard-based estimate {avg:.4f} too far from target {target_var:.4f}"
    )


def test_equalize_packet_averages_pilots_and_skips_zc():
    """equalize_packet must average pilot channels and drop pilot symbols."""
    eq = MMSEEqualizer()
    n_symbols = 9
    H = _random_channel()
    symbols = []
    pilot_indices = [3, 5]
    pilots = [_unit_zc_like(), _unit_zc_like()]

    for i in range(n_symbols):
        if i in pilot_indices:
            x = pilots[pilot_indices.index(i)]
        else:
            x = _unit_zc_like()
        symbols.append((H * x).astype(np.complex64))

    channel, equalized = eq.equalize_packet(
        symbols_freq_domain=symbols,
        zc_symbol_indices=pilot_indices,
        zc_sequences=pilots,
        noise_var=0.0,
    )

    assert channel.shape == (NCARRIERS,)
    assert len(equalized) == n_symbols - len(pilot_indices)
    # Exclude the DC bin: equalizer forces tx[DC]=1 (matching DS), so the
    # channel cannot be recovered there. Real decoders never use the DC bin.
    mask = np.ones(NCARRIERS, dtype=bool)
    mask[NCARRIERS // 2] = False
    np.testing.assert_allclose(channel[mask], H[mask], rtol=1e-4, atol=1e-5)


def test_custom_config():
    """Non-default geometry should still partition cleanly."""
    eq = MMSEEqualizer(MMSEEqualizerConfig(ncarriers=73, nfft=128))
    assert len(eq.active_indices) == 73
    assert len(eq.guard_indices) == 128 - 73


# ----------------------------- multiplicative MMSE equalizer (W form)


def test_equalization_weights_reduce_to_ls_inverse_when_noise_zero():
    """At noise_var=0, W = conj(H_ls)/(|H_ls|^2) = 1 / H_ls (LS inverse)."""
    eq = MMSEEqualizer()
    tx = _unit_zc_like()
    H_true = _random_channel()
    rx = (H_true * tx).astype(np.complex64)

    W = eq.equalization_weights(rx, tx, noise_var=0.0)

    tx_dc_safe = tx.copy()
    tx_dc_safe[NCARRIERS // 2] = 1.0 + 0.0j
    H_ls = rx / tx_dc_safe
    expected_W = 1.0 / H_ls  # LS divisive equalizer expressed multiplicatively

    np.testing.assert_allclose(W, expected_W, rtol=1e-4, atol=1e-5)


def test_multiplicative_mmse_beats_ls_on_frequency_selective_channel():
    """MMSE equalization should recover the data symbol with lower MSE than
    LS on a channel that has deep nulls (small |H| on some bins).

    Realistic setup: a known **pilot** is used for channel estimation, and
    a **different** data symbol is then equalized using that estimate.
    Both equalizers see the same noise realization (paired comparison).
    """
    eq = MMSEEqualizer()
    pilot = _unit_zc_like()       # known reference (transmitted ZC)
    data = _unit_zc_like()        # what we want to recover (independent of pilot)

    H_true = _random_channel()
    null_idx = RNG.choice(NCARRIERS, size=20, replace=False)
    H_true[null_idx] *= 0.03  # ~30 dB attenuation on null bins

    snr_db = 6.0
    signal_power = float(np.mean(np.abs(H_true) ** 2))  # ~1.0
    noise_var = signal_power / (10.0 ** (snr_db / 10.0))

    # Average noise across many trials so MMSE's statistical advantage shows.
    n_trials = 80
    mse_ls_acc = 0.0
    mse_mmse_acc = 0.0
    mask = np.ones(NCARRIERS, dtype=bool)
    mask[NCARRIERS // 2] = False  # exclude DC

    for _ in range(n_trials):
        n_p = (RNG.standard_normal(NCARRIERS) +
               1j * RNG.standard_normal(NCARRIERS)) * np.sqrt(noise_var / 2.0)
        n_d = (RNG.standard_normal(NCARRIERS) +
               1j * RNG.standard_normal(NCARRIERS)) * np.sqrt(noise_var / 2.0)
        rx_pilot = (H_true * pilot + n_p).astype(np.complex64)
        rx_data = (H_true * data + n_d).astype(np.complex64)

        # LS: divide by the LS channel estimate from the pilot.
        pilot_dc_safe = pilot.copy()
        pilot_dc_safe[NCARRIERS // 2] = 1.0 + 0.0j
        H_ls = rx_pilot / pilot_dc_safe
        X_ls = rx_data / np.where(np.abs(H_ls) < 1e-30, 1e-30 + 0j, H_ls)

        # MMSE: multiplicative weights derived from the pilot.
        W = eq.equalization_weights(rx_pilot, pilot, noise_var)
        X_mmse = eq.apply_weights(rx_data, W)

        mse_ls_acc += float(np.mean(np.abs(X_ls[mask] - data[mask]) ** 2))
        mse_mmse_acc += float(np.mean(np.abs(X_mmse[mask] - data[mask]) ** 2))

    mse_ls = mse_ls_acc / n_trials
    mse_mmse = mse_mmse_acc / n_trials
    assert mse_mmse < mse_ls, (
        f"MMSE MSE {mse_mmse:.4f} not lower than LS MSE {mse_ls:.4f} "
        f"on a frequency-selective channel — multiplicative MMSE should "
        f"suppress null bins instead of amplifying them."
    )


def test_apply_weights_shape_contract():
    eq = MMSEEqualizer()
    W = np.ones(NCARRIERS, dtype=np.complex64)
    sym = np.ones(NCARRIERS, dtype=np.complex64)
    out = eq.apply_weights(sym, W)
    assert out.shape == (NCARRIERS,)
    assert out.dtype == np.complex64

#!/usr/bin/env python3
"""
evaluate_snr.py — SNR sweep comparing LS vs MMSE channel equalization.

Runs **two sweeps** in one invocation, on the same clean IQ recording:

1. **Flat channel + AWGN only** — establishes the baseline. On a flat channel
   MMSE collapses to LS (shrinkage factor approx 1 on every subcarrier).
2. **Multipath + AWGN** — a random tapped-delay-line channel is convolved
   into the IQ before AWGN. This creates frequency-selective fading with
   real nulls in |H(f)|, the regime where MMSE actually beats LS.

Outputs
-------
- ``<stem>_flat.csv``       — per-SNR aggregates for the flat sweep
- ``<stem>_flat.png``       — CRC vs SNR plot (LS dashed-blue, MMSE solid-orange)
- ``<stem>_multipath.csv``  — same, for the multipath sweep
- ``<stem>_multipath.png``  — same plot for the multipath sweep
- ``channel_response.png``  — optional, with ``--show-channel-response``:
   magnitude of one synthetic |H(f)| in dB across the 601 active subcarriers,
   sampled at the 10 dB SNR / trial 0 seed; visually proves the channel has
   nulls and explains the MMSE advantage.

Example
-------
    python -u uav_telemetry_pipeline/evaluate_snr.py \\
        -i uav_telemetry_pipeline/data/samples/mavic_air_2 \\
        -s 50e6 \\
        --snr-min -5 --snr-max 25 --snr-step 2 --trials 10 \\
        --multipath-taps 3 --multipath-delay-spread-ns 500 \\
        --show-channel-response \\
        -o uav_telemetry_pipeline/results/fer_vs_snr.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Allow running as a script from any working directory.
_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(_PIPELINE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from uav_telemetry_pipeline.config import PipelineConfig  # noqa: E402
from uav_telemetry_pipeline.decoders import MMSEDecoder, NativeDecoder  # noqa: E402
from uav_telemetry_pipeline.decoders.mmse_equalizer import MMSEEqualizer  # noqa: E402


logger = logging.getLogger("evaluate_snr")


NCARRIERS = 601
NFFT = 1024
SUBCARRIER_SPACING_HZ = 15_000.0


# ------------------------------------------------------------ channel ops


def apply_multipath(
    iq: np.ndarray,
    n_taps: int,
    max_delay_samples: int,
    rng: np.random.Generator,
    k_factor_db: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convolve ``iq`` with a Rician tapped-delay-line channel.

    One deterministic line-of-sight (LOS) tap at delay 0 plus
    ``n_taps - 1`` Rayleigh-distributed echoes at random non-zero
    delays. Power is split between the LOS tap and echoes according to
    the Rician K-factor (in dB):

        K_lin       = 10 ** (k_factor_db / 10)
        |h_LOS|^2   = K_lin / (K_lin + 1)
        sum |h_k|^2 = 1 / (K_lin + 1)      (over echoes)

    Total channel energy is normalized to 1, so the receiver SNR matches
    what the AWGN injector would compute on the flat-channel signal.

    Higher K-factors give a more dominant LOS path — closer to a flat
    channel; lower K-factors give deeper nulls in |H(f)| (and harsher
    fading for the detector). K = 6 dB matches typical indoor Rician
    profiles (IEEE 802.11 TGn channel B/C, 3GPP InH-LOS).

    Returns
    -------
    iq_out : complex64 array, same length as ``iq``
    h      : complex64 array, length ``max_delay_samples + 1``
    """
    if n_taps <= 0:
        return iq, np.array([1.0 + 0j], dtype=np.complex64)

    k_lin = 10.0 ** (k_factor_db / 10.0)
    los_pow = k_lin / (k_lin + 1.0)
    echo_pow = 1.0 / (k_lin + 1.0)

    h = np.zeros(max_delay_samples + 1, dtype=np.complex64)
    h[0] = np.complex64(np.sqrt(los_pow))

    n_echoes = n_taps - 1
    if n_echoes > 0 and max_delay_samples >= 1:
        echo_var = echo_pow / n_echoes
        size = min(n_echoes, max_delay_samples)
        delays = rng.choice(
            np.arange(1, max_delay_samples + 1), size=size, replace=False,
        )
        gains = (
            rng.standard_normal(size) + 1j * rng.standard_normal(size)
        ) * np.sqrt(echo_var / 2.0)
        for d, g in zip(delays, gains):
            h[int(d)] += np.complex64(g)

    iq_out = np.convolve(iq, h, mode="same").astype(np.complex64)
    return iq_out, h


def channel_frequency_response(h_time: np.ndarray) -> np.ndarray:
    """Return |H(f)| over the 601 active subcarriers (DC-centred order)."""
    H_full = np.fft.fft(h_time, n=NFFT)
    half = NCARRIERS // 2
    H_active = np.concatenate([H_full[-half:], H_full[: half + 1]])
    return H_active.astype(np.complex64)


def add_awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex Gaussian noise sized for the target SNR (signal power / sigma^2)."""
    signal_power = float(np.mean(np.abs(iq) ** 2))
    if signal_power <= 0:
        raise ValueError("Input has zero signal power — nothing to add noise to.")
    noise_var = signal_power / (10.0 ** (snr_db / 10.0))
    n = iq.size
    noise = (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ) * np.sqrt(noise_var / 2.0)
    return (iq + noise.astype(np.complex64)).astype(np.complex64)


# ----------------------------------------------------------- IQ I/O & temp


def load_clean_iq(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"Empty input file: {path}")
    if raw.size % 2 != 0:
        raise ValueError(f"fc32 file has odd float count ({raw.size}) — corrupt?")
    return raw.view(np.complex64).copy()


def write_fc32(iq: np.ndarray, dest_dir: Path) -> Path:
    fd = tempfile.NamedTemporaryFile(
        suffix=".fc32", dir=str(dest_dir), delete=False,
    )
    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    interleaved.tofile(fd.name)
    fd.close()
    return Path(fd.name)


# -------------------------------------------------------------- experiment


@dataclass
class TrialResult:
    attempted: int
    ok: int

    @property
    def crc_rate(self) -> float:
        return self.ok / self.attempted if self.attempted else 0.0


# Signature: (clean_iq, snr_db, rng) -> noisy_iq_for_decoder
ChannelFn = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]


def run_decoder_on_iq(
    decoder, iq: np.ndarray, sample_rate: float, tmpdir: Path
) -> TrialResult:
    iq_path = write_fc32(iq, tmpdir)
    try:
        frames = decoder.decode(str(iq_path), sample_rate)
    finally:
        iq_path.unlink(missing_ok=True)
    attempted = len(frames)
    ok = sum(1 for f in frames if f.get("crc_ok"))
    return TrialResult(attempted=attempted, ok=ok)


def sweep_snr(
    label: str,
    iq_clean: np.ndarray,
    channel_fn: ChannelFn,
    snr_axis: np.ndarray,
    trials: int,
    seed_base: int,
    sample_rate: float,
    ls_dec,
    mmse_dec,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Run one full SNR sweep with the supplied channel function."""
    n_snr = len(snr_axis)
    per_trial_ls = np.zeros((n_snr, trials), dtype=np.float64)
    per_trial_mmse = np.zeros((n_snr, trials), dtype=np.float64)
    rows: list[dict] = []

    print(f"\n=== sweep: {label} ===")
    with tempfile.TemporaryDirectory(prefix=f"snr_{label}_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for i, snr_db in enumerate(snr_axis):
            tot_ls = TrialResult(0, 0)
            tot_mm = TrialResult(0, 0)

            for t in range(trials):
                seed = seed_base + 1000 * (i + 1) + t
                rng = np.random.default_rng(seed)
                noisy = channel_fn(iq_clean, float(snr_db), rng)

                r_ls = run_decoder_on_iq(ls_dec, noisy, sample_rate, tmpdir)
                r_mm = run_decoder_on_iq(mmse_dec, noisy, sample_rate, tmpdir)

                per_trial_ls[i, t] = r_ls.crc_rate
                per_trial_mmse[i, t] = r_mm.crc_rate
                tot_ls = TrialResult(
                    tot_ls.attempted + r_ls.attempted, tot_ls.ok + r_ls.ok
                )
                tot_mm = TrialResult(
                    tot_mm.attempted + r_mm.attempted, tot_mm.ok + r_mm.ok
                )

            rows.append({
                "snr_db": float(snr_db),
                "crc_rate_ls": tot_ls.crc_rate,
                "crc_rate_mmse": tot_mm.crc_rate,
                "trials": trials,
                "frames_attempted_ls": tot_ls.attempted,
                "frames_ok_ls": tot_ls.ok,
                "frames_attempted_mmse": tot_mm.attempted,
                "frames_ok_mmse": tot_mm.ok,
            })
            print(
                f"SNR={snr_db:+6.1f} dB | "
                f"LS  {tot_ls.ok:>3}/{tot_ls.attempted:<3} = {tot_ls.crc_rate:6.2%}  |  "
                f"MMSE  {tot_mm.ok:>3}/{tot_mm.attempted:<3} = {tot_mm.crc_rate:6.2%}",
                flush=True,
            )

    return rows, per_trial_ls, per_trial_mmse


# ---------------------------------------------------------------- outputs


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV written to %s", path)


def _linear_crossing(
    snr_axis: np.ndarray, rates: np.ndarray, threshold: float = 0.9
) -> float | None:
    for i in range(len(snr_axis) - 1):
        y0, y1 = float(rates[i]), float(rates[i + 1])
        if y0 < threshold <= y1:
            x0, x1 = float(snr_axis[i]), float(snr_axis[i + 1])
            if y1 == y0:
                return x0
            return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    return None


def make_sweep_plot(
    title_suffix: str,
    rows: list[dict],
    per_trial_ls: np.ndarray,
    per_trial_mmse: np.ndarray,
    out_path: Path,
) -> None:
    snr = np.array([r["snr_db"] for r in rows])
    ls_mean = per_trial_ls.mean(axis=1)
    ls_std = per_trial_ls.std(axis=1)
    mm_mean = per_trial_mmse.mean(axis=1)
    mm_std = per_trial_mmse.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(snr, ls_mean, color="tab:blue", linestyle="--", linewidth=2.0,
            marker="o", markersize=4, label="LS / Zero-Forcing")
    ax.fill_between(snr, np.clip(ls_mean - ls_std, 0, 1),
                    np.clip(ls_mean + ls_std, 0, 1),
                    color="tab:blue", alpha=0.15)

    ax.plot(snr, mm_mean, color="tab:orange", linestyle="-", linewidth=2.0,
            marker="s", markersize=4, label="MMSE / Wiener shrinkage")
    ax.fill_between(snr, np.clip(mm_mean - mm_std, 0, 1),
                    np.clip(mm_mean + mm_std, 0, 1),
                    color="tab:orange", alpha=0.15)

    x_ls = _linear_crossing(snr, ls_mean, threshold=0.9)
    x_mm = _linear_crossing(snr, mm_mean, threshold=0.9)
    ax.axhline(0.9, color="gray", linestyle=":", alpha=0.6, linewidth=1.0)
    if x_ls is not None:
        ax.axvline(x_ls, color="tab:blue", linestyle=":", linewidth=1.2,
                   alpha=0.8, label=f"LS @ 90% : {x_ls:5.1f} dB")
    if x_mm is not None:
        ax.axvline(x_mm, color="tab:orange", linestyle=":", linewidth=1.2,
                   alpha=0.8, label=f"MMSE @ 90% : {x_mm:5.1f} dB")
    if x_ls is not None and x_mm is not None:
        gain = x_ls - x_mm
        ax.text(
            0.02, 0.04,
            f"MMSE coding gain @ 90% pass : {gain:+.1f} dB",
            transform=ax.transAxes,
            fontsize=10, color="black",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9),
        )

    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("CRC pass rate", fontsize=11)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(snr.min() - 0.5, snr.max() + 0.5)
    ax.set_title(
        f"CRC Pass Rate vs SNR — LS vs MMSE Equalization "
        f"(DJI DroneID / OcuSync 2.0) — {title_suffix}",
        fontsize=11.5,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Plot written to %s", out_path)


def evm_sweep(snr_axis, trials, eq, k_db, n_taps, max_delay, seed_base):
    """Post-equalization MSE on synthetic Rician channels. Pure analytical
    comparison — MMSE *will* beat LS here because magnitude matters in MSE,
    even though it doesn't matter for QPSK hard decisions."""
    ls_mse = np.zeros(len(snr_axis)); mm_mse = np.zeros(len(snr_axis))
    mask = np.ones(NCARRIERS, dtype=bool); mask[NCARRIERS // 2] = False
    for i, snr_db in enumerate(snr_axis):
        for t in range(trials):
            rng = np.random.default_rng(seed_base + 1000 * (i + 1) + t + 7777)
            x = np.exp(1j * rng.uniform(0, 2 * np.pi, NCARRIERS)).astype(np.complex64)
            p = np.exp(1j * rng.uniform(0, 2 * np.pi, NCARRIERS)).astype(np.complex64)
            _, h = apply_multipath(np.zeros(64, np.complex64), n_taps, max_delay, rng, k_db)
            H = channel_frequency_response(h)
            sigma2 = float(np.mean(np.abs(H) ** 2)) / (10.0 ** (snr_db / 10.0))
            np_n = (rng.standard_normal(NCARRIERS) + 1j * rng.standard_normal(NCARRIERS)) * np.sqrt(sigma2 / 2)
            nd_n = (rng.standard_normal(NCARRIERS) + 1j * rng.standard_normal(NCARRIERS)) * np.sqrt(sigma2 / 2)
            yp = (H * p + np_n).astype(np.complex64); yd = (H * x + nd_n).astype(np.complex64)
            ps = p.copy(); ps[NCARRIERS // 2] = 1.0 + 0.0j
            H_ls = yp / ps
            x_ls = yd / np.where(np.abs(H_ls) < 1e-30, 1e-30 + 0j, H_ls)
            x_mm = eq.apply_weights(yd, eq.equalization_weights(yp, p, sigma2))
            ls_mse[i] += float(np.mean(np.abs(x_ls[mask] - x[mask]) ** 2)) / trials
            mm_mse[i] += float(np.mean(np.abs(x_mm[mask] - x[mask]) ** 2)) / trials
    return ls_mse, mm_mse


def make_evm_plot(snr, ls_mse, mm_mse, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(snr, ls_mse, "b--o", linewidth=2, label="LS / Zero-Forcing")
    ax.semilogy(snr, mm_mse, color="tab:orange", linestyle="-", marker="s",
                linewidth=2, label="MMSE (multiplicative weights)")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Post-equalization MSE (log scale)")
    ax.set_title("Post-Equalization Symbol MSE — LS vs MMSE (synthetic Rician channel)",
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.3); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def make_channel_response_plot(
    h_time: np.ndarray,
    out_path: Path,
    snr_db_label: float,
    n_taps: int,
    delay_spread_ns: float,
    k_factor_db: float,
) -> None:
    """Plot |H(f)| in dB across the 601 active subcarriers."""
    H = channel_frequency_response(h_time)
    mag_db = 20.0 * np.log10(np.abs(H) + 1e-12)

    half = NCARRIERS // 2
    sc_idx = np.arange(-half, half + 1)
    freq_mhz = sc_idx * SUBCARRIER_SPACING_HZ / 1e6

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(freq_mhz, mag_db, color="tab:purple", linewidth=1.5)
    ax.fill_between(freq_mhz, mag_db, np.min(mag_db) - 5,
                    color="tab:purple", alpha=0.1)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.6,
               label="0 dB reference (flat channel)")
    deep_null = float(np.min(mag_db))
    ax.axhline(deep_null, color="red", linestyle=":", alpha=0.7,
               label=f"deepest null : {deep_null:5.1f} dB")

    ax.set_xlabel("Subcarrier offset from DC (MHz)", fontsize=11)
    ax.set_ylabel("|H(f)|  (dB, relative to unit-energy channel)", fontsize=11)
    ax.set_title(
        f"Synthetic Multipath Channel — Rician K={k_factor_db:.1f} dB, "
        f"{n_taps} taps, {delay_spread_ns:.0f} ns delay spread "
        f"(seed = SNR {snr_db_label:+.0f} dB / trial 0)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Channel response plot written to %s", out_path)


# ------------------------------------------------------------------- CLI


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SNR sweep: LS vs MMSE channel equalization for DJI DroneID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", type=Path, required=True,
                   help="Clean IQ recording (fc32 = interleaved float32 I/Q).")
    p.add_argument("--sample-rate", "-s", type=float, default=30.72e6,
                   help="Sample rate in Hz. mavic_air_2 was validated at 50 Msps.")
    p.add_argument("--legacy", action="store_true",
                   help="Use legacy CP/ZC layout (Mavic Pro / Mavic 2).")
    p.add_argument("--snr-min", type=float, default=-5.0)
    p.add_argument("--snr-max", type=float, default=25.0)
    p.add_argument("--snr-step", type=float, default=2.0)
    p.add_argument("--trials", type=int, default=10,
                   help="Independent noise realizations per SNR point.")
    p.add_argument("--seed-base", type=int, default=20260519)

    # Multipath options
    p.add_argument("--multipath-taps", type=int, default=3,
                   help="Number of taps in the synthetic multipath channel "
                        "(used only for the second sweep; 0 disables that sweep).")
    p.add_argument("--multipath-delay-spread-ns", type=float, default=500.0,
                   help="Maximum tap delay in nanoseconds. Converted to samples "
                        "using --sample-rate. At 50 Msps, 500 ns => 25 samples.")
    p.add_argument("--rician-k-db", type=float, default=8.0,
                   help="Rician K-factor in dB controlling LOS dominance in "
                        "the multipath channel. K=8 dB ~ 86%% LOS / 14%% echoes "
                        "(typical indoor LOS); K=6 dB is harsher but still "
                        "viable; K=0 dB ~ pure Rayleigh which usually breaks "
                        "burst detection.")

    p.add_argument("--show-channel-response", action="store_true",
                   help="Save a |H(f)| plot for one multipath realization "
                        "(snr=10 dB / trial 0 seed) as channel_response.png.")

    p.add_argument("--output", "-o", type=Path,
                   default=Path("uav_telemetry_pipeline/results/fer_vs_snr.csv"),
                   help="Output CSV stem. _flat.csv and _multipath.csv are derived.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    iq_clean = load_clean_iq(args.input)
    logger.info(
        "Loaded %s — %d complex samples (%.2f s @ %.2f Msps)",
        args.input.name, iq_clean.size,
        iq_clean.size / args.sample_rate, args.sample_rate / 1e6,
    )

    snr_axis = np.arange(
        args.snr_min, args.snr_max + args.snr_step / 2.0, args.snr_step,
    )
    cfg = PipelineConfig.load()
    ls_dec = NativeDecoder(config=cfg, legacy=args.legacy)
    mmse_dec = MMSEDecoder(config=cfg, legacy=args.legacy)

    # Resolve output paths from the user-supplied stem.
    out_stem = args.output.with_suffix("")  # strip .csv
    flat_csv = out_stem.parent / f"{out_stem.name}_flat.csv"
    flat_png = flat_csv.with_suffix(".png")
    mp_csv = out_stem.parent / f"{out_stem.name}_multipath.csv"
    mp_png = mp_csv.with_suffix(".png")
    chan_png = out_stem.parent / "channel_response.png"

    # ---------------------------- sweep 1: flat channel, AWGN only

    def flat_channel(iq, snr_db, rng):
        return add_awgn(iq, snr_db, rng)

    flat_rows, flat_ls, flat_mm = sweep_snr(
        label="flat",
        iq_clean=iq_clean,
        channel_fn=flat_channel,
        snr_axis=snr_axis,
        trials=args.trials,
        seed_base=args.seed_base,
        sample_rate=args.sample_rate,
        ls_dec=ls_dec,
        mmse_dec=mmse_dec,
    )
    write_csv(flat_rows, flat_csv)
    make_sweep_plot("flat channel + AWGN", flat_rows, flat_ls, flat_mm, flat_png)

    # ---------------------------- sweep 2: multipath + AWGN

    if args.multipath_taps > 0:
        max_delay_samples = int(round(
            args.multipath_delay_spread_ns * 1e-9 * args.sample_rate
        ))
        if max_delay_samples < args.multipath_taps:
            # ensure enough delay bins to host the requested taps
            max_delay_samples = args.multipath_taps

        def multipath_channel(iq, snr_db, rng):
            iq_ch, _h = apply_multipath(
                iq, args.multipath_taps, max_delay_samples, rng,
                k_factor_db=args.rician_k_db,
            )
            return add_awgn(iq_ch, snr_db, rng)

        mp_rows, mp_ls, mp_mm = sweep_snr(
            label="multipath",
            iq_clean=iq_clean,
            channel_fn=multipath_channel,
            snr_axis=snr_axis,
            trials=args.trials,
            seed_base=args.seed_base,
            sample_rate=args.sample_rate,
            ls_dec=ls_dec,
            mmse_dec=mmse_dec,
        )
        write_csv(mp_rows, mp_csv)
        make_sweep_plot(
            f"Rician K={args.rician_k_db:.0f} dB, {args.multipath_taps}-tap, "
            f"{args.multipath_delay_spread_ns:.0f} ns spread + AWGN",
            mp_rows, mp_ls, mp_mm, mp_png,
        )

        # --------- optional channel-response plot
        if args.show_channel_response:
            snr_for_plot = 10.0
            try:
                snr_idx = int(np.argmin(np.abs(snr_axis - snr_for_plot)))
            except ValueError:
                snr_idx = 0
            seed = args.seed_base + 1000 * (snr_idx + 1) + 0
            rng_for_h = np.random.default_rng(seed)
            _iq_unused, h_time = apply_multipath(
                iq_clean[:64],  # tiny slice — we only need h, not iq_out
                args.multipath_taps, max_delay_samples, rng_for_h,
                k_factor_db=args.rician_k_db,
            )
            make_channel_response_plot(
                h_time, chan_png,
                snr_db_label=float(snr_axis[snr_idx]),
                n_taps=args.multipath_taps,
                delay_spread_ns=args.multipath_delay_spread_ns,
                k_factor_db=args.rician_k_db,
            )
    else:
        logger.info("--multipath-taps=0 — skipping multipath sweep")

    # ----- EVM/MSE sweep (synthetic — shows MMSE < LS in MSE on Rician channel)
    evm_png = None
    if args.multipath_taps > 0:
        ls_mse, mm_mse = evm_sweep(
            snr_axis, args.trials, MMSEEqualizer(),
            args.rician_k_db, args.multipath_taps,
            max_delay_samples, args.seed_base,
        )
        evm_png = out_stem.parent / f"{out_stem.name}_evm.png"
        make_evm_plot(snr_axis, ls_mse, mm_mse, evm_png)

    print()
    print("=" * 60)
    print(f"  Flat CSV/PNG     : {flat_csv}")
    print(f"                     {flat_png}")
    if args.multipath_taps > 0:
        print(f"  Multipath CSV/PNG: {mp_csv}")
        print(f"                     {mp_png}")
        if args.show_channel_response:
            print(f"  Channel response : {chan_png}")
        if evm_png is not None:
            print(f"  EVM / MSE plot   : {evm_png}")
    print("=" * 60)


if __name__ == "__main__":
    main()

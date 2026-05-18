"""Summarize all DIAG + SPEC JSONs under results/recording3/."""

import glob
import json
import os

out = "/home/kali/projects/PFE/uav_telemetry_pipeline/results/recording3"

print("=" * 110)
print("DIAGNOSE SUMMARY (ZC correlation; peaks_above counts peaks >= 0.15)")
print("=" * 110)
print(f"{'file':<70} {'best':>6} {'peaks':>6}  decoder")
print("-" * 110)
for p in sorted(glob.glob(f"{out}/*_DIAG.json")):
    d = json.load(open(p))
    diag = d.get("diagnostic") or {}
    name = os.path.basename(p).replace("_DIAG.json", "")
    print(
        f"{name:<70} "
        f"{diag.get('best_score', 0):>6.3f} "
        f"{diag.get('peaks_above_threshold', 0):>6}  "
        f"{d.get('decoder_used') or '-'}"
    )

print()
print("=" * 110)
print("SPECTRUM SUMMARY (band labels detected per file)")
print("=" * 110)
print(f"{'file':<70}  band labels")
print("-" * 110)
for p in sorted(glob.glob(f"{out}/*_SPEC.json")):
    d = json.load(open(p))
    labels: dict[str, int] = {}
    for ch in d.get("spectrum", []):
        for b in ch["bands"]:
            labels[b["label"]] = labels.get(b["label"], 0) + 1
    name = os.path.basename(p).replace("_SPEC.json", "")
    print(f"{name:<70}  {labels}")

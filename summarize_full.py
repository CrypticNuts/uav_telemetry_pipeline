"""Summarize all *_FULL.json files under results/recording3/."""

import glob
import json
import os

out = "/home/kali/projects/PFE/uav_telemetry_pipeline/results/recording3"

print("=" * 100)
print("FULL-PIPELINE SUMMARY (decoders attempted in order)")
print("=" * 100)
print(
    f"{'file':<40} {'frames':>7} {'crc_ok':>7} "
    f"{'decoder':>14}  attempts"
)
print("-" * 100)
for p in sorted(glob.glob(f"{out}/*_FULL.json")):
    d = json.load(open(p))
    name = os.path.basename(p).replace("_FULL.json", "")
    atts = ",".join(
        f"{a['decoder']}({a['frames']}/{a['crc_ok']})"
        for a in d.get("attempts", [])
    )
    print(
        f"{name:<40} {d['frames_decoded']:>7} "
        f"{d['crc_ok_frames']:>7} {str(d['decoder_used']):>14}  {atts}"
    )

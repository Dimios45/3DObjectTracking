"""Merge per-object JSON result files into one combined results file."""
import json, sys, glob
from pathlib import Path

inputs = sys.argv[1:-1]
output = sys.argv[-1]

merged = {}
for path in inputs:
    with open(path) as f:
        data = json.load(f)
    for k, v in data.items():
        if k != "average":
            merged[k] = v

if merged:
    merged["average"] = sum(merged.values()) / len(merged)

with open(output, "w") as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(merged)-1} objects → {output}")
for k, v in merged.items():
    print(f"  {k:12s}: {v:.3f}")

# CoTracker / LocoTrack OPT Evaluation

Custom scripts and results for comparing 2D point trackers (CoTracker2, CoTracker3,
LocoTrack) against the 3D trackers in this repo (M3T/ICG/SRT3D/RBGT) on the OPT dataset.

These files were developed inside a local clone of
[facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker) and are
salvaged here because that clone is a separate git repo and its contents are not
tracked by this one. To re-run the evaluations, place the scripts at the root of a
co-tracker clone (the scripts import the `cotracker` package relatively).

## Contents

- `evaluate_cotracker_opt.py` / `evaluate_cotracker3_opt.py` — CoTracker2/3 evaluation on OPT
- `evaluate_locotrack_opt.py` — LocoTrack evaluation on OPT (needs `locotrack_pytorch/`)
- `merge_results.py` — merge per-GPU / per-object result JSONs
- `download_opt.py` — OPT dataset downloader
- `run_pear_tracking.py` — pear demo tracking (see `assets/`)
- `locotrack_pytorch/` — LocoTrack PyTorch implementation used by the eval script
- `*.json` — evaluation results (`ct3_gpu*` = CoTracker3 per-GPU shards, `lt_*` = LocoTrack per-object)
- `*.log` — evaluation run logs
- `assets/`, `saved_videos/` — pear demo input recording, masks, and result frames/videos

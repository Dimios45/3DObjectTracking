"""
LocoTrack evaluation on the OPT dataset (mask-based, no mesh required).

Pipeline mirrors evaluate_cotracker_opt.py:
  1. At frame 0, sample N query points from the GT object mask
  2. Run LocoTrack offline on the full sequence
  3. For each frame, check what fraction of visible tracks land inside the GT mask
  4. Report Track Success Rate (TSR) averaged over all sequences

API differences vs CoTracker2:
  - Video:   (B, T, H, W, C) float32  in [-1, 1]   (CoTracker: (B,T,C,H,W) [0-255])
  - Queries: (B, N, 3) as (t, y, x)                (CoTracker: (frame, x, y))
  - Tracks:  output['tracks'] (B, N, T, 2) (x,y)   (CoTracker: (B,T,N,2))
  - Vis:     output['occlusion'] logits → sigmoid   (CoTracker: probability)

Usage:
  conda run -n graspmas python3 evaluate_locotrack_opt.py \
      --dataset /mnt/data/mritunjoyh/datasets/opt \
      --output ./locotrack_opt_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# LocoTrack source lives in locotrack_pytorch/ next to this script
sys.path.insert(0, str(Path(__file__).parent / "locotrack_pytorch"))
from models.locotrack_model import load_model

INTRINSICS = dict(w=512, h=424)

BODY_NAMES   = ["soda", "chest", "ironman", "house", "bike", "jet"]
BODY_SHORTS  = {"soda": "so", "chest": "ch", "ironman": "ir",
                "house": "ho", "bike": "bi", "jet": "je"}
ORIENTATIONS = ["b", "f", "l", "r"]
MOTION_PATTERNS = [
    "tr_1", "tr_2", "tr_3", "tr_4", "tr_5",
    "zo_1", "zo_2", "zo_3", "zo_4", "zo_5",
    "ir_1", "ir_2", "ir_3", "ir_4", "ir_5",
    "or_1", "or_2", "or_3", "or_4", "or_5",
    "fl", "ml", "fm",
]

N_QUERY_PTS = 200


# ── Image / mask helpers ──────────────────────────────────────────────────────

def load_frames_locotrack(frame_dir: Path) -> torch.Tensor:
    """Load color frames as (1, T, H, W, 3) float32 tensor in [-1, 1]."""
    paths = sorted(frame_dir.glob("*.png")) + sorted(frame_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No images in {frame_dir}")
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    arr = np.stack(frames, axis=0).astype(np.float32)   # T×H×W×3  [0,255]
    arr = arr / 255.0 * 2.0 - 1.0                       # → [-1, 1]
    t = torch.from_numpy(arr)                            # T×H×W×3
    return t.unsqueeze(0)                                # 1×T×H×W×3


def load_mask(mask_path: Path):
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return (m > 127) if m is not None else None


def sample_mask_points(mask: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Return N (x, y) pixel positions sampled from the object mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xs), size=min(n, len(xs)))
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)   # N×2 (x,y)


def mask_paths_for_seq(seq_dir: Path) -> list:
    mask_dir = seq_dir / "mask"
    if not mask_dir.exists():
        return []
    return sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg"))


# ── LocoTrack inference ───────────────────────────────────────────────────────

_locotrack_model = None   # module-level cache

def get_locotrack(device: str, model_size: str = "base"):
    global _locotrack_model
    if _locotrack_model is None:
        print(f"Loading LocoTrack-{model_size} weights...", flush=True)
        _locotrack_model = load_model(model_size=model_size).to(device).eval()
    return _locotrack_model


def run_locotrack(video: torch.Tensor, queries_xy: np.ndarray,
                  device: str, model_size: str = "base") -> tuple:
    """
    Run LocoTrack on the video.

    video:      (1, T, H, W, 3) float32 in [-1, 1]
    queries_xy: N×2  (x, y) pixel positions at frame 0

    Returns:
        tracks:     (T, N, 2) float32 numpy  (x, y) in original pixel coords
        visibility: (T, N) bool numpy  True = visible
    """
    model = get_locotrack(device, model_size)
    H, W = video.shape[2], video.shape[3]
    N = len(queries_xy)

    # Build query tensor: (1, N, 3) as (t, y, x) — LocoTrack convention
    q = np.zeros((N, 3), dtype=np.float32)
    q[:, 0] = 0.0                    # frame 0
    q[:, 1] = queries_xy[:, 1]       # y
    q[:, 2] = queries_xy[:, 0]       # x
    q_t = torch.from_numpy(q).unsqueeze(0).to(device)   # 1×N×3

    video = video.to(device)
    with torch.no_grad():
        out = model(video, q_t)

    # tracks: (1, N, T, 2)  → (T, N, 2)
    tracks = out['tracks'][0].permute(1, 0, 2).cpu().numpy()   # T×N×2

    # occlusion: (1, N, T)  sigmoid → threshold → flip (occluded → NOT visible)
    occ = torch.sigmoid(out['occlusion'][0])                   # N×T
    vis = (occ < 0.5).permute(1, 0).cpu().numpy()              # T×N  bool

    return tracks.astype(np.float32), vis


# ── Track Success Rate ────────────────────────────────────────────────────────

def track_success_rate(tracks: np.ndarray, vis: np.ndarray,
                       mask_paths: list, start_idx: int = 1) -> list:
    H, W = INTRINSICS["h"], INTRINSICS["w"]
    tsr_values = []
    for t in range(start_idx, len(tracks)):
        if t >= len(mask_paths):
            break
        mask = load_mask(mask_paths[t])
        if mask is None:
            continue
        pts = tracks[t]        # N×2 (x, y)
        v   = vis[t]           # N bool
        if v.sum() == 0:
            tsr_values.append(0.0)
            continue
        xs = np.clip(pts[v, 0].astype(int), 0, W - 1)
        ys = np.clip(pts[v, 1].astype(int), 0, H - 1)
        tsr_values.append(float(mask[ys, xs].mean()))
    return tsr_values


# ── Per-sequence evaluation ───────────────────────────────────────────────────

def evaluate_sequence(seq_name: str, dataset_dir: Path,
                      device: str, model_size: str) -> list:
    seq_dir   = dataset_dir / "3D" / seq_name
    color_dir = seq_dir / "color"
    if not color_dir.exists():
        print(f"  [skip] no color dir for {seq_name}")
        return []

    mask_paths = mask_paths_for_seq(seq_dir)
    if not mask_paths:
        print(f"  [skip] no mask dir for {seq_name}")
        return []

    mask0 = load_mask(mask_paths[0])
    if mask0 is None or mask0.sum() == 0:
        print(f"  [skip] empty mask at frame 0 for {seq_name}")
        return []

    queries = sample_mask_points(mask0, N_QUERY_PTS)
    if len(queries) < 4:
        return []

    try:
        video = load_frames_locotrack(color_dir)   # 1×T×H×W×3
    except Exception as e:
        print(f"  [skip] {seq_name}: {e}")
        return []

    T_frames = video.shape[1]
    n_masks  = len(mask_paths)
    n_eval   = min(T_frames, n_masks)

    tracks, vis = run_locotrack(video[:, :n_eval], queries, device, model_size)
    return track_success_rate(tracks, vis, mask_paths, start_idx=1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--output",     default="./locotrack_opt_results.json")
    parser.add_argument("--model_size", default="base", choices=["small", "base"])
    parser.add_argument("--bodies",       nargs="+", default=BODY_NAMES)
    parser.add_argument("--orientations", nargs="+", default=ORIENTATIONS)
    parser.add_argument("--patterns",     nargs="+", default=MOTION_PATTERNS)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {}

    for body_name in args.bodies:
        short = BODY_SHORTS[body_name]
        print(f"\n{'='*60}\nObject: {body_name}")
        body_tsr = []
        for orientation in args.orientations:
            for pattern in args.patterns:
                seq_name = f"{short}_{pattern}_{orientation}"
                print(f"  {seq_name}", end=" ... ", flush=True)
                t0 = time.time()
                tsr = evaluate_sequence(seq_name, dataset_dir, device, args.model_size)
                elapsed = time.time() - t0
                if tsr:
                    avg = float(np.mean(tsr))
                    body_tsr.extend(tsr)
                    print(f"TSR={avg:.3f}  frames={len(tsr)}  t={elapsed:.1f}s")
                else:
                    print("skipped")

        if body_tsr:
            body_avg = float(np.mean(body_tsr))
            all_results[body_name] = body_avg
            print(f"  → {body_name} average TSR = {body_avg:.3f}")

    if all_results:
        overall = float(np.mean(list(all_results.values())))
        all_results["average"] = overall
        print(f"\n{'='*60}\nFinal Track Success Rate:")
        for k, v in all_results.items():
            print(f"  {k:12s}: {v:.3f}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

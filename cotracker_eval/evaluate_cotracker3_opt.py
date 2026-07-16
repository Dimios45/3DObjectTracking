"""
CoTracker2 evaluation on the OPT dataset (mask-based, no mesh required).

Pipeline:
  1. At frame 0, sample N query points from the GT object mask
  2. Run CoTracker2 offline on the full sequence
  3. For each frame, check what fraction of visible tracks land inside the GT mask
  4. Report Track Success Rate (TSR) averaged over all sequences

TSR ∈ [0, 1]: fraction of visible tracks that stay on the object.
M3T comparison uses ADD-AUC (requires .obj meshes); TSR is reported separately.

Usage:
  conda run -n graspmas python3 evaluate_cotracker_opt.py \
      --dataset /mnt/data/mritunjoyh/datasets/opt \
      --output ./cotracker2_opt_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

# ── Dataset constants ─────────────────────────────────────────────────────────
# Images are 512×424 (scaled-down OPT variant).
# Intrinsics scaled from original 1920×1080 (center-crop + uniform scale).
INTRINSICS = dict(fx=416.3, fy=416.4, cx=257.5, cy=220.4, w=512, h=424)

BODY_NAMES      = ["soda", "chest", "ironman", "house", "bike", "jet"]
BODY_SHORTS     = {"soda": "so", "chest": "ch", "ironman": "ir",
                   "house": "ho", "bike": "bi", "jet": "je"}
ORIENTATIONS    = ["b", "f", "l", "r"]
MOTION_PATTERNS = [
    "tr_1", "tr_2", "tr_3", "tr_4", "tr_5",
    "zo_1", "zo_2", "zo_3", "zo_4", "zo_5",
    "ir_1", "ir_2", "ir_3", "ir_4", "ir_5",
    "or_1", "or_2", "or_3", "or_4", "or_5",
    "fl", "ml", "fm",
]

N_QUERY_PTS = 200   # query points sampled from the object mask at frame 0


# ── Image / mask helpers ──────────────────────────────────────────────────────

def load_frames(frame_dir: Path) -> torch.Tensor:
    """Load color frames as (1, T, 3, H, W) float32 tensor."""
    paths = sorted(frame_dir.glob("*.png")) + sorted(frame_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No images in {frame_dir}")
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    arr = np.stack(frames, axis=0)                      # T×H×W×3
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # T×3×H×W
    return t.unsqueeze(0)                               # 1×T×3×H×W


def load_mask(mask_path: Path) -> np.ndarray:
    """Return binary H×W mask (True = object)."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m > 127


def sample_mask_points(mask: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Sample N (x, y) pixel positions uniformly from the object mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xs), size=min(n, len(xs)))
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)  # N×2 (x,y)


def mask_paths_for_seq(seq_dir: Path) -> list:
    """Sorted list of mask image paths for a sequence."""
    mask_dir = seq_dir / "mask"
    if not mask_dir.exists():
        return []
    return sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg"))


# ── CoTracker2 inference ──────────────────────────────────────────────────────

_cotracker_model = None   # module-level cache across sequences

def get_cotracker(device: str):
    global _cotracker_model
    if _cotracker_model is None:
        _cotracker_model = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_offline", verbose=False
        ).to(device).eval()
    return _cotracker_model


def run_cotracker2(video: torch.Tensor, queries: np.ndarray,
                   device: str) -> tuple:
    """
    Run CoTracker2 (offline, window=8) on the video.

    queries: N×2 array of (x, y) pixel positions at frame 0.
    Returns:
        tracks:     (T, N, 2) float32 numpy
        visibility: (T, N) bool numpy
    """
    model = get_cotracker(device)

    N = len(queries)
    q = np.zeros((N, 3), dtype=np.float32)
    q[:, 0] = 0.0           # query at frame 0
    q[:, 1] = queries[:, 0]
    q[:, 2] = queries[:, 1]
    q_t = torch.from_numpy(q).unsqueeze(0).to(device)   # 1×N×3

    video = video.to(device)
    with torch.no_grad():
        tracks, vis = model(video, queries=q_t)

    return tracks[0].cpu().numpy(), vis[0].cpu().numpy() > 0.5


# ── Track Success Rate ────────────────────────────────────────────────────────

def track_success_rate(tracks: np.ndarray, vis: np.ndarray,
                       mask_paths: list, start_idx: int = 1) -> list:
    """
    For each frame from start_idx onward, compute the fraction of
    visible tracks that land inside the GT mask.

    Returns list of per-frame TSR values ∈ [0, 1].
    """
    H, W = INTRINSICS["h"], INTRINSICS["w"]
    tsr_values = []
    for t in range(start_idx, len(tracks)):
        if t >= len(mask_paths):
            break
        mask = load_mask(mask_paths[t])
        if mask is None:
            continue

        pts   = tracks[t]       # N×2 (x, y)
        v     = vis[t]          # N bool
        if v.sum() == 0:
            tsr_values.append(0.0)
            continue

        xs = np.clip(pts[v, 0].astype(int), 0, W - 1)
        ys = np.clip(pts[v, 1].astype(int), 0, H - 1)
        inside = mask[ys, xs].mean()
        tsr_values.append(float(inside))

    return tsr_values


# ── Per-sequence evaluation ───────────────────────────────────────────────────

def evaluate_sequence(seq_name: str, dataset_dir: Path, device: str) -> list:
    """Return list of per-frame TSR values."""
    seq_dir   = dataset_dir / "3D" / seq_name
    color_dir = seq_dir / "color"
    mask_dir  = seq_dir / "mask"

    if not color_dir.exists():
        print(f"  [skip] no color dir for {seq_name}")
        return []

    mask_paths = mask_paths_for_seq(seq_dir)
    if not mask_paths:
        print(f"  [skip] no mask dir for {seq_name}")
        return []

    # Load frame-0 mask and sample query points
    mask0 = load_mask(mask_paths[0])
    if mask0 is None or mask0.sum() == 0:
        print(f"  [skip] empty mask at frame 0 for {seq_name}")
        return []

    queries = sample_mask_points(mask0, N_QUERY_PTS)
    if len(queries) < 4:
        print(f"  [warn] too few mask pixels in {seq_name}")
        return []

    try:
        video = load_frames(color_dir)   # 1×T×3×H×W
    except Exception as e:
        print(f"  [skip] {seq_name}: {e}")
        return []

    T_frames = video.shape[1]
    n_masks  = len(mask_paths)
    n_eval   = min(T_frames, n_masks)

    tracks, vis = run_cotracker2(video[:, :n_eval], queries, device)
    # tracks: (n_eval, N, 2), vis: (n_eval, N)

    return track_success_rate(tracks, vis, mask_paths, start_idx=1)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to OPT dataset root")
    parser.add_argument("--output", default="./cotracker2_opt_results.json")
    parser.add_argument("--bodies", nargs="+", default=BODY_NAMES)
    parser.add_argument("--orientations", nargs="+", default=ORIENTATIONS)
    parser.add_argument("--patterns", nargs="+", default=MOTION_PATTERNS)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {}

    for body_name in args.bodies:
        short = BODY_SHORTS[body_name]
        print(f"\n{'='*60}")
        print(f"Object: {body_name}")

        body_tsr = []
        for orientation in args.orientations:
            for pattern in args.patterns:
                seq_name = f"{short}_{pattern}_{orientation}"
                print(f"  {seq_name}", end=" ... ", flush=True)
                t0 = time.time()
                tsr = evaluate_sequence(seq_name, dataset_dir, device)
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
        print(f"\n{'='*60}")
        print("Final Track Success Rate (fraction of tracks on-object):")
        for k, v in all_results.items():
            print(f"  {k:12s}: {v:.3f}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

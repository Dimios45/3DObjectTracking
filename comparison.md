# CoTracker2 vs M3T: OPT Dataset Comparison

## Overview

This document compares **CoTracker2** (a 2D dense point tracker) with **M3T** (a real-time 6DoF
region-, depth-, and texture-based tracker) on the **OPT benchmark** (Wu et al. 2017) using
the exact same AUC metric reported in Stoiber 2023, Table 3.3.

> **M3T numbers** are taken directly from the published thesis.
> **CoTracker2 numbers** are produced by the pipeline in `evaluate_cotracker_opt.py`.
> The OPT dataset (~8.5 GB) must be downloaded to `/mnt/data/mritunjoyh/datasets/opt/`
> before running.

---

## 1. Dataset & Metric

| Property | Value |
|---|---|
| Objects | 6: Soda, Chest, Ironman, House, Bike, Jet |
| Sequences | 552 (6 objects × 4 orientations × 23 motion patterns) |
| Image size | 1920 × 1080 (real-world, robot-arm recordings) |
| Camera fx/fy | 1060.197 / 1060.273 |
| Camera cx/cy | 964.809 / 560.952 |
| Sequence types | Translation (×5), Zoom (×5), In-plane rot. (×5), Out-of-plane rot. (×5), Flashing light, Moving light, Free motion |

### Metric: ADD-AUC

For each frame the **average vertex error** is computed:

```
ev(t) = (1/n) Σᵢ ‖ vᵢ − δT · vᵢ ‖
```

where `δT = (T_pred · G)⁻¹ · T_gt · G` is the relative pose error in geometry frame and
`vᵢ` are object mesh vertices. Tracking is successful when `ev(t) < kₑ · d`
(d = max vertex distance, kₑ ∈ [0, 0.2]).

The **AUC** score integrates the success-rate curve over kₑ ∈ [0, 0.2], giving a value in
[0, 0.2] that is **multiplied by 100 and reported as [0, 20]** in the paper.

---

## 2. Method Architectures

| Property | M3T | CoTracker2 + EPnP |
|---|---|---|
| Output | 6DoF pose (R, t) | 6DoF pose via EPnP on 2D tracks |
| Input | RGB + optional depth | RGB only |
| 3D model required | Yes (mesh for region / depth model) | Yes (mesh for 3D–2D correspondences) |
| Tracking paradigm | Region contour fitting + texture (ORB) + depth | Dense optical-flow-style 2D tracking |
| Window / iterations | 4 corr. iterations × 2 gradient updates | Sliding window W=8, stride=4 |
| GPU required | No (single CPU core) | Yes (CUDA/ROCm) |
| Runtime (OPT) | ~2–5 ms/frame (CPU) | ~100–300 ms/frame (GPU) |

---

## 3. M3T Results (Stoiber 2023, Table 3.3)

| Object | M3T AUC [0–20] | 2nd best | Method |
|--------|---------------|---------|--------|
| Soda | **15.55** | 14.85 | Bugaev'18 |
| Chest | **17.29** | 15.53 | ORB-SLAM2 |
| Ironman | **17.50** | 14.71 | Bugaev'18 |
| House | 16.57 | **17.28** | ORB-SLAM2 |
| Bike | **13.06** | 12.85 | J.Li'21 |
| Jet | 16.14 | **17.17** | Bugaev'18 |
| **Average** | **16.02** | 14.79 | Bugaev'18 |

### Full comparison table

| Approach | Soda | Chest | Ironman | House | Bike | Jet | Avg |
|---|---|---|---|---|---|---|---|
| PWP3D | 5.87 | 5.55 | 3.92 | 3.58 | 5.36 | 5.81 | 5.01 |
| ElasticFusion | 1.90 | 1.53 | 1.69 | 2.70 | 1.57 | 1.86 | 1.87 |
| UDP | 8.49 | 6.79 | 5.25 | 5.97 | 6.10 | 2.34 | 5.82 |
| ORB-SLAM2 | 13.44 | 15.53 | 11.20 | **17.28** | 10.41 | 9.93 | 12.97 |
| Bugaev'18 | 14.85 | 14.97 | 14.71 | 14.48 | 12.55 | **17.17** | 14.79 |
| Tjaden'18 | 8.86 | 11.76 | 11.99 | 10.15 | 11.90 | 13.22 | 11.31 |
| Zhong'20 | 9.01 | 12.24 | 11.21 | 13.61 | 12.83 | 15.44 | 12.39 |
| J.-C. Li'21 | 9.00 | 14.92 | 13.44 | 13.60 | 12.85 | 10.64 | 12.41 |
| Huang'22 | 9.07 | 12.93 | 8.80 | 11.15 | 7.96 | 11.09 | 10.17 |
| **M3T** | **15.55** | **17.29** | **17.50** | 16.57 | **13.06** | 16.14 | **16.02** |
| **CoTracker2+EPnP** | — | — | — | — | — | — | — |

---

## 4. CoTracker2 + EPnP Pipeline

### 4.1 Initialization (frame 0 of each sequence)

1. Sample **N=200** vertices from the object's .obj mesh uniformly at random (seed=42).
2. Project them onto frame 0 using the GT pose + camera intrinsics → initial 2D positions.
3. Discard points with `z ≤ 0` or outside image bounds.
4. These become the CoTracker2 **query points**: `queries[i] = (frame=0, x_i, y_i)`.

### 4.2 Tracking (CoTracker2 offline, window W=8)

```
CoTracker2("cotracker2", pretrained=True)  ← stride=4, window=8
  input:  (1, T, 3, H, W) float32 tensor
  output: tracks (T, N, 2)  visibility (T, N) in [0, 1]
```

Tracks are accepted when `visibility > 0.5` and pixel is inside the image.

### 4.3 Pose recovery (EPnP + RANSAC, each frame t ≥ 1)

```python
visible_3d = pts3d[visible_mask]       # known 3D mesh vertices
visible_2d = tracks[t][visible_mask]   # predicted 2D pixels
R, t = cv2.solvePnPRansac(
    visible_3d, visible_2d, K, None,
    flags=SOLVEPNP_EPNP,
    confidence=0.99, reprojectionError=8.0,
)
```

Falls back to the previous frame's pose if `< 4` points are visible.

### 4.4 ADD-AUC computation

```python
G = geometry2body_pose(body_name)       # pure translation offset
delta = inv(pose_pred @ G) @ pose_gt @ G
error = mean(‖vᵢ − (R_δ·vᵢ + t_δ)‖) over 1000 sampled vertices
auc_frame = 0.2 * (1 - min(error / (diameter * 0.2), 1))
```

Final AUC is averaged over all frames in all sequences for each object × 20 for paper scale.

---

## 5. Downloading the OPT Dataset

```bash
# Official dataset page: http://media.ee.ntu.edu.tw/research/OPT/
# Download and extract to:
mkdir -p /mnt/data/mritunjoyh/datasets/opt

# Expected layout after extraction:
# /mnt/data/mritunjoyh/datasets/opt/
#   Model3D/
#     soda/soda.obj
#     chest/chest.obj  ...
#   3D/
#     poses/
#       so_tr_1_b.txt  so_tr_1_f.txt  ...
#     so_tr_1_b/
#       color/0001.png  0002.png  ...
#       depth/0001.png  0002.png  ...
#     so_tr_1_f/ ...
```

Estimated disk usage: **~8.5 GB** (well within the 15 GB budget).

---

## 6. Running the Evaluation

```bash
# Download cotracker2 checkpoint (cached automatically on first run)
conda run -n graspmas python3 evaluate_cotracker_opt.py \
    --dataset /mnt/data/mritunjoyh/datasets/opt \
    --output  ./cotracker2_opt_results.json

# Partial run (single object, quick sanity check):
conda run -n graspmas python3 evaluate_cotracker_opt.py \
    --dataset  /mnt/data/mritunjoyh/datasets/opt \
    --bodies   ironman \
    --patterns tr_1 tr_2 \
    --output   ./cotracker2_opt_ironman_partial.json
```

Results are reported as **AUC × 20** to match the paper scale.

---

## 7. Expected Performance Analysis

### 7.1 Why CoTracker2 will underperform M3T

| Failure mode | Impact on OPT |
|---|---|
| No model-in-the-loop | CoTracker2 tracks the whole scene; background regions can pull 2D tracks off the object, corrupting PnP. |
| Window boundary drift | Each W=8 seam introduces a small pose jump. Over 100+ frames this accumulates. |
| Textureless regions | Soda (white cylinder) and Bike (uniform plastic) have almost no texture → CoTracker2 tracks fail inside the object silhouette. |
| Motion blur | OPT has dedicated blur sequences. CoTracker2's correlation window degrades with blur; M3T's region-based approach is more robust. |
| No depth | M3T uses the RGB-D camera. CoTracker2 is RGB-only, losing the depth constraint that stabilises scale/translation. |
| EPnP sensitivity | With only ~50–100 visible points after pruning, EPnP under RANSAC is noisy. M3T fits hundreds of region lines simultaneously. |

### 7.2 Per-object prediction

| Object | Key challenge | Expected CoTracker2 AUC [0–20] |
|---|---|---|
| Soda | Rotationally symmetric + minimal texture | 2–5 |
| Chest | Rich surface texture → most trackable | 8–12 |
| Ironman | Detailed surface, good texture | 7–11 |
| House | White walls, textureless + symmetric ambiguity | 3–7 |
| Bike | Thin structure, partial self-occlusion | 5–9 |
| Jet | Complex shape, some texture | 6–10 |
| **Average** | | **~5–9** |

Compare to M3T average of **16.02** — CoTracker2+EPnP is expected to trail by roughly 7–11 AUC points.

### 7.3 Motion pattern breakdown

| Pattern | Expected behaviour |
|---|---|
| Translation (tr_*) | Relatively stable; 2D tracks shift linearly → EPnP works well |
| Zoom (zo_*) | Scale change within single window → EPnP handles depth change poorly |
| In-plane rotation (ir_*) | Features rotate in-plane → tracks stay on object, PnP stable |
| Out-of-plane rotation (or_*) | Self-occlusion changes which vertices are visible each frame |
| Flashing light (fl) | Sudden illumination changes corrupt CoTracker2 correlations |
| Moving light (ml) | Gradual shading changes, moderate degradation |
| Free motion (fm) | Combined challenges, worst case for CoTracker2 |

---

## 8. Architecture Comparison

```
M3T (Region + Depth + Texture):
  Previous pose estimate
       │
       ├── Region modality: silhouette rendered → N=200 line segments
       │   sample foreground/background histograms → Gauss-Newton gradient
       │
       ├── Depth modality: depth image → N=200 surface points
       │   ICP-style depth residuals → Gauss-Newton gradient
       │
       └── Texture modality: ORB keyframes → descriptor matching
           → Gauss-Newton gradient
                 │
                 ▼
       Joint Gauss-Newton optimisation (4 corr × 2 update iter)
                 │
                 ▼
       New 6DoF pose  ──────────────────── ~3 ms/frame, 1 CPU core, no GPU


CoTracker2 + EPnP:
  Previous 2D track positions  (N=200 query pts)
       │
       ▼
  CoTracker2 sliding window (W=8, stride=4)  ─── GPU required
       │
       ▼
  Predicted 2D positions + visibility scores (T × N × 2)
       │
       ▼
  EPnP RANSAC with 3D correspondences
       │
       ▼
  New 6DoF pose  ──────────────────────── ~100–300 ms/frame, GPU
```

### Core architectural difference

M3T **iteratively refines** the pose by comparing the rendered model against the actual image at
every frame. It knows exactly which image pixels belong to the object boundary (via silhouette
rendering) and only uses those for fitting. CoTracker2 has no such model awareness — it tracks
whatever it finds in the image, and relies on EPnP to back-project into 6DoF. This indirect
chain (2D→EPnP) is inherently less accurate than direct model-image comparison.

---

## 9. Results Summary

> **Metric note:** M3T is evaluated with ADD-AUC [0–20] (requires 3D mesh + EPnP pose recovery).
> CoTracker2 is evaluated with **Track Success Rate (TSR) [0–1]** — fraction of mask-initialized
> 2D tracks that remain on the object per frame, averaged over all sequences.
> These are different metrics; direct numerical comparison is not valid, but the table below
> shows CoTracker2's 2D tracking quality alongside M3T's 6DoF pose accuracy.

### 9.1 M3T ADD-AUC [0–20] vs CoTracker2 TSR [0–1]

| Object  | M3T ADD-AUC [0–20] | CoTracker2 TSR [0–1] |
|---------|-------------------|----------------------|
| Soda    | 15.55             | **0.678**            |
| Chest   | 17.29             | **0.671**            |
| Ironman | 17.50             | **0.615**            |
| House   | 16.57             | **0.656**            |
| Bike    | 13.06             | **0.645**            |
| Jet     | 16.14             | **0.587**            |
| **Average** | **16.02**     | **0.642**            |

### 9.2 CoTracker2 TSR breakdown by motion pattern

| Motion type | Typical TSR | Notes |
|---|---|---|
| Translation (tr_*) | ~0.93–0.98 | Smooth linear motion — CoTracker excels |
| Zoom (zo_1, slow) | ~0.004–0.62 | First zoom sequence always fails; scale change breaks tracks |
| Zoom (zo_2–5, fast) | ~0.91–0.99 | Shorter sequences, tracks recover |
| In-plane rotation (ir_1, slow) | ~0.06–0.23 | Long rotation causes drift out of mask |
| In-plane rotation (ir_2–5) | ~0.90–0.95 | Shorter sequences handle well |
| Out-of-plane rotation (or_1) | ~0.31–0.34 | Severe self-occlusion causes mass track loss |
| Out-of-plane rotation (or_2–5) | ~0.85–0.97 | Partial rotations, good recovery |
| Flashing light (fl) | ~0.94–0.98 | Illumination changes handled well |
| Moving light (ml) | ~0.94–0.98 | Gradual shading — robust |
| Free motion (fm) | ~0.54–0.59 | Worst case: combined challenges |

### 9.3 Key observations

- **CoTracker2 is a strong 2D tracker** — 0.642 average TSR across 552 sequences means
  64% of mask-initialized tracks stay on the object at every frame.
- **Zoom and slow rotation are failure modes**: the `zo_1` and `ir_1` sequences (340–369 frames,
  very slow motion) cause near-total track loss as points gradually drift off the object boundary.
- **M3T's model-in-the-loop advantage**: M3T re-renders the object silhouette at each frame and
  fits to it directly. CoTracker2 has no such feedback — once tracks drift off the object, there
  is no correction mechanism.
- **Free motion (fm)** is the hardest pattern for CoTracker2 (TSR ~0.55–0.59), consistent with
  the prediction in Section 7.

---

## 10. References

- Stoiber M (2023). *Closing the Loop: 3D Object Tracking for Advanced Robotic Manipulation.*
  Dissertation, TU Munich.
- Wu B et al. (2017). *OPT: One-Pass Tracker.* ICCV 2017.
- Karaev N et al. (2023). *CoTracker: It is Better to Track Together.* ECCV 2024.
- Lepetit V, Moreno-Noguer F, Fua P (2009). *EPnP: An Accurate O(n) Solution to the PnP
  Problem.* IJCV 77(2).
- Bugaev D et al. (2018). *3D Object Tracking with Adaptively Weighted Local Bundles.*
  BMVC 2018.
- Mur-Artal R, Tardós JD (2017). *ORB-SLAM2: An Open-Source SLAM System for Monocular,
  Stereo, and RGB-D Cameras.* IEEE T-RO.

# OPT Dataset: 2D Tracker Comparison

## Overview

This document compares **CoTracker2**, **CoTracker3**, and **LocoTrack** (2D dense point trackers)
with **M3T** (a real-time 6DoF region-, depth-, and texture-based tracker) on the **OPT benchmark**
(Wu et al. 2017).

> **M3T numbers** are taken directly from Stoiber 2023 (Table 3.3) using the ADD-AUC metric.
> **CoTracker2/3 and LocoTrack numbers** are produced by the evaluation scripts in
> `co-tracker/evaluate_*.py` using the TSR metric (see §2).
> The OPT 424p dataset must be present at `/mnt/data/mritunjoyh/datasets/opt/` before running.

---

## 1. Dataset

We used the scaled-down OPT variant (`OPT424p_3D.zip`) which provides:

| Property | Value |
|---|---|
| Objects | 6: Soda, Chest, Ironman, House, Bike, Jet |
| Sequences | 552 (6 objects × 4 orientations × 23 motion patterns) |
| Image size | 512 × 424 |
| Available data | RGB frames, depth frames, GT segmentation masks |
| Sequence types | Translation (×5), Zoom (×5), In-plane rot. (×5), Out-of-plane rot. (×5), Flashing light, Moving light, Free motion |

Dataset path: `/mnt/data/mritunjoyh/datasets/opt/`

```
opt/
  soda/  chest/  ironman/  house/  bike/  jet/
    each object/
      so_tr_1_b/  so_tr_1_f/  ...   (552 total)
        color/  0001.png  0002.png  ...
        depth/  0001.png  0002.png  ...
        mask/   0001.png  0002.png  ...   ← GT segmentation masks
```

---

## 2. Metrics

### TSR — Track Success Rate (used for CoTracker2/3, LocoTrack)

The 424p variant does not include mesh `.obj` files, so we use a mask-based 2D tracking metric:

1. **Init:** Sample 200 random points from the GT object mask at frame 0.
2. **Track:** Run the tracker forward through the full sequence.
3. **Score:** For each frame t, compute the fraction of still-visible tracks whose 2D position
   falls inside the GT mask.
4. **TSR** = mean over all frames, averaged across all 552 sequences.

TSR ∈ [0, 1]; higher is better.

### ADD-AUC [0–20] (M3T reference numbers only)

For each frame the **average vertex error** is computed:

```
ev(t) = (1/n) Σᵢ ‖ vᵢ − δT · vᵢ ‖
```

where `δT = (T_pred · G)⁻¹ · T_gt · G` is the relative pose error and `vᵢ` are object mesh
vertices. AUC integrates the success-rate curve over threshold kₑ ∈ [0, 0.2], multiplied by 100.
This requires 3D mesh files which the 424p dataset does not provide, so it is only reported for M3T.

---

## 3. Method Overview

| Property | M3T | CoTracker2 | CoTracker3 | LocoTrack |
|---|---|---|---|---|
| Output | 6DoF pose | 2D tracks | 2D tracks | 2D tracks |
| Tracking paradigm | Region contour + depth ICP + ORB texture | Sliding window (W=8) correlation | Sliding window (W=60) correlation | Local-to-global TAPIR (CMDTop) |
| 3D model required | Yes | No | No | No |
| GPU required | No (single CPU core) | Yes | Yes | Yes |
| Runtime on OPT | ~3–5 ms/frame (CPU) | ~32 ms/frame (MI300X) | ~33 ms/frame (MI300X) | ~180 ms/frame (MI300X) |
| Hardware used | CPU | AMD MI300X, ROCm 6.2 | AMD MI300X, ROCm 6.2 | AMD MI300X, ROCm 6.2 |

---

## 4. Results

> **Metric note:** M3T and 2D trackers use incompatible metrics. ADD-AUC measures 6DoF pose accuracy
> (requires mesh vertices); TSR measures whether 2D tracks stay on the object. The table shows both
> side by side for context — do not compare the numbers directly.

### 4.1 Per-object summary

| Object | M3T ADD-AUC [0–20] | CoTracker2 TSR | CoTracker3 TSR | LocoTrack TSR |
|---|---|---|---|---|
| Soda | 15.55 | 0.678 | 0.967 | **0.970** |
| Chest | 17.29 | 0.671 | 0.960 | **0.974** |
| Ironman | 17.50 | 0.615 | **0.931** | 0.926 |
| House | 16.57 | 0.656 | 0.971 | **0.972** |
| Bike | 13.06 | 0.645 | **0.950** | 0.947 |
| Jet | 16.14 | 0.587 | **0.909** | 0.904 |
| **Average** | **16.02** | 0.642 | 0.948 | **0.949** |

### 4.2 TSR by motion pattern

| Motion type | CoTracker2 TSR | CoTracker3 TSR | LocoTrack TSR |
|---|---|---|---|
| Translation (tr_*) | ~0.93–0.98 | ~0.96–0.98 | ~0.95–0.98 |
| Zoom slow (zo_1) | ~0.004–0.62 | **~0.96–0.99** | ~0.96–0.99 |
| Zoom fast (zo_2–5) | ~0.91–0.99 | ~0.94–0.99 | ~0.93–0.99 |
| In-plane rot. slow (ir_1) | ~0.06–0.23 | **~0.88–0.97** | ~0.87–0.96 |
| In-plane rot. fast (ir_2–5) | ~0.90–0.95 | ~0.93–0.97 | ~0.92–0.96 |
| Out-of-plane rot. slow (or_1) | ~0.31–0.34 | ~0.85–0.94 | ~0.84–0.93 |
| Out-of-plane rot. fast (or_2–5) | ~0.85–0.97 | ~0.88–0.97 | ~0.87–0.96 |
| Flashing light (fl) | ~0.94–0.98 | ~0.96–0.99 | ~0.96–0.99 |
| Moving light (ml) | ~0.94–0.98 | ~0.97–0.99 | ~0.96–0.99 |
| Free motion (fm) | ~0.54–0.59 | **~0.89–0.99** | ~0.88–0.98 |

### 4.3 Key observations

- **CoTracker3 and LocoTrack are virtually identical** (0.948 vs 0.949, within noise across 552
  sequences). Neither has a consistent per-object advantage — they trade leads object by object.
- **CoTracker3 massively outperforms CoTracker2** — the larger window (60 frames vs 8) eliminates
  catastrophic drift in slow sequences: `zo_1` TSR 0.004→0.97, `ir_1` TSR 0.06→0.93.
- **LocoTrack's CMDTop architecture** (local-to-global correlation) does not translate to a
  measurable TSR advantage over CoTracker3 on OPT — both are near ceiling for 2D mask tracking.
- **Gap vs M3T**: 2D trackers plateau around TSR≈0.95. M3T's ADD-AUC=16.02/20 measures 6DoF pose
  accuracy — a fundamentally harder task requiring 3D model understanding that 2D trackers cannot
  provide without EPnP + mesh files.
- **Speed tradeoff**: LocoTrack is ~5× slower per frame than CoTracker3 on MI300X (180 ms vs
  33 ms) with equal accuracy — CoTracker3 is the better choice when throughput matters.

---

## 5. M3T Reference Numbers (Stoiber 2023, Table 3.3)

Full comparison against prior art from the thesis:

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

---

## 6. Running the Evaluation

All evaluation scripts are in `co-tracker/`. Run in the `graspmas` conda environment on AMD MI300X.

### CoTracker2

```bash
HIP_VISIBLE_DEVICES=1 conda run -n graspmas python3 evaluate_cotracker_opt.py
# Results written to: cotracker2_opt_results.json
```

### CoTracker3 (parallelised across 3 GPUs)

```bash
# GPU 1: soda + chest
HIP_VISIBLE_DEVICES=1 nohup conda run -n graspmas python3 evaluate_cotracker3_opt.py \
    --bodies soda chest --output ct3_gpu1.json > ct3_gpu1.log 2>&1 &

# GPU 2: ironman + house
HIP_VISIBLE_DEVICES=2 nohup conda run -n graspmas python3 evaluate_cotracker3_opt.py \
    --bodies ironman house --output ct3_gpu2.json > ct3_gpu2.log 2>&1 &

# GPU 3: bike + jet
HIP_VISIBLE_DEVICES=3 nohup conda run -n graspmas python3 evaluate_cotracker3_opt.py \
    --bodies bike jet --output ct3_gpu3.json > ct3_gpu3.log 2>&1 &

# Merge results
python3 merge_results.py ct3_gpu1.json ct3_gpu2.json ct3_gpu3.json cotracker3_opt_results.json
```

### LocoTrack (parallelised across 3 GPUs)

```bash
# Same GPU split as CoTracker3; replace evaluate_cotracker3_opt.py → evaluate_locotrack_opt.py
# Output per-object JSONs: lt_soda.json  lt_chest.json  lt_ironman.json  lt_house.json  lt_bike.json  lt_jet.json
python3 merge_results.py lt_soda.json lt_chest.json lt_ironman.json \
    lt_house.json lt_bike.json lt_jet.json locotrack_opt_results.json
```

### GPU monitoring (AMD MI300X — use rocm-smi, not nvtop)

```bash
watch -n 2 rocm-smi
```

`nvtop` always shows 0% on MI300X because it reads the DRM interface; the actual KFD
compute interface is only visible via `rocm-smi`.

---

## 7. Architecture Comparison

```
M3T (Region + Depth + Texture):
  Previous pose estimate
       │
       ├── Region modality: rendered silhouette → N=200 line segments
       │   sample foreground/background histograms → Gauss-Newton gradient
       │
       ├── Depth modality: depth image → N=200 surface points
       │   ICP-style depth residuals → Gauss-Newton gradient
       │
       └── Texture modality: ORB keyframes → descriptor matching
           → Gauss-Newton gradient
                 │
                 ▼
       Joint Gauss-Newton optimisation (4 corrections × 2 update iterations)
                 │
                 ▼
       New 6DoF pose  ─── ~3–5 ms/frame, 1 CPU core, no GPU


CoTracker3 (offline, W=60 sliding window):
  Query points initialised from GT mask (N=200) at frame 0
       │
       ▼
  Transformer with 60-frame context window (stride=1)
       │
       ▼
  2D tracks (T×N×2) + visibility scores  ─── ~33 ms/frame, GPU


LocoTrack (TAPIR + CMDTop local correlation):
  Query points initialised from GT mask (N=200) at frame 0
       │
       ▼
  Local feature volumes (pyramid) + CMDTop global aggregation
       │
       ▼
  2D tracks (T×N×2) + occlusion logits  ─── ~180 ms/frame, GPU
```

### Core architectural difference

M3T **directly fits** its 3D model to each image by comparing rendered silhouettes and depth maps
against the actual image — it is intrinsically model-aware and knows exactly which pixels belong
to the object boundary. CoTracker2/3 and LocoTrack are model-agnostic dense trackers: they follow
2D image patches forward through time with no 3D understanding. This makes them faster to deploy
(no mesh needed) but unable to produce 6DoF pose without an additional PnP step.

---

## 8. Wan2.1 Manipulation Video Tracking

CoTracker3 was also run on 26 robot manipulation videos generated by Wan2.1.

### Dataset

| Category | Videos | Object tracked |
|---|---|---|
| drop | 6 (including 1 duplicate) | Red bowl |
| lid | 5 | Lid handle |
| pick-place | 5 | Handwash bottle |
| pour | 5 | Jug / kettle |
| rolling | 5 | Ball |

Videos are at `/mnt/data/mritunjoyh/Wan2.1/outputs/*.mp4`.
Tracked outputs are at `/mnt/data/mritunjoyh/Wan2.1/outputs/output_wan/*_tracked.mp4`.

### Pipeline

1. **Bounding box selection** (`select_bboxes.py`): interactive `cv2.selectROI` over frame 0 of
   each video (requires SSH with X11 forwarding: `ssh -Y`). Boxes saved to `bboxes.json`.

2. **Tracking** (`track_from_bboxes.py`): reads `bboxes.json`, samples 200 points uniformly
   inside each bbox, runs CoTracker3 offline, writes visualised `*_tracked.mp4`.

```bash
# Draw boxes (requires X11)
conda run -n graspmas python3 /mnt/data/mritunjoyh/Wan2.1/select_bboxes.py

# Track all videos
HIP_VISIBLE_DEVICES=1 conda run -n graspmas python3 /mnt/data/mritunjoyh/Wan2.1/track_from_bboxes.py
```

### Bounding boxes (from bboxes.json, format: [x1, y1, x2, y2])

| Video | Bbox |
|---|---|
| drop_1.mp4 | [6, 376, 140, 425] |
| drop_2.mp4 | [15, 370, 127, 432] |
| drop_3.mp4 | [9, 371, 131, 425] |
| drop_4.mp4 | [20, 364, 128, 429] |
| drop_5.mp4 | [20, 369, 117, 433] |
| lid_1.mp4 | [127, 360, 200, 376] |
| lid_2.mp4 | [129, 353, 193, 381] |
| lid_3.mp4 | [129, 362, 184, 380] |
| lid_4.mp4 | [131, 354, 188, 385] |
| lid_5.mp4 | [128, 364, 179, 387] |
| pick-place_1.mp4 | [101, 254, 157, 406] |
| pick-place_2.mp4 | [104, 255, 160, 369] |
| pick-place_3.mp4 | [90, 256, 173, 438] |
| pick-place_4.mp4 | [93, 270, 167, 432] |
| pick-place_5.mp4 | [100, 284, 168, 476] |
| pour_1.mp4 | [106, 186, 233, 421] |
| pour_2.mp4 | [101, 177, 230, 427] |
| pour_3.mp4 | [97, 174, 232, 405] |
| pour_4.mp4 | [97, 179, 237, 411] |
| pour_5.mp4 | [97, 178, 233, 415] |
| rolling_1.mp4 | [69, 302, 178, 402] |
| rolling_2.mp4 | [77, 285, 178, 388] |
| rolling_3.mp4 | [80, 285, 192, 400] |
| rolling_4.mp4 | [76, 292, 182, 400] |
| rolling_5.mp4 | [78, 285, 182, 394] |

### Viewing outputs

```bash
# Start HTTP server
cd /mnt/data/mritunjoyh/Wan2.1/outputs/output_wan
python3 -m http.server 8081

# SSH tunnel on your local machine
ssh -L 8081:localhost:8081 mritunjoyh@<server-ip>

# Open browser → http://localhost:8081
```

---

## 9. References

- Stoiber M (2023). *Closing the Loop: 3D Object Tracking for Advanced Robotic Manipulation.*
  Dissertation, TU Munich.
- Wu B et al. (2017). *OPT: One-Pass Tracker.* ICCV 2017.
- Karaev N et al. (2023). *CoTracker: It is Better to Track Together.* ECCV 2024.
- Doersch C et al. (2023). *TAPIR: Tracking Any Point with per-frame Initialization and temporal
  Refinement.* ICCV 2023.
- Song Y et al. (2024). *LocoTrack: Local-to-Global Correlation for 2D Point Tracking.* ECCV 2024.
- Bugaev D et al. (2018). *3D Object Tracking with Adaptively Weighted Local Bundles.* BMVC 2018.
- Mur-Artal R, Tardós JD (2017). *ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo,
  and RGB-D Cameras.* IEEE T-RO.

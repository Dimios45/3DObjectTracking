# Comparative Study: 3D Object Tracking Algorithms on Rolling Ball

All four algorithms track the same rolling ball video (`rolling_1.mp4`, 720×544, 16 fps, 81 frames)
using a sphere mesh (radius=0.035 m) and approximate camera intrinsics (fu=fv=600, ppu=360, ppv=272).

---

## Algorithm Overview

| Property | RBGT (2020) | SRT3D (2022) | ICG (2022) | M3T (2023) |
|---|---|---|---|---|
| Paper | Stoiber et al., ECCV 2020 | Stoiber et al., IROS 2022 | Stoiber et al., T-RO 2022 | Stoiber et al., T-RO 2023 |
| Namespace | `rbgt::` | `srt3d::` | `icg::` | `m3t::` |
| Modalities | Region only | Region only | Region + Depth | Region + Depth + Texture |
| Kinematic chains | No | No | No | Yes (Link / Optimizer) |
| GPU acceleration | No | No | No | Texture modality (CUDA) |
| Config style | Hardcoded C++ | Hardcoded C++ | YAML files | YAML files |
| Body pose convention | `world2body` | `world2body` | `body2world` (YAML) | `link2world` (YAML) |
| Tracker setup call | Manual `Init()` per object | `SetUpTracker()` | `SetUp()` per object | `SetUp()` per object |
| Tracker run call | `StartTracker(bool)` | `StartTracker(bool)` | `RunTrackerProcess(bool,bool)` | `RunTrackerProcess(bool,bool)` |
| Viewer type | `NormalImageViewer` | `NormalViewer` | `NormalColorViewer` | `NormalColorViewer` |
| Camera class | `ImageLoaderCamera` (Init-based) | `LoaderCamera` (constructor) | `LoaderColorCamera` (YAML) | `LoaderColorCamera` (YAML) |
| Model generation | Explicit `GenerateModel()` + `SaveModel()` | Auto in `SetUp()` | Auto in `SetUp()` | Auto in `SetUp()` |
| Detector | None (pose set directly) | None (pose set directly) | `StaticDetector` with body_ptr | `StaticDetector` with optimizer_ptr |

---

## Output Videos

| Algorithm | Output Video |
|---|---|
| M3T | `Wan2.1/outputs/rolling_1_tracked.mp4` |
| ICG | `Wan2.1/outputs/rolling_1_tracked_icg.mp4` |
| SRT3D | `Wan2.1/outputs/rolling_1_tracked_srt3d.mp4` |
| RBGT | `Wan2.1/outputs/rolling_1_tracked_rbgt.mp4` |

---

## Changes Made to Get Each Algorithm Working

### 1. M3T

**New file:** `M3T/examples/run_ball_tracking.cpp`

Wrote a custom driver that wires up the full M3T object graph:
`LoaderColorCamera` (from YAML) → `RendererGeometry` → `NormalColorViewer` →
`Body` → `RegionModel` → `RegionModality` → `Link` → `Optimizer` → `StaticDetector` → `Tracker`.

Key points:
- Camera and body loaded from YAML metafiles (`camera.yaml`, `ball.yaml`)
- Initial pose provided via `static_detector.yaml` using the `link2world_pose` key
- Viewer set to headless (`display_images` off by default in M3T) with `StartSavingImages`
- `RunTrackerProcess(true, true)` — auto-detects initial pose and starts tracking immediately

**Supporting config files written** (in `Wan2.1/tracking/ball/`):
- `camera.yaml` — LoaderColorCamera metafile (intrinsics + frame path template)
- `ball.yaml` — Body metafile (sphere.obj path, scale, geometry2body pose)
- `static_detector.yaml` — initial pose as `link2world_pose` (from `mark_ball.py` output)

```cmake
# CMakeLists.txt addition:
add_executable(run_ball_tracking run_ball_tracking.cpp)
target_link_libraries(run_ball_tracking PUBLIC m3t)
```

---

### 2. ICG

**New file:** `ICG/examples/run_ball_tracking.cpp`

Structure mirrors M3T but uses `icg::` namespace and the ICG-era API (no Link layer):
`LoaderColorCamera` → `RendererGeometry` → `NormalColorViewer` →
`Body` → `RegionModel` → `RegionModality` → `Optimizer` → `StaticDetector` → `Tracker`.

Key differences from M3T:
- `Optimizer("name")` takes **no body argument** — body is inferred from modalities
- `StaticDetector` takes `body_ptr` directly, **not** `optimizer_ptr`
- Body uses `body2world_pose` convention (same as M3T), but YAML key is `body2world_pose` not `link2world_pose`

**New config file:** `static_detector_icg.yaml` — same initial pose but with `body2world_pose` key

```cmake
add_executable(run_ball_tracking run_ball_tracking.cpp)
target_link_libraries(run_ball_tracking PUBLIC icg)
```

---

### 3. SRT3D

**New file:** `SRT3D/examples/run_ball_tracking.cpp`

Fully hardcoded driver (no YAML at all):
`LoaderCamera` → `RendererGeometry` → `NormalViewer` →
`Body` → `Model` → `RegionModality` → `Tracker`.

Key differences:
- All parameters set inline in C++ (intrinsics, pose, paths)
- Uses `world2body` convention — initial pose is the **inverse** of the M3T pose:
  `world2body.translation() = {0.013202, -0.027180, -0.232974}`
- `NormalViewer` gets `set_display_images(false)` + `StartSavingImages(output_dir, "png")`
- `SetUpTracker()` then `StartTracker(true)` (true = start tracking immediately)

**Two library bugs fixed** (upstream code, not our driver):

**Bug 1 — `src/tracker.cpp::UpdateViewers()`:**
```cpp
// Before: cv::waitKey called unconditionally — crashes headless
char key = cv::waitKey(viewer_time_);

// After: only call when a viewer actually has display enabled
bool any_display = false;
for (auto &viewer_ptr : viewer_ptrs_) {
    viewer_ptr->UpdateViewer(iteration);
    if (viewer_ptr->display_images()) any_display = true;
}
if (any_display) {
    char key = cv::waitKey(viewer_time_);
    ...
}
```
`cv::waitKey` requires an active OpenCV window. With `display_images_=false` no `imshow` call
is made, so there are no windows — calling `waitKey` anyway crashes OpenCV's event loop in
headless/SSH sessions.

**Bug 2 — `src/normal_viewer.cpp::UpdateViewer()`:**
```cpp
// Before: bool function with no return statement — undefined behaviour
bool NormalViewer::UpdateViewer(int save_index) {
    ...
    DisplayAndSaveImage(save_index, viewer_image);
}   // ← missing return!

// After:
    DisplayAndSaveImage(save_index, viewer_image);
    return true;   // ← added
}
```
The missing `return` is undefined behaviour. The compiler generated a corrupt function epilogue
for the `bool` return path. When the function exited normally, control fell into garbage code
that invoked `_Unwind_Resume` (C++ stack unwinder) with a garbage exception pointer (`exc=0x2`),
causing a `SIGSEGV`. Diagnosed with GDB: crash at `NormalViewer::UpdateViewer [clone .cold]`.
The ASAN build also warned: *"control reaches end of non-void function"* at line 60.

```cmake
add_executable(run_ball_tracking run_ball_tracking.cpp)
target_link_libraries(run_ball_tracking PUBLIC srt3d)
```

---

### 4. RBGT

**New file:** `RBGT/examples/run_ball_tracking.cpp`

Oldest API — all objects use an `Init()` pattern instead of constructors:
`ImageLoaderCamera::Init()` → `NormalImageViewer::Init()` →
`Body` → `Model::LoadModel()`/`GenerateModel()`/`SaveModel()` → `RegionModality::Init()` → `Tracker`.

Key differences:
- `rbgt::Tracker()` constructor takes **no name argument**
- Camera uses `Init(name, dir, prefix, start_idx, end_idx)` — frame filenames have **no leading zeros**
  (`color_camera_1.png` not `color_camera_image_0001.png`)
- `NormalImageViewer::Init(name, renderer_geometry_ptr, camera_ptr)` — separate Init call
- Model: `GenerateModel(*body, sphere_radius, n_divides, n_points)` + `SaveModel()` separately;
  on subsequent runs `LoadModel()` is called first and generation is skipped
- `body_ptr->set_occlusion_mask_id(1)` required before adding to renderer geometry
- Same `world2body` convention as SRT3D

```cmake
add_executable(run_ball_tracking run_ball_tracking.cpp)
target_link_libraries(run_ball_tracking PUBLIC rbgt)
```

---

## Initial Pose Setup

All algorithms use the same physical ball position. The initial pose was obtained by running
`mark_ball.py` on the first frame of `rolling_1.mp4`, which outputs body2world translation:

```
tx = -0.013202 m,  ty = 0.027180 m,  tz = 0.232974 m   (no rotation)
```

| Algorithm | Pose key used | Value |
|---|---|---|
| M3T | `link2world_pose` in `static_detector.yaml` | `[-0.013202, 0.027180, 0.232974]` |
| ICG | `body2world_pose` in `static_detector_icg.yaml` | `[-0.013202, 0.027180, 0.232974]` |
| SRT3D | `body_ptr->set_world2body_pose(...)` in C++ | `[0.013202, -0.027180, -0.232974]` (inverse) |
| RBGT | `body_ptr->set_world2body_pose(...)` in C++ | `[0.013202, -0.027180, -0.232974]` (inverse) |

---

## Why Older Algorithms Can Look Better Than M3T on This Specific Video

M3T is the most powerful of the four, but for this setup (single body, region-only, no depth,
no texture), the older algorithms can appear to track more crisply for these reasons:

1. **Direct pose update vs Link/Optimizer indirection.** RBGT/SRT3D/ICG update `body2world_pose`
   directly each Newton step. M3T applies the update to a `Link`, which propagates to the body
   through an extra transformation, accumulating floating-point rounding.

2. **Parameters tuned for region-only.** All four use the same core algorithm (contour sampling →
   color histogram → Newton optimization). But RBGT/SRT3D/ICG were benchmarked and tuned
   exclusively for region-modality tracking. M3T's region modality defaults were designed to
   complement depth and texture modalities, not to carry the load alone.

3. **Viewer rendering style creates a perception difference.** RBGT/SRT3D use `NormalImageViewer`
   / `NormalViewer` which draws the rendered surface normals as solid colors over the image,
   making the sphere overlay look vivid and clearly positioned. M3T/ICG use `NormalColorViewer`
   which blends normals with the camera's actual pixel colors, which can look less distinct on a
   white/grey ball.

4. **M3T's strength is multi-modality.** With depth and texture active, M3T outperforms all
   predecessors significantly. With region-only it is essentially the same algorithm but with more
   overhead and slightly different parameter defaults.

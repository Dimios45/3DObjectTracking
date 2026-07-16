import os
import torch
import numpy as np
from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {DEFAULT_DEVICE}")
if DEFAULT_DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

VIDEO_PATH = "./assets/raw_recording.mp4"
MASK_PATH = "./assets/pear_mask.png"
GRID_SIZE = 30

video = read_video_from_path(VIDEO_PATH)
print(f"Video shape: {video.shape}")  # (T, H, W, 3)
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()  # (1, T, 3, H, W)

segm_mask = np.array(Image.open(MASK_PATH))
segm_mask = torch.from_numpy(segm_mask)[None, None]  # (1, 1, H, W)
print(f"Mask shape: {segm_mask.shape}, unique values: {segm_mask.unique()}")

model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
model = model.to(DEFAULT_DEVICE)
video = video.to(DEFAULT_DEVICE)
segm_mask = segm_mask.to(DEFAULT_DEVICE)

pred_tracks, pred_visibility = model(
    video,
    grid_size=GRID_SIZE,
    grid_query_frame=0,
    segm_mask=segm_mask,
)
print(f"Tracks shape: {pred_tracks.shape}")

os.makedirs("./saved_videos", exist_ok=True)
vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility, query_frame=0, filename="pear_tracking")
print("Saved to ./saved_videos/pear_tracking.mp4")

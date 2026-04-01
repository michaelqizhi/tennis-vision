"""Diagnostic script: why TrackNet V2 labeling wrapper only detects 24.8%.

The hypothesis: the labeling wrapper uses softmax(dim=1)[0, 1] which extracts
the probability of *class 1* from a 256-class output. Class 1 is a near-background
class in the Gaussian heatmap encoding, so the extracted heatmap is essentially
empty. The correct approach is argmax(dim=1) like the main pipeline.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_config
from src.models.tracknet import load_tracknet

cfg = get_config()
VIDEO = ROOT / "tests" / "fixtures" / "tennis_test.mp4"
INP_W = cfg.model.tracknet_input_width   # 640
INP_H = cfg.model.tracknet_input_height  # 360

# ── Load V2 model ──
weights = cfg.resolve_path(cfg.model.tracknet_weights)
print(f"Loading TrackNet V2 from {weights}")
model = load_tracknet(str(weights), "cpu")

# ── Grab 3 consecutive frames from the middle of the video ──
cap = cv2.VideoCapture(str(VIDEO))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start = total // 2  # middle of video for a representative frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start)

frames = []
for _ in range(3):
    ok, f = cap.read()
    if not ok:
        raise RuntimeError("Could not read frames from video")
    frames.append(f)
cap.release()

orig_h, orig_w = frames[0].shape[:2]
print(f"Video: {orig_w}x{orig_h}, using frames {start}-{start+2}")

# ── Preprocess (identical to both pipelines) ──
resized = [cv2.resize(f, (INP_W, INP_H)) for f in frames]
imgs = np.concatenate(resized, axis=2).astype(np.float32) / 255.0
imgs = np.rollaxis(imgs, 2, 0)
inp = torch.from_numpy(np.expand_dims(imgs, 0)).float()

# ── Run model ──
with torch.no_grad():
    out = model(inp)

print(f"\nRaw model output shape: {out.shape}")
print(f"  (expected: [1, 256, {INP_H}*{INP_W}] = [1, 256, {INP_H*INP_W}])")

# ── Method 1: BUGGY — softmax(dim=1)[0, 1] (what labeling wrapper does) ──
softmax_out = out.softmax(dim=1)
buggy_heatmap = softmax_out[0, 1].numpy()  # class-1 probability

print(f"\n=== BUGGY: softmax(dim=1)[0, 1] (class-1 probability) ===")
print(f"  Shape: {buggy_heatmap.shape}")
print(f"  Min:   {buggy_heatmap.min():.6f}")
print(f"  Max:   {buggy_heatmap.max():.6f}")
print(f"  Mean:  {buggy_heatmap.mean():.6f}")

# Scale to uint8 like _weighted_centroid does
buggy_uint8 = (buggy_heatmap * 255).astype(np.uint8)
print(f"  After *255 uint8 — min: {buggy_uint8.min()}, max: {buggy_uint8.max()}")
_, buggy_thresh = cv2.threshold(buggy_uint8.reshape(INP_H, INP_W), 127, 255, cv2.THRESH_BINARY)
n_above = np.count_nonzero(buggy_thresh)
print(f"  Pixels above threshold=127: {n_above} / {INP_H*INP_W} ({100*n_above/(INP_H*INP_W):.2f}%)")

# ── Method 2: CORRECT — argmax(dim=1) (what main pipeline does) ──
argmax_out = out.argmax(dim=1)[0].numpy()  # class index 0-255

print(f"\n=== CORRECT: argmax(dim=1) (class index as heatmap) ===")
print(f"  Shape: {argmax_out.shape}")
print(f"  Min:   {argmax_out.min()}")
print(f"  Max:   {argmax_out.max()}")
print(f"  Mean:  {argmax_out.mean():.4f}")
n_nonzero = np.count_nonzero(argmax_out)
print(f"  Non-zero pixels: {n_nonzero} / {INP_H*INP_W} ({100*n_nonzero/(INP_H*INP_W):.2f}%)")

# Normalize to [0,1] then scale — the correct way for _weighted_centroid
correct_heatmap = argmax_out.astype(np.float32) / 255.0
correct_uint8 = (correct_heatmap * 255).astype(np.uint8)
print(f"  After /255 then *255 uint8 — min: {correct_uint8.min()}, max: {correct_uint8.max()}")
_, correct_thresh = cv2.threshold(correct_uint8.reshape(INP_H, INP_W), 127, 255, cv2.THRESH_BINARY)
n_above = np.count_nonzero(correct_thresh)
print(f"  Pixels above threshold=127: {n_above} / {INP_H*INP_W} ({100*n_above/(INP_H*INP_W):.2f}%)")

# Show what centroid detection would find
moments = cv2.moments(correct_thresh)
if moments["m00"] > 0:
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    scale_x = orig_w / INP_W
    scale_y = orig_h / INP_H
    print(f"  Detected ball at: ({cx*scale_x:.1f}, {cy*scale_y:.1f}) in original coords")
else:
    print("  No ball detected even with correct method")

# ── Also show softmax class distribution at a high-value pixel ──
if argmax_out.max() > 0:
    best_pixel_idx = argmax_out.argmax()
    best_class = argmax_out.flat[best_pixel_idx]
    probs_at_best = softmax_out[0, :, best_pixel_idx].numpy()
    print(f"\n=== Softmax probabilities at best pixel (class={best_class}) ===")
    print(f"  P(class 0) = {probs_at_best[0]:.6f}")
    print(f"  P(class 1) = {probs_at_best[1]:.6f}  ← what buggy code extracts")
    print(f"  P(class {best_class}) = {probs_at_best[best_class]:.6f}  ← actual dominant class")
    top5_idx = np.argsort(probs_at_best)[-5:][::-1]
    print(f"  Top-5 classes: {[(int(i), f'{probs_at_best[i]:.4f}') for i in top5_idx]}")

# ── Run the full buggy vs correct detection on multiple frames ──
print(f"\n{'='*60}")
print("Running detection on first 100 frames: buggy vs correct")
print(f"{'='*60}")

cap = cv2.VideoCapture(str(VIDEO))
frame_buf = []
buggy_det = 0
correct_det = 0
n_frames = min(100, total - 2)

for i in range(n_frames + 2):
    ok, f = cap.read()
    if not ok:
        break
    frame_buf.append(f)

    if len(frame_buf) < 3:
        continue

    r = [cv2.resize(frame_buf[j], (INP_W, INP_H)) for j in range(-3, 0)]
    imgs = np.concatenate(r, axis=2).astype(np.float32) / 255.0
    imgs = np.rollaxis(imgs, 2, 0)
    inp = torch.from_numpy(np.expand_dims(imgs, 0)).float()

    with torch.no_grad():
        out = model(inp)

    # Buggy: softmax class 1
    bm = (out.softmax(dim=1)[0, 1].numpy() * 255).astype(np.uint8).reshape(INP_H, INP_W)
    _, bt = cv2.threshold(bm, 127, 255, cv2.THRESH_BINARY)
    if cv2.moments(bt)["m00"] > 0:
        buggy_det += 1

    # Correct: argmax normalized
    am = out.argmax(dim=1)[0].numpy().astype(np.float32) / 255.0
    am = (am * 255).astype(np.uint8).reshape(INP_H, INP_W)
    _, at = cv2.threshold(am, 127, 255, cv2.THRESH_BINARY)
    if cv2.moments(at)["m00"] > 0:
        correct_det += 1

    # Keep only last 3 frames in buffer
    if len(frame_buf) > 3:
        frame_buf.pop(0)

cap.release()

print(f"  Buggy (softmax[0,1]):  {buggy_det}/{n_frames} = {100*buggy_det/n_frames:.1f}%")
print(f"  Correct (argmax/255):  {correct_det}/{n_frames} = {100*correct_det/n_frames:.1f}%")
print(f"\nConclusion: softmax(dim=1)[0,1] extracts class-1 probability from a")
print(f"256-class model, producing a nearly-empty heatmap. Use argmax instead.")

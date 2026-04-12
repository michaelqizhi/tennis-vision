"""PyTorch Dataset for near-half court keypoint detection.

Loads from labels JSON (produced by scripts/generate_near_half_labels.py),
generates 7 Gaussian heatmap targets, and applies training augmentations using
albumentations with ReplayCompose to detect CoarseDropout occlusion.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# Channel index (0-6) → KP index
# Ordered: singles-L, doubles-L, service-L, center, service-R, doubles-R, singles-R
CHANNEL_TO_KP_IDX: list[int] = [5, 2, 10, 13, 11, 3, 7]
KP_IDX_TO_CHANNEL: dict[int, int] = {kp: ch for ch, kp in enumerate(CHANNEL_TO_KP_IDX)}

# Symmetric keypoint pair channels for horizontal flip
# (left, right) — channel 3 (center) stays
_FLIP_PAIRS: list[tuple[int, int]] = [
    (0, 6),  # KP 5 (SL-BL) ↔ KP 7 (SL-BR) — singles baseline L↔R
    (1, 5),  # KP 2 (BL-BL) ↔ KP 3 (BL-BR) — doubles baseline L↔R
    (2, 4),  # KP 10 (SV-BL) ↔ KP 11 (SV-BR) — service line L↔R
]

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

STRIDE = 8

# Visibility array ordering in labels.json (matches original generation order)
_LABEL_VIS_ORDER: list[int] = [2, 3, 5, 7, 10, 11, 13]


def generate_heatmap(
    kp_x: float,
    kp_y: float,
    width: int,
    height: int,
    sigma: float = 2.5,
    visible: bool = True,
) -> np.ndarray:
    """Generate a single Gaussian heatmap centered on (kp_x, kp_y).

    Args:
        kp_x: Keypoint x-coordinate (pixels, at the target resolution).
        kp_y: Keypoint y-coordinate (pixels, at the target resolution).
        width: Heatmap width in pixels.
        height: Heatmap height in pixels.
        sigma: Gaussian spread in pixels (default 2.5).
        visible: If False, returns an all-zero heatmap.

    Returns:
        Float32 array of shape (height, width) in [0, 1].
    """
    if not visible:
        return np.zeros((height, width), dtype=np.float32)

    yy, xx = np.mgrid[0:height, 0:width]
    heatmap = np.exp(-((xx - kp_x) ** 2 + (yy - kp_y) ** 2) / (2.0 * sigma ** 2))
    return heatmap.astype(np.float32)


def check_coarse_dropout_occlusion(
    replay_data: dict[str, Any],
    keypoints: list[tuple[float, float]],
    visibility: list[float],
    margin: int = 5,
) -> list[float]:
    """Update visibility based on CoarseDropout occlusion.

    Args:
        replay_data: Replay data from albumentations ReplayCompose.
        keypoints: List of (x, y) keypoint positions after augmentation.
        visibility: Current visibility flags (may already be 0 if out of bounds).
        margin: Pixel margin around keypoint to check for occlusion.

    Returns:
        Updated visibility list with occluded keypoints set to 0.
    """
    new_vis = list(visibility)

    # Extract transforms from replay
    transforms = replay_data.get("transforms", [])
    for tfm in transforms:
        if tfm.get("__class_fullname__") != "CoarseDropout":
            continue
        if not tfm.get("applied", False):
            continue

        params = tfm.get("params", {})
        holes = params.get("holes", [])

        # Check each hole against each keypoint
        for hole in holes:
            x1, y1, x2, y2 = hole
            for i, ((kx, ky), v) in enumerate(zip(keypoints, new_vis)):
                if v < 0.5:
                    continue  # Already invisible
                # Check if keypoint is within hole bounds (with margin)
                if (x1 - margin <= kx <= x2 + margin and 
                    y1 - margin <= ky <= y2 + margin):
                    new_vis[i] = 0.0

    return new_vis


class NearHalfCourtDataset(Dataset):
    """Dataset for near-half court keypoint detection.

    Loads from a labels JSON file produced by scripts/generate_near_half_labels.py.

    Each sample dict contains:
        - "image": (3, H, W) float32 tensor, normalized with ImageNet mean/std.
        - "heatmaps": (7, H, W) float32 tensor with Gaussian peaks.
        - "visibility": (7,) float32 tensor (1.0 = visible, 0.0 = off-screen).
        - "metadata": Dict with image_path, video_stem, frame_idx, etc.

    Args:
        labels_json: Path to the labels JSON file.
        images_dir: Root directory containing crop images.
        input_w: Target image width (default 640).
        input_h: Target image height (default 360).
        sigma: Gaussian heatmap spread in pixels (default 2.5).
        augment: If True, apply training augmentations.
    """

    # KP index → channel (0-6)
    KP_IDX_TO_CHANNEL: dict[int, int] = KP_IDX_TO_CHANNEL
    CHANNEL_TO_KP_IDX: list[int] = CHANNEL_TO_KP_IDX
    NUM_KEYPOINTS: int = 7

    def __init__(
        self,
        labels_json: str | Path | None = None,
        images_dir: str | Path | None = None,
        input_w: int = 640,
        input_h: int = 360,
        stride: int = STRIDE,
        sigma: float = 2.5,
        augment: bool = True,
        frames: list[dict[str, Any]] | None = None,
    ) -> None:
        self.labels_json = Path(labels_json) if labels_json else None
        self.images_dir = Path(images_dir) if images_dir else None
        self.input_w = input_w
        self.input_h = input_h
        self.stride = stride
        self.output_w = input_w // stride
        self.output_h = input_h // stride
        self.sigma = sigma
        self.augment = augment

        if frames is not None:
            self.frames: list[dict[str, Any]] = frames
        elif self.labels_json is not None:
            with open(self.labels_json, "r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
            self.frames = data.get("frames", [])
        else:
            raise ValueError("Either labels_json or frames must be provided")

        if not self.frames:
            raise ValueError("No frames found")

        # Build augmentation pipelines
        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()

    def _build_train_transform(self) -> A.ReplayCompose:
        """Build training augmentation pipeline with ReplayCompose."""
        return A.ReplayCompose(
            [
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    scale=(0.95, 1.05),
                    rotate=(-3, 3),
                    fit_output=False,
                    p=0.7,
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.8,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=15,
                    p=0.5,
                ),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                ], p=0.3),
                A.ImageCompression(quality_range=(75, 100), p=0.3),
                A.GaussNoise(std_range=(0.01, 0.03), mean_range=(0, 0), p=0.4),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(8, 32),
                    hole_width_range=(8, 32),
                    fill=0,
                    p=0.3,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                remove_invisible=False,
            ),
        )

    def _build_val_transform(self) -> A.Compose:
        """Build validation pipeline (resize + normalize only)."""
        return A.Compose(
            [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                remove_invisible=False,
            ),
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict]:
        entry = self.frames[idx]

        # --- Load image ---
        img_rel = Path(entry["image_path"])
        if self.images_dir is not None and not img_rel.is_absolute():
            img_path = self.images_dir / img_rel
        else:
            img_path = img_rel
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # Resize to model input size
        image = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Load keypoints ---
        kp_dict: dict[str, list[float]] = entry.get("keypoints", {})
        vis_list: list[float] = entry.get("visibility", [1.0] * self.NUM_KEYPOINTS)

        # Scale keypoints if label coords differ from input resolution
        crop_w = entry.get("crop_w", self.input_w)
        crop_h = entry.get("crop_h", self.input_h)
        scale_x = self.input_w / crop_w
        scale_y = self.input_h / crop_h

        # Map visibility from label ordering to KP index
        kp_vis: dict[int, float] = {}
        for i, kp_idx in enumerate(_LABEL_VIS_ORDER):
            kp_vis[kp_idx] = vis_list[i] if i < len(vis_list) else 1.0

        # Build ordered keypoint and visibility lists (channel 0-6)
        keypoints: list[tuple[float, float]] = []
        visibility: list[float] = []
        for ch, kp_idx in enumerate(self.CHANNEL_TO_KP_IDX):
            key = str(kp_idx)
            vis = kp_vis.get(kp_idx, 0.0)
            if key in kp_dict and vis > 0.5:
                kx, ky = kp_dict[key]
                keypoints.append((float(kx) * scale_x, float(ky) * scale_y))
                visibility.append(1.0)
            else:
                keypoints.append((0.0, 0.0))
                visibility.append(0.0)

        # --- Apply augmentations ---
        if self.augment:
            transformed = self.train_transform(image=image, keypoints=keypoints)
            image_aug = transformed["image"]
            keypoints_aug = list(transformed["keypoints"])
            replay_data = transformed["replay"]

            # Update visibility for out-of-bounds keypoints
            new_vis = []
            for (kx, ky), v in zip(keypoints_aug, visibility):
                if v < 0.5:
                    new_vis.append(0.0)
                elif kx < 0 or kx >= self.input_w or ky < 0 or ky >= self.input_h:
                    new_vis.append(0.0)
                else:
                    new_vis.append(1.0)

            # Update visibility for CoarseDropout occlusion
            new_vis = check_coarse_dropout_occlusion(
                replay_data, keypoints_aug, new_vis
            )

            # Remap keypoints if HorizontalFlip was applied
            if self._was_flipped(replay_data):
                keypoints_aug, new_vis = self._remap_flip_keypoints(
                    keypoints_aug, new_vis
                )

            keypoints_final = keypoints_aug
            visibility_final = new_vis
        else:
            transformed = self.val_transform(image=image, keypoints=keypoints)
            image_aug = transformed["image"]
            keypoints_final = list(transformed["keypoints"])
            visibility_final = visibility

        # --- Generate heatmap targets at full input resolution ---
        # Targets are generated at the model's output resolution (input_w × input_h)
        # so no upsampling is needed in the training loop. sigma is in full-res pixels.
        heatmaps = np.stack(
            [
                generate_heatmap(
                    kx, ky,
                    self.input_w, self.input_h,
                    sigma=self.sigma, visible=v > 0.5,
                )
                for (kx, ky), v in zip(keypoints_final, visibility_final)
            ],
            axis=0,
        )  # (7, input_h, input_w)

        # --- Convert to tensors ---
        heatmap_tensor = torch.from_numpy(heatmaps)
        vis_tensor = torch.tensor(visibility_final, dtype=torch.float32)
        # Store keypoint coordinates for accurate reproj error computation
        kp_coords = torch.tensor(
            [list(kp) for kp in keypoints_final], dtype=torch.float32
        )  # (7, 2) in full image space

        # Metadata
        video_stem = entry.get("video_stem", "")
        if not video_stem or video_stem == "?":
            parts = Path(entry["image_path"]).parts
            video_stem = parts[-2] if len(parts) >= 2 else "manual"

        metadata = {
            "image_path": str(img_path),
            "video_stem": video_stem,
            "frame_idx": entry.get("frame_idx", -1),
            "source": entry.get("source", ""),
        }

        return {
            "image": image_aug,
            "heatmaps": heatmap_tensor,
            "visibility": vis_tensor,
            "keypoints_coords": kp_coords,
            "metadata": metadata,
        }

    def _was_flipped(self, replay_data: dict[str, Any]) -> bool:
        """Check if HorizontalFlip was applied in the replay."""
        transforms = replay_data.get("transforms", [])
        for tfm in transforms:
            if tfm.get("__class_fullname__") == "HorizontalFlip" and tfm.get("applied", False):
                return True
        return False

    def _remap_flip_keypoints(
        self,
        keypoints: list[tuple[float, float]],
        visibility: list[float],
    ) -> tuple[list[tuple[float, float]], list[float]]:
        """Remap keypoints after horizontal flip (swap left/right pairs)."""
        new_kps = list(keypoints)
        new_vis = list(visibility)

        for left_ch, right_ch in _FLIP_PAIRS:
            new_kps[left_ch], new_kps[right_ch] = new_kps[right_ch], new_kps[left_ch]
            new_vis[left_ch], new_vis[right_ch] = new_vis[right_ch], new_vis[left_ch]

        return new_kps, new_vis


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LABELS = _PROJECT_ROOT / "data" / "near_half_train" / "labels.json"
_DEFAULT_IMAGES_DIR = _PROJECT_ROOT / "data" / "near_half_train"

# Hardcoded test/val splits — DO NOT change once set.
# Test set: evaluated ≤3 times (baseline, post-tuning, pre-deploy).
TEST_VIDEO_STEMS: frozenset[str] = frozenset({
    "vid30_gopro_florida",             # green hard court, GoPro distortion
    "vid29_usta35_conlon_rustagi",     # blue outdoor, hard shadows
    "vid16_college_zheng_harvard",     # indoor blue, watermark
    "vid34_grass_bellavista_night",    # night grass, blur — hardest conditions
})

VAL_VIDEO_STEMS: frozenset[str] = frozenset({
    "vid21_red_clay_joao",            # red clay indoor, warm light
    "vid27_grass_match_2024",         # grass, strong sun/shadow split
    "vid22_usta40_doubles_irish",     # indoor doubles
    "vid31_dusk_fall45",              # dusk/low light, SwingVision overlay
    "vid18_usta40_clay_4k",           # green/worn surface, overcast
})


def _get_video_stem(frame: dict) -> str:
    """Extract video stem from a frame entry."""
    vs = frame.get("video_stem", "")
    if vs and vs != "?":
        return vs
    img_path = Path(frame["image_path"])
    if len(img_path.parts) >= 2:
        return img_path.parts[-2]
    return "unknown"


def _load_and_filter_frames(
    labels_json: str | Path,
    images_dir: str | Path,
) -> list[dict[str, Any]]:
    """Load frames from labels JSON and filter out missing images."""
    with open(labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_frames = data.get("frames", [])

    images_dir_path = Path(images_dir)
    frames = []
    for fr in all_frames:
        img_rel = Path(fr["image_path"])
        img_path = images_dir_path / img_rel if not img_rel.is_absolute() else img_rel
        if img_path.exists():
            frames.append(fr)
    return frames


def create_train_val_datasets(
    labels_json: str | Path | None = None,
    images_dir: str | Path | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    **dataset_kwargs: Any,
) -> tuple[NearHalfCourtDataset, NearHalfCourtDataset]:
    """Create train/val datasets using the hardcoded split.

    Test videos are always excluded. Val videos use the hardcoded
    VAL_VIDEO_STEMS. The val_ratio parameter is only used as a fallback
    if VAL_VIDEO_STEMS is empty.

    Args:
        labels_json: Path to labels JSON (default: data/near_half_train/labels.json).
        images_dir: Root directory for crop images (default: data/near_half_train).
        val_ratio: Fallback fraction if VAL_VIDEO_STEMS is empty (default 0.2).
        seed: Random seed for fallback split (default 42).
        **dataset_kwargs: Additional arguments passed to NearHalfCourtDataset.

    Returns:
        (train_dataset, val_dataset) tuple.
    """
    if labels_json is None:
        labels_json = _DEFAULT_LABELS
    if images_dir is None:
        images_dir = _DEFAULT_IMAGES_DIR

    frames = _load_and_filter_frames(labels_json, images_dir)

    # Always exclude test videos
    frames = [f for f in frames if _get_video_stem(f) not in TEST_VIDEO_STEMS]

    if VAL_VIDEO_STEMS:
        val_stems = VAL_VIDEO_STEMS
    else:
        # Fallback: random split by video stem
        video_stems = sorted(set(_get_video_stem(f) for f in frames))
        rng = random.Random(seed)
        rng.shuffle(video_stems)
        n_val = max(1, int(len(video_stems) * val_ratio))
        val_stems = set(video_stems[:n_val])

    train_frames = [f for f in frames if _get_video_stem(f) not in val_stems]
    val_frames = [f for f in frames if _get_video_stem(f) in val_stems]

    train_dataset = NearHalfCourtDataset(
        images_dir=images_dir,
        augment=True,
        frames=train_frames,
        **dataset_kwargs,
    )
    val_dataset = NearHalfCourtDataset(
        images_dir=images_dir,
        augment=False,
        frames=val_frames,
        **dataset_kwargs,
    )

    return train_dataset, val_dataset


def create_test_dataset(
    labels_json: str | Path | None = None,
    images_dir: str | Path | None = None,
    **dataset_kwargs: Any,
) -> NearHalfCourtDataset:
    """Create the held-out test dataset from TEST_VIDEO_STEMS.

    Args:
        labels_json: Path to labels JSON (default: data/near_half_train/labels.json).
        images_dir: Root directory for crop images (default: data/near_half_train).
        **dataset_kwargs: Additional arguments passed to NearHalfCourtDataset.

    Returns:
        Test dataset (no augmentation).
    """
    if labels_json is None:
        labels_json = _DEFAULT_LABELS
    if images_dir is None:
        images_dir = _DEFAULT_IMAGES_DIR

    frames = _load_and_filter_frames(labels_json, images_dir)
    test_frames = [f for f in frames if _get_video_stem(f) in TEST_VIDEO_STEMS]

    return NearHalfCourtDataset(
        images_dir=images_dir,
        augment=False,
        frames=test_frames,
        **dataset_kwargs,
    )

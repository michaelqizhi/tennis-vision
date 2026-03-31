"""Tennis Vision project configuration."""

import os
from pathlib import Path
from dataclasses import dataclass, field

import torch
import yaml


@dataclass
class ModelConfig:
    """Configuration for ML model inference."""
    tracknet_weights: str = "weights/tracknet.pt"
    court_weights: str = "weights/court_detector.pt"
    # TrackNet inference resolution (must match training resolution)
    tracknet_input_width: int = 640
    tracknet_input_height: int = 360
    # Number of output classes for TrackNet heatmap
    tracknet_out_channels: int = 256
    # Court detector inference resolution
    court_input_width: int = 640
    court_input_height: int = 360
    court_out_channels: int = 15


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    max_frames: int = 0  # 0 = no limit
    output_codec: str = "mp4v"
    output_ext: str = ".mp4"


@dataclass
class BallTrackingConfig:
    """Configuration for ball tracking postprocessing."""
    confidence_threshold: int = 127
    hough_min_dist: int = 1
    hough_param1: int = 50
    hough_param2: int = 2
    hough_min_radius: int = 2
    hough_max_radius: int = 7
    # Outlier removal
    max_outlier_dist: float = 100.0
    # Interpolation
    max_gap: int = 4
    max_dist_gap: float = 80.0
    min_track_length: int = 5
    # Visualization
    trace_length: int = 7
    ball_color: tuple = (0, 255, 0)  # BGR green
    ball_radius: int = 5


@dataclass
class CourtDetectionConfig:
    """Configuration for court keypoint detection postprocessing."""
    heatmap_threshold: int = 130
    min_radius: int = 8
    max_radius: int = 35
    refine_crop_size: int = 40
    # Minimum keypoints for valid detection
    min_keypoints: int = 4


@dataclass
class Config:
    """Top-level project configuration."""
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    ball_tracking: BallTrackingConfig = field(default_factory=BallTrackingConfig)
    court_detection: CourtDetectionConfig = field(default_factory=CourtDetectionConfig)

    @property
    def weights_dir(self) -> Path:
        return self.project_root / "weights"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output"

    def resolve_path(self, rel_path: str) -> Path:
        """Resolve a path relative to the project root."""
        p = Path(rel_path)
        if p.is_absolute():
            return p
        return self.project_root / p


def load_config(config_path: str | None = None) -> Config:
    """Load configuration, optionally overriding from a YAML file.

    Args:
        config_path: Optional path to a config.yaml file.

    Returns:
        Config instance with defaults and any overrides applied.
    """
    cfg = Config()

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            overrides = yaml.safe_load(f) or {}

        if "device" in overrides:
            cfg.device = overrides["device"]
        if "model" in overrides:
            for k, v in overrides["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
        if "video" in overrides:
            for k, v in overrides["video"].items():
                if hasattr(cfg.video, k):
                    setattr(cfg.video, k, v)
        if "ball_tracking" in overrides:
            for k, v in overrides["ball_tracking"].items():
                if hasattr(cfg.ball_tracking, k):
                    setattr(cfg.ball_tracking, k, v)
        if "court_detection" in overrides:
            for k, v in overrides["court_detection"].items():
                if hasattr(cfg.court_detection, k):
                    setattr(cfg.court_detection, k, v)

    return cfg


# Singleton default config
_default_config: Config | None = None


def get_config() -> Config:
    """Get the default project configuration (singleton)."""
    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config

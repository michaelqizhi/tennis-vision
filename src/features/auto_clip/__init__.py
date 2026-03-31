"""Auto-clip — rally boundary detection and video clipping.

Detects rally boundaries from ball tracking data and extracts
individual rally clips from source video.
"""

from src.features.auto_clip.rally_detector import (
    Rally,
    RallyDetectorConfig,
    detect_rallies,
    get_rally_clips,
)
from src.features.auto_clip.clipper import clip_rally, clip_rallies

__all__ = [
    "Rally",
    "RallyDetectorConfig",
    "detect_rallies",
    "get_rally_clips",
    "clip_rally",
    "clip_rallies",
]

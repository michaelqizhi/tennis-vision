"""Factory for creating ball tracker instances based on config.

Provides a single entry point to get the correct tracker (V2 or V4)
based on the ``tracknet_version`` setting in config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.features.ball_tracking.detector import BallTracker
    from src.features.ball_tracking.tracknetv4_tracker import BallTrackerV4

from src.config import Config, get_config


def create_ball_tracker(
    config: Config | None = None,
) -> "BallTracker | BallTrackerV4":
    """Create a ball tracker instance based on config.

    Uses ``config.model.tracknet_version`` to select the model:
      - ``"v2"``: Original TrackNet V2 (640×360, argmax heatmap)
      - ``"v4"``: TrackNet V4 with motion attention (512×288, sigmoid heatmap)

    Args:
        config: Project configuration. Uses default if not provided.

    Returns:
        A tracker instance (BallTracker for V2, BallTrackerV4 for V4).

    Raises:
        ValueError: If the configured version is not supported.
    """
    cfg = config or get_config()
    version = cfg.model.tracknet_version.lower().strip()

    if version == "v2":
        from src.features.ball_tracking.detector import BallTracker
        return BallTracker(cfg)
    elif version == "v4":
        from src.features.ball_tracking.tracknetv4_tracker import BallTrackerV4
        return BallTrackerV4(cfg)
    else:
        raise ValueError(
            f"Unsupported tracknet_version={version!r}. "
            f"Supported versions: 'v2', 'v4'."
        )

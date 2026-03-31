"""Video file writer using OpenCV."""

import cv2
import numpy as np


class VideoWriter:
    """Write frames to a video file.

    Args:
        path: Output video file path.
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        codec: FourCC codec string (default: 'mp4v').
    """

    def __init__(self, path: str, fps: float, width: int, height: int,
                 codec: str = "mp4v"):
        self.path = path
        self.fps = fps
        self.width = width
        self.height = height
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {path}")

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single BGR frame.

        Args:
            frame: BGR image as numpy array. Will be resized if dimensions
                   don't match the writer's configured size.
        """
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self._writer.write(frame)

    def write_frames(self, frames: list[np.ndarray]) -> None:
        """Write multiple frames sequentially."""
        for frame in frames:
            self.write_frame(frame)

    def release(self) -> None:
        """Release the video writer resource."""
        self._writer.release()

    def __del__(self) -> None:
        self.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *args) -> None:
        self.release()

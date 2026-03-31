"""Video file reader using OpenCV."""

from typing import Iterator
import cv2
import numpy as np


class VideoReader:
    """Read frames from a video file.

    Args:
        path: Path to the video file.
        max_frames: Maximum number of frames to read (0 = no limit).
    """

    def __init__(self, path: str, max_frames: int = 0):
        self.path = path
        self.max_frames = max_frames
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

    @property
    def fps(self) -> float:
        """Frames per second of the source video."""
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        """Total number of frames (estimate from container metadata)."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Iterate over frames one at a time (memory-efficient).

        Yields:
            BGR frame as a numpy array.
        """
        count = 0
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame
            count += 1
            if self.max_frames > 0 and count >= self.max_frames:
                break

    def read_sampled_frames(self, max_samples: int = 20) -> list[np.ndarray]:
        """Read a small number of evenly-spaced frames by seeking.

        Useful for tasks like court detection that only need a few
        representative frames rather than every frame in the video.

        Args:
            max_samples: Maximum number of frames to read.

        Returns:
            List of BGR frames (at most max_samples).
        """
        total = self.frame_count
        if total <= 0:
            return []

        step = max(1, total // max_samples)
        frames: list[np.ndarray] = []
        for idx in range(0, total, step):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self._cap.read()
            if ret:
                frames.append(frame)
            if len(frames) >= max_samples:
                break

        # Reset position for any subsequent reads
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frames

    def release(self) -> None:
        """Release the video capture resource."""
        self._cap.release()

    def __del__(self) -> None:
        self.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *args) -> None:
        self.release()

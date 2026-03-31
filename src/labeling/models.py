"""Base detector interface for the labeling pipeline.

All ball detection models must implement the BaseDetector abstract class
so the inference runner can treat them uniformly.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Abstract base class for ball detectors.

    Each detector wraps a single ML model and provides a uniform
    ``detect`` interface that returns the ball position in pixel
    coordinates for a given frame.

    Subclasses must implement ``load``, ``detect``, and ``unload``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this detector (e.g. 'tracknet_v2')."""

    @abstractmethod
    def load(self, device: str = "cpu") -> None:
        """Load model weights into memory.

        Args:
            device: PyTorch device string ('cpu' or 'cuda').
        """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[tuple[float, float, float]]:
        """Detect the tennis ball in a single frame.

        Args:
            frame: BGR image as a numpy array (H, W, 3), uint8.

        Returns:
            Tuple of (x, y, confidence) in *original frame* pixel
            coordinates, or ``None`` if no ball is detected.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory (free VRAM)."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"

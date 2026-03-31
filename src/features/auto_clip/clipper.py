"""Video clipper — extract rally segments from source video.

Clips individual rallies from the source video using OpenCV's
frame-by-frame seeking. For large videos, consider using ffmpeg
subprocess calls for faster seeking.
"""

from pathlib import Path

import cv2

from src.video.writer import VideoWriter


def clip_rally(
    input_path: str,
    output_path: str,
    start_frame: int,
    end_frame: int,
    codec: str = "mp4v",
) -> str:
    """Extract a single rally segment from a video file.

    Args:
        input_path: Path to source video.
        output_path: Path for the output clip.
        start_frame: First frame to include.
        end_frame: Last frame to include.
        codec: FourCC codec for output video.

    Returns:
        Path to the saved clip file.
    """
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    if end_frame < start_frame:
        raise ValueError(
            f"end_frame ({end_frame}) must be >= start_frame ({start_frame})"
        )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        writer = VideoWriter(output_path, fps, width, height, codec=codec)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write_frame(frame)
        finally:
            writer.release()
    finally:
        cap.release()

    return output_path


def clip_rallies(
    input_path: str,
    output_dir: str,
    clip_ranges: list[tuple[int, int]],
    name_prefix: str = "rally",
    codec: str = "mp4v",
) -> list[str]:
    """Extract multiple rally clips from a source video.

    Args:
        input_path: Path to source video.
        output_dir: Directory for output clips.
        clip_ranges: List of (start_frame, end_frame) tuples.
        name_prefix: Prefix for clip filenames.
        codec: FourCC codec for output video.

    Returns:
        List of paths to saved clip files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: list[str] = []
    for i, (start, end) in enumerate(clip_ranges, 1):
        clip_name = f"{name_prefix}_{i:03d}.mp4"
        clip_path = str(out_dir / clip_name)
        clip_rally(input_path, clip_path, start, end, codec=codec)
        clip_paths.append(clip_path)

    return clip_paths

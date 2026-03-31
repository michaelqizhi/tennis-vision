"""Renderer — composite court diagrams with heatmaps and save outputs.

Provides high-level functions that combine court drawing + heatmap plotting
and output to PNG files or video frame overlays.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

from src.features.court_detect.court_template import ITF_DIMS, CourtDimensions
from src.features.visualization.court_diagram import (
    CourtStyle,
    draw_court,
    draw_service_boxes,
)
from src.features.visualization.heatmap import (
    HeatmapStyle,
    filter_court_positions,
    plot_heatmap,
    plot_scatter,
    add_position_count,
)


def render_shot_heatmap(
    positions: list[tuple[float, float]],
    output_path: str,
    title: str = "Shot Placement Heatmap",
    region: str = "full",
    court_style: CourtStyle | None = None,
    heatmap_style: HeatmapStyle | None = None,
    dims: CourtDimensions | None = None,
    use_kde: bool = True,
) -> str:
    """Render a shot placement heatmap and save to PNG.

    Args:
        positions: Ball landing positions in court coordinates (meters).
        output_path: Path for the output PNG file.
        title: Title displayed above the court.
        region: Court region filter ('full', 'service_boxes', etc.).
        court_style: Court diagram styling.
        heatmap_style: Heatmap overlay styling.
        dims: Court dimensions.
        use_kde: If True, use KDE heatmap. If False, scatter only.

    Returns:
        Path to the saved PNG file.
    """
    dims = dims or ITF_DIMS
    filtered = filter_court_positions(positions, dims, region)

    fig, ax = draw_court(dims, court_style, title=title)

    if use_kde:
        plot_heatmap(ax, filtered, dims, heatmap_style)
    else:
        plot_scatter(ax, filtered, heatmap_style)

    add_position_count(ax, filtered, dims)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path


def render_serve_placement(
    positions: list[tuple[float, float]],
    output_path: str,
    title: str = "Serve Placement",
    court_style: CourtStyle | None = None,
    heatmap_style: HeatmapStyle | None = None,
    dims: CourtDimensions | None = None,
    use_kde: bool = True,
) -> str:
    """Render serve placement heatmap filtered to service boxes.

    Args:
        positions: Serve landing positions in court coordinates (meters).
        output_path: Path for the output PNG file.
        title: Title displayed above the court.
        court_style: Court diagram styling.
        heatmap_style: Heatmap overlay styling.
        dims: Court dimensions.
        use_kde: If True, use KDE heatmap. If False, scatter only.

    Returns:
        Path to the saved PNG file.
    """
    dims = dims or ITF_DIMS
    filtered = filter_court_positions(positions, dims, region="service_boxes")

    fig, ax = draw_court(dims, court_style, title=title)
    draw_service_boxes(ax, dims)

    if use_kde:
        plot_heatmap(ax, filtered, dims, heatmap_style)
    else:
        plot_scatter(ax, filtered, heatmap_style)

    add_position_count(ax, filtered, dims)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path


def render_court_overlay(
    positions: list[tuple[float, float]],
    width: int = 400,
    height: int = 700,
    dims: CourtDimensions | None = None,
    court_style: CourtStyle | None = None,
    heatmap_style: HeatmapStyle | None = None,
    use_kde: bool = True,
) -> np.ndarray:
    """Render a court heatmap as a BGR image suitable for video overlay.

    Args:
        positions: Ball positions in court coordinates (meters).
        width: Output image width in pixels.
        height: Output image height in pixels.
        dims: Court dimensions.
        court_style: Court diagram styling.
        heatmap_style: Heatmap overlay styling.
        use_kde: If True, use KDE heatmap.

    Returns:
        BGR image as numpy array (height x width x 3).
    """
    dims = dims or ITF_DIMS
    style = court_style or CourtStyle(
        figsize=(width / 100, height / 100), dpi=100,
    )

    fig, ax = draw_court(dims, style)

    if use_kde:
        plot_heatmap(ax, positions, dims, heatmap_style)
    else:
        plot_scatter(ax, positions, heatmap_style)

    # Render figure to numpy array
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    # RGBA → RGB
    buf = buf[:, :, :3].copy()
    plt.close(fig)

    # Resize and convert RGB → BGR
    img = cv2.resize(buf, (width, height), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def embed_overlay_in_frame(
    frame: np.ndarray,
    overlay: np.ndarray,
    position: str = "bottom-right",
    margin: int = 10,
    alpha: float = 0.85,
) -> np.ndarray:
    """Embed a court overlay image into a video frame corner.

    Args:
        frame: BGR video frame.
        overlay: BGR overlay image (court heatmap).
        position: Corner position ('top-left', 'top-right',
                  'bottom-left', 'bottom-right').
        margin: Pixel margin from frame edge.
        alpha: Opacity of the overlay (0=transparent, 1=opaque).

    Returns:
        Frame with overlay embedded.
    """
    fh, fw = frame.shape[:2]
    oh, ow = overlay.shape[:2]

    # Ensure overlay fits
    if oh + 2 * margin > fh or ow + 2 * margin > fw:
        scale = min((fh - 2 * margin) / oh, (fw - 2 * margin) / ow, 1.0)
        if scale < 0.1:
            return frame
        overlay = cv2.resize(
            overlay,
            (int(ow * scale), int(oh * scale)),
            interpolation=cv2.INTER_AREA,
        )
        oh, ow = overlay.shape[:2]

    valid_positions = {
        "top-left": (margin, margin),
        "top-right": (margin, fw - ow - margin),
        "bottom-left": (fh - oh - margin, margin),
        "bottom-right": (fh - oh - margin, fw - ow - margin),
    }
    if position not in valid_positions:
        valid = ", ".join(sorted(valid_positions.keys()))
        raise ValueError(f"Invalid position '{position}'. Must be one of: {valid}")
    y0, x0 = valid_positions[position]

    # Clamp to valid range
    y0 = max(0, min(y0, fh - oh))
    x0 = max(0, min(x0, fw - ow))

    roi = frame[y0:y0 + oh, x0:x0 + ow]
    blended = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0)
    result = frame.copy()
    result[y0:y0 + oh, x0:x0 + ow] = blended
    return result

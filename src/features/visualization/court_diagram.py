"""Court diagram drawing — 2D top-down tennis court using matplotlib.

Draws an ITF-standard tennis court at real-world scale (meters) with
configurable styling. Used as the base layer for heatmap overlays.
"""

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from src.features.court_detect.court_template import ITF_DIMS, CourtDimensions


@dataclass
class CourtStyle:
    """Visual styling for the court diagram."""
    court_color: str = "#2D6A4F"       # dark green
    line_color: str = "white"
    line_width: float = 2.0
    background_color: str = "#1B4332"  # darker green surround
    net_color: str = "#CCCCCC"
    net_width: float = 1.5
    figsize: tuple[float, float] = (8, 14)  # width, height in inches
    dpi: int = 150
    margin_meters: float = 2.0        # margin around court


def draw_court(
    dims: CourtDimensions | None = None,
    style: CourtStyle | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Draw a 2D top-down tennis court diagram.

    Coordinates use meters with origin at court center (net midpoint).
    X-axis: left-right. Y-axis: near baseline to far baseline.

    Args:
        dims: Court dimensions. Defaults to ITF standard.
        style: Visual styling. Defaults to green court.
        title: Optional title above the court.

    Returns:
        (figure, axes) tuple for further annotation.
    """
    dims = dims or ITF_DIMS
    style = style or CourtStyle()

    fig, ax = plt.subplots(1, 1, figsize=style.figsize, dpi=style.dpi)
    fig.patch.set_facecolor(style.background_color)
    ax.set_facecolor(style.background_color)

    hw = dims.doubles_width / 2.0
    sw = dims.singles_width / 2.0
    hl = dims.court_length / 2.0
    sl = dims.service_line_distance

    # Draw court surface (doubles area)
    court_rect = patches.Rectangle(
        (-hw, -hl), dims.doubles_width, dims.court_length,
        linewidth=0, facecolor=style.court_color,
    )
    ax.add_patch(court_rect)

    lw = style.line_width
    lc = style.line_color

    # Baselines
    ax.plot([-hw, hw], [-hl, -hl], color=lc, linewidth=lw)
    ax.plot([-hw, hw], [hl, hl], color=lc, linewidth=lw)

    # Doubles sidelines
    ax.plot([-hw, -hw], [-hl, hl], color=lc, linewidth=lw)
    ax.plot([hw, hw], [-hl, hl], color=lc, linewidth=lw)

    # Singles sidelines
    ax.plot([-sw, -sw], [-hl, hl], color=lc, linewidth=lw)
    ax.plot([sw, sw], [-hl, hl], color=lc, linewidth=lw)

    # Service lines
    ax.plot([-sw, sw], [-sl, -sl], color=lc, linewidth=lw)
    ax.plot([-sw, sw], [sl, sl], color=lc, linewidth=lw)

    # Center service line
    ax.plot([0, 0], [-sl, sl], color=lc, linewidth=lw)

    # Center marks on baselines
    center_mark_len = 0.1
    ax.plot([0, 0], [-hl, -hl + center_mark_len], color=lc, linewidth=lw)
    ax.plot([0, 0], [hl - center_mark_len, hl], color=lc, linewidth=lw)

    # Net
    ax.plot([-hw - 0.3, hw + 0.3], [0, 0],
            color=style.net_color, linewidth=style.net_width,
            linestyle="--", alpha=0.8)

    # Set limits with margin
    m = style.margin_meters
    ax.set_xlim(-hw - m, hw + m)
    ax.set_ylim(-hl - m, hl + m)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)

    return fig, ax


def draw_service_boxes(
    ax: Axes,
    dims: CourtDimensions | None = None,
    highlight_color: str = "#3D8B5F",
    alpha: float = 0.3,
) -> None:
    """Highlight the four service boxes on the court.

    Args:
        ax: Matplotlib axes from draw_court().
        dims: Court dimensions.
        highlight_color: Fill color for service boxes.
        alpha: Transparency of the highlight.
    """
    dims = dims or ITF_DIMS
    sw = dims.singles_width / 2.0
    sl = dims.service_line_distance

    for x0, y0, x1, y1 in [
        (-sw, -sl, 0, 0),     # far-side ad box
        (0, -sl, sw, 0),      # far-side deuce box
        (-sw, 0, 0, sl),      # near-side deuce box
        (0, 0, sw, sl),       # near-side ad box
    ]:
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=0, facecolor=highlight_color, alpha=alpha,
        )
        ax.add_patch(rect)

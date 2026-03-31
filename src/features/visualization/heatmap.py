"""Heatmap generation — plot ball landing positions on a court diagram.

Supports both scatter plots and kernel density estimation (KDE) heatmaps.
Can filter positions to specific court regions (e.g., service boxes only).
"""

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from src.features.court_detect.court_template import ITF_DIMS, CourtDimensions


@dataclass
class HeatmapStyle:
    """Visual styling for heatmap overlays."""
    colormap: str = "YlOrRd"
    scatter_color: str = "#FF6B35"
    scatter_size: float = 30.0
    scatter_alpha: float = 0.6
    kde_alpha: float = 0.7
    kde_levels: int = 50
    point_edge_color: str = "white"
    point_edge_width: float = 0.5


def filter_court_positions(
    positions: list[tuple[float, float]],
    dims: CourtDimensions | None = None,
    region: str = "full",
) -> list[tuple[float, float]]:
    """Filter court positions to a specific region.

    Args:
        positions: List of (x_meters, y_meters) court coordinates.
        dims: Court dimensions.
        region: One of 'full', 'service_boxes', 'near_service' (y>0),
                'far_service' (y<0), 'near_half' (y>0), 'far_half' (y<0).

    Returns:
        Filtered positions within the specified region.
    """
    dims = dims or ITF_DIMS
    hw = dims.doubles_width / 2.0
    sw = dims.singles_width / 2.0
    hl = dims.court_length / 2.0
    sl = dims.service_line_distance

    def in_court(x: float, y: float) -> bool:
        return -hw <= x <= hw and -hl <= y <= hl

    def in_service_boxes(x: float, y: float) -> bool:
        return -sw <= x <= sw and -sl <= y <= sl

    def in_near_service(x: float, y: float) -> bool:
        return -sw <= x <= sw and 0 <= y <= sl

    def in_far_service(x: float, y: float) -> bool:
        return -sw <= x <= sw and -sl <= y <= 0

    def in_near_half(x: float, y: float) -> bool:
        return -hw <= x <= hw and 0 <= y <= hl

    def in_far_half(x: float, y: float) -> bool:
        return -hw <= x <= hw and -hl <= y <= 0

    filters = {
        "full": in_court,
        "service_boxes": in_service_boxes,
        "near_service": in_near_service,
        "far_service": in_far_service,
        "near_half": in_near_half,
        "far_half": in_far_half,
    }
    fn = filters.get(region)
    if fn is None:
        valid = ", ".join(sorted(filters.keys()))
        raise ValueError(f"Invalid region '{region}'. Must be one of: {valid}")
    return [(x, y) for x, y in positions if fn(x, y)]


def plot_scatter(
    ax: Axes,
    positions: list[tuple[float, float]],
    style: HeatmapStyle | None = None,
    label: str | None = None,
) -> None:
    """Plot ball positions as a scatter overlay on the court.

    Args:
        ax: Matplotlib axes (from draw_court()).
        positions: List of (x_meters, y_meters) court coordinates.
        style: Visual styling.
        label: Optional legend label.
    """
    if not positions:
        return

    style = style or HeatmapStyle()
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    ax.scatter(
        xs, ys,
        c=style.scatter_color,
        s=style.scatter_size,
        alpha=style.scatter_alpha,
        edgecolors=style.point_edge_color,
        linewidths=style.point_edge_width,
        zorder=5,
        label=label,
    )


def plot_heatmap(
    ax: Axes,
    positions: list[tuple[float, float]],
    dims: CourtDimensions | None = None,
    style: HeatmapStyle | None = None,
    grid_resolution: int = 100,
) -> None:
    """Plot a KDE heatmap of ball landing positions on the court.

    Uses Gaussian kernel density estimation to create a smooth density surface.

    Args:
        ax: Matplotlib axes (from draw_court()).
        positions: List of (x_meters, y_meters) court coordinates.
        dims: Court dimensions for grid bounds.
        style: Visual styling.
        grid_resolution: Grid resolution for KDE evaluation.
    """
    if len(positions) < 3:
        # Too few points for KDE — fall back to scatter
        plot_scatter(ax, positions, style)
        return

    dims = dims or ITF_DIMS
    style = style or HeatmapStyle()

    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])

    # Check for degenerate data (all points at same location)
    if np.std(xs) < 1e-6 and np.std(ys) < 1e-6:
        plot_scatter(ax, positions, style)
        return

    try:
        kde = gaussian_kde(np.vstack([xs, ys]))
    except np.linalg.LinAlgError:
        plot_scatter(ax, positions, style)
        return

    hw = dims.doubles_width / 2.0
    hl = dims.court_length / 2.0

    xi = np.linspace(-hw, hw, grid_resolution)
    yi = np.linspace(-hl, hl, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    zi = kde(np.vstack([xi_grid.ravel(), yi_grid.ravel()]))
    zi = zi.reshape(xi_grid.shape)

    ax.contourf(
        xi_grid, yi_grid, zi,
        levels=style.kde_levels,
        cmap=style.colormap,
        alpha=style.kde_alpha,
        zorder=3,
    )

    # Also plot actual points on top
    plot_scatter(ax, positions, style)


def add_position_count(
    ax: Axes,
    positions: list[tuple[float, float]],
    dims: CourtDimensions | None = None,
) -> None:
    """Add a text annotation showing the number of positions plotted.

    Args:
        ax: Matplotlib axes.
        positions: The positions being displayed.
        dims: Court dimensions.
    """
    dims = dims or ITF_DIMS
    hl = dims.court_length / 2.0
    hw = dims.doubles_width / 2.0
    margin = 1.5

    ax.text(
        hw + margin, -hl - margin + 0.5,
        f"n = {len(positions)}",
        color="white", fontsize=10, ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
    )

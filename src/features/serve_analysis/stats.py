"""Serve statistics — aggregate stats from serve detection results.

Computes 1st serve percentage, double fault count, ace count,
serve placement distribution, and per-serve details.
"""

from dataclasses import dataclass, field

from src.features.serve_analysis.serve_detector import ServeEvent
from src.features.serve_analysis.double_fault import DoubleFault


@dataclass
class ServeStats:
    """Aggregate serve statistics for a match or video segment."""
    total_serves: int
    first_serves: int
    second_serves: int
    first_serve_in: int
    second_serve_in: int
    first_serve_faults: int
    double_faults: int
    aces: int
    first_serve_pct: float  # first serves in / total first serves
    second_serve_pct: float  # second serves in / total second serves
    # Serve landing positions for heatmap generation (court coords)
    first_serve_positions: list[tuple[float, float]] = field(default_factory=list)
    second_serve_positions: list[tuple[float, float]] = field(default_factory=list)
    all_serve_positions: list[tuple[float, float]] = field(default_factory=list)
    # Per-serve details
    serve_details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "total_serves": self.total_serves,
            "first_serves": self.first_serves,
            "second_serves": self.second_serves,
            "first_serve_in": self.first_serve_in,
            "second_serve_in": self.second_serve_in,
            "first_serve_faults": self.first_serve_faults,
            "double_faults": self.double_faults,
            "aces": self.aces,
            "first_serve_pct": round(self.first_serve_pct, 1),
            "second_serve_pct": round(self.second_serve_pct, 1),
            "first_serve_positions": [
                [round(x, 3), round(y, 3)] for x, y in self.first_serve_positions
            ],
            "second_serve_positions": [
                [round(x, 3), round(y, 3)] for x, y in self.second_serve_positions
            ],
            "all_serve_positions": [
                [round(x, 3), round(y, 3)] for x, y in self.all_serve_positions
            ],
            "serve_details": self.serve_details,
        }


def compute_serve_stats(
    serves: list[ServeEvent],
    double_faults: list[DoubleFault],
) -> ServeStats:
    """Compute aggregate serve statistics.

    Args:
        serves: List of classified serve events.
        double_faults: List of detected double faults.

    Returns:
        ServeStats with all aggregate metrics.
    """
    if not serves:
        return ServeStats(
            total_serves=0,
            first_serves=0,
            second_serves=0,
            first_serve_in=0,
            second_serve_in=0,
            first_serve_faults=0,
            double_faults=0,
            aces=0,
            first_serve_pct=0.0,
            second_serve_pct=0.0,
        )

    first_serves = [s for s in serves if s.serve_number == 1]
    second_serves = [s for s in serves if s.serve_number == 2]

    first_in = [s for s in first_serves if not s.is_fault]
    second_in = [s for s in second_serves if not s.is_fault]
    first_faults = [s for s in first_serves if s.is_fault]
    aces = [s for s in serves if s.is_ace]

    first_pct = (len(first_in) / len(first_serves) * 100.0) if first_serves else 0.0
    second_pct = (len(second_in) / len(second_serves) * 100.0) if second_serves else 0.0

    # Collect landing positions for heatmaps
    first_positions = [
        s.landing_position for s in first_serves
        if s.landing_position is not None
    ]
    second_positions = [
        s.landing_position for s in second_serves
        if s.landing_position is not None
    ]
    all_positions = [
        s.landing_position for s in serves
        if s.landing_position is not None
    ]

    # Per-serve details
    details = [
        {
            "serve_id": s.serve_id,
            "rally_id": s.rally_id,
            "serve_number": s.serve_number,
            "server_end": s.server_end,
            "is_fault": s.is_fault,
            "is_ace": s.is_ace,
            "landing_x": round(s.landing_position[0], 3) if s.landing_position else None,
            "landing_y": round(s.landing_position[1], 3) if s.landing_position else None,
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
        }
        for s in serves
    ]

    return ServeStats(
        total_serves=len(serves),
        first_serves=len(first_serves),
        second_serves=len(second_serves),
        first_serve_in=len(first_in),
        second_serve_in=len(second_in),
        first_serve_faults=len(first_faults),
        double_faults=len(double_faults),
        aces=len(aces),
        first_serve_pct=first_pct,
        second_serve_pct=second_pct,
        first_serve_positions=first_positions,
        second_serve_positions=second_positions,
        all_serve_positions=all_positions,
        serve_details=details,
    )


def format_serve_summary(stats: ServeStats) -> str:
    """Format serve stats as a human-readable summary.

    Args:
        stats: Computed serve statistics.

    Returns:
        Formatted string for console output.
    """
    if stats.total_serves == 0:
        return "No serves detected."

    lines = [
        "=== Serve Statistics ===",
        f"  Total serves:         {stats.total_serves}",
        f"  1st serves:           {stats.first_serves} "
        f"({stats.first_serve_in} in, {stats.first_serve_faults} faults)",
        f"  1st serve %:          {stats.first_serve_pct:.1f}%",
        f"  2nd serves:           {stats.second_serves} "
        f"({stats.second_serve_in} in)",
        f"  2nd serve %:          {stats.second_serve_pct:.1f}%",
        f"  Double faults:        {stats.double_faults}",
        f"  Aces:                 {stats.aces}",
    ]
    return "\n".join(lines)

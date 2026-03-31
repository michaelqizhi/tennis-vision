"""Rally statistics — shot counting and aggregate match stats.

Counts shots per rally and computes summary statistics across
all detected rallies.
"""

from src.features.rally_stats.counter import (
    RallyShots,
    count_shots_pixel,
    count_shots_court,
    count_all_rally_shots,
)
from src.features.rally_stats.stats import (
    RallyStats,
    compute_rally_stats,
    format_stats_summary,
)

__all__ = [
    "RallyShots",
    "count_shots_pixel",
    "count_shots_court",
    "count_all_rally_shots",
    "RallyStats",
    "compute_rally_stats",
    "format_stats_summary",
]

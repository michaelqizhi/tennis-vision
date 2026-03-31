"""Rally statistics — aggregate stats across rallies.

Computes summary statistics from rally detection and shot counting:
rally count, shot distribution, average rally length, longest rally, etc.
"""

from dataclasses import dataclass, field

import numpy as np

from src.features.auto_clip.rally_detector import Rally
from src.features.rally_stats.counter import RallyShots


@dataclass
class RallyStats:
    """Aggregate statistics for all rallies in a match/video."""
    rally_count: int
    total_shots: int
    shots_per_rally: list[int]
    rally_durations: list[float]
    avg_rally_length: float
    avg_shots_per_rally: float
    longest_rally_duration: float
    longest_rally_shots: int
    shortest_rally_duration: float
    shortest_rally_shots: int
    median_rally_duration: float
    median_shots_per_rally: float
    shot_count_distribution: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "rally_count": self.rally_count,
            "total_shots": self.total_shots,
            "shots_per_rally": self.shots_per_rally,
            "rally_durations": [round(d, 2) for d in self.rally_durations],
            "avg_rally_length": round(self.avg_rally_length, 2),
            "avg_shots_per_rally": round(self.avg_shots_per_rally, 2),
            "longest_rally_duration": round(self.longest_rally_duration, 2),
            "longest_rally_shots": self.longest_rally_shots,
            "shortest_rally_duration": round(self.shortest_rally_duration, 2),
            "shortest_rally_shots": self.shortest_rally_shots,
            "median_rally_duration": round(self.median_rally_duration, 2),
            "median_shots_per_rally": round(self.median_shots_per_rally, 2),
            "shot_count_distribution": self.shot_count_distribution,
        }


def compute_rally_stats(
    rallies: list[Rally],
    rally_shots: list[RallyShots],
) -> RallyStats:
    """Compute aggregate statistics from rally and shot data.

    Args:
        rallies: List of detected rallies.
        rally_shots: Shot count data for each rally (must align with rallies).

    Returns:
        RallyStats with all aggregate metrics.
    """
    if len(rallies) != len(rally_shots):
        raise ValueError(
            f"rallies ({len(rallies)}) and rally_shots ({len(rally_shots)}) "
            f"must have the same length"
        )

    if not rallies or not rally_shots:
        return RallyStats(
            rally_count=0,
            total_shots=0,
            shots_per_rally=[],
            rally_durations=[],
            avg_rally_length=0.0,
            avg_shots_per_rally=0.0,
            longest_rally_duration=0.0,
            longest_rally_shots=0,
            shortest_rally_duration=0.0,
            shortest_rally_shots=0,
            median_rally_duration=0.0,
            median_shots_per_rally=0.0,
            shot_count_distribution={},
        )

    durations = [r.duration for r in rallies]
    shots = [rs.shot_count for rs in rally_shots]
    total_shots = sum(shots)

    # Shot count distribution
    distribution: dict[int, int] = {}
    for s in shots:
        distribution[s] = distribution.get(s, 0) + 1

    return RallyStats(
        rally_count=len(rallies),
        total_shots=total_shots,
        shots_per_rally=shots,
        rally_durations=durations,
        avg_rally_length=float(np.mean(durations)),
        avg_shots_per_rally=float(np.mean(shots)),
        longest_rally_duration=float(max(durations)),
        longest_rally_shots=int(max(shots)),
        shortest_rally_duration=float(min(durations)),
        shortest_rally_shots=int(min(shots)),
        median_rally_duration=float(np.median(durations)),
        median_shots_per_rally=float(np.median(shots)),
        shot_count_distribution=dict(sorted(distribution.items())),
    )


def format_stats_summary(stats: RallyStats) -> str:
    """Format rally stats as a human-readable summary string.

    Args:
        stats: Computed rally statistics.

    Returns:
        Formatted string for console output.
    """
    if stats.rally_count == 0:
        return "No rallies detected."

    lines = [
        "=== Rally Statistics ===",
        f"  Rallies detected:     {stats.rally_count}",
        f"  Total shots:          {stats.total_shots}",
        f"  Avg rally length:     {stats.avg_rally_length:.1f}s",
        f"  Avg shots/rally:      {stats.avg_shots_per_rally:.1f}",
        f"  Longest rally:        {stats.longest_rally_duration:.1f}s "
        f"({stats.longest_rally_shots} shots)",
        f"  Shortest rally:       {stats.shortest_rally_duration:.1f}s "
        f"({stats.shortest_rally_shots} shots)",
        f"  Median rally length:  {stats.median_rally_duration:.1f}s",
        f"  Median shots/rally:   {stats.median_shots_per_rally:.1f}",
        "",
        "  Shot count distribution:",
    ]
    for shot_count, freq in stats.shot_count_distribution.items():
        bar = "█" * freq
        lines.append(f"    {shot_count:3d} shots: {bar} ({freq})")

    return "\n".join(lines)

"""Serve analysis — serve detection, fault classification, and statistics."""

from src.features.serve_analysis.serve_detector import (
    ServeEvent,
    ServeDetectorConfig,
    detect_serves,
)
from src.features.serve_analysis.fault_detector import (
    is_in_service_box,
    is_in_target_box,
    classify_serve_fault,
    reclassify_serves,
)
from src.features.serve_analysis.double_fault import (
    DoubleFault,
    detect_double_faults,
)
from src.features.serve_analysis.stats import (
    ServeStats,
    compute_serve_stats,
    format_serve_summary,
)

__all__ = [
    "ServeEvent",
    "ServeDetectorConfig",
    "detect_serves",
    "is_in_service_box",
    "is_in_target_box",
    "classify_serve_fault",
    "reclassify_serves",
    "DoubleFault",
    "detect_double_faults",
    "ServeStats",
    "compute_serve_stats",
    "format_serve_summary",
]
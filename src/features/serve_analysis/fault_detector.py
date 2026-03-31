"""Serve fault detection — classify serves as in or fault.

Determines whether a serve landed inside the correct service box
based on the server's position and the ball's landing coordinates.
"""

from src.features.court_detect.court_template import ITF_DIMS, CourtDimensions
from src.features.serve_analysis.serve_detector import ServeEvent


def is_in_service_box(
    x: float,
    y: float,
    dims: CourtDimensions | None = None,
) -> bool:
    """Check if a point is inside any of the four service boxes.

    Args:
        x: X-coordinate in meters (court coords, origin at center).
        y: Y-coordinate in meters (court coords, origin at center).
        dims: Court dimensions.

    Returns:
        True if the point is inside any service box.
    """
    dims = dims or ITF_DIMS
    sw = dims.singles_width / 2.0
    sl = dims.service_line_distance
    return -sw <= x <= sw and -sl <= y <= sl


def is_in_target_box(
    x: float,
    y: float,
    server_end: str,
    dims: CourtDimensions | None = None,
) -> bool:
    """Check if a point is in the correct target service box for the server.

    The target box is on the opposite side of the net from the server.
    - Server at "near" (y > 0) → target boxes at y in [-sl, 0]
    - Server at "far" (y < 0) → target boxes at y in [0, sl]

    Args:
        x: X-coordinate in meters.
        y: Y-coordinate in meters.
        server_end: "near" or "far".
        dims: Court dimensions.

    Returns:
        True if point is in the correct half's service boxes.

    Raises:
        ValueError: If server_end is not "near" or "far".
    """
    if server_end not in ("near", "far"):
        raise ValueError(f"server_end must be 'near' or 'far', got '{server_end}'")

    dims = dims or ITF_DIMS
    sw = dims.singles_width / 2.0
    sl = dims.service_line_distance

    if not (-sw <= x <= sw):
        return False

    if server_end == "near":
        # Target is far-side service boxes (y in [-sl, 0])
        return -sl <= y <= 0
    else:
        # Target is near-side service boxes (y in [0, sl])
        return 0 <= y <= sl


def classify_serve_fault(
    serve: ServeEvent,
    dims: CourtDimensions | None = None,
) -> bool:
    """Classify whether a serve is a fault based on landing position.

    A serve is a fault if:
    1. No landing position detected (ball lost — likely net fault)
    2. Landing position is outside the target service boxes

    Args:
        serve: The serve event to classify.
        dims: Court dimensions.

    Returns:
        True if the serve is a fault.
    """
    if serve.landing_position is None:
        return True

    x, y = serve.landing_position
    return not is_in_target_box(x, y, serve.server_end, dims)


def reclassify_serves(
    serves: list[ServeEvent],
    dims: CourtDimensions | None = None,
) -> list[ServeEvent]:
    """Re-classify serve faults using court geometry.

    Updates the is_fault field on each serve based on whether the
    landing position is inside the correct target service box.
    Also updates serve_number based on fault sequencing.

    Args:
        serves: List of serve events (modified in place).
        dims: Court dimensions.

    Returns:
        The same list with updated fault classifications.
    """
    for serve in serves:
        serve.is_fault = classify_serve_fault(serve, dims)

    # Re-number serves (1st/2nd) based on fault sequencing
    _renumber_serves(serves)

    return serves


def _renumber_serves(serves: list[ServeEvent]) -> None:
    """Update serve numbers based on fault sequencing.

    A 2nd serve follows a fault. After a non-fault (rally started)
    or after a double fault, the next serve is a 1st serve.
    """
    prev_was_fault = False
    for serve in serves:
        if prev_was_fault:
            serve.serve_number = 2
        else:
            serve.serve_number = 1
        # After a played rally (not a fault), next serve is 1st
        # After a 2nd-serve fault (double fault), next is also 1st
        if not serve.is_fault:
            prev_was_fault = False
        elif serve.serve_number == 2:
            prev_was_fault = False  # double fault → reset
        else:
            prev_was_fault = True

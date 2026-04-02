"""CVAT XML ground truth parser for tennis ball tracking evaluation.

Extracts per-frame ball positions and infers rally boundaries from
annotation presence/absence.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GroundTruthFrame:
    """Ground truth annotation for a single frame."""
    frame_number: int
    x: float
    y: float
    visible: bool  # True = visible, False = occluded


@dataclass
class GroundTruthData:
    """Parsed ground truth: per-frame annotations + rally boundaries."""
    frames: dict[int, GroundTruthFrame]
    rallies: list[tuple[int, int]]  # (start_frame, end_frame) inclusive


def parse_cvat_xml(
    xml_path: str | Path,
    gap_tolerance: int = 5,
    max_frame: int | None = None,
) -> GroundTruthData:
    """Parse CVAT XML annotations for ball tracking ground truth.

    Supports both CVAT "track" format (interpolation mode) and
    per-image "image/box|points" format.

    Rally boundaries are inferred from contiguous annotated frame spans.
    A gap of <= gap_tolerance frames within an annotated span is bridged
    (for brief occlusions mid-rally).

    Args:
        xml_path: Path to CVAT XML annotation file.
        gap_tolerance: Max gap (frames) between annotations within a rally.
        max_frame: If set, discard annotations beyond this frame number.

    Returns:
        GroundTruthData with per-frame annotations and rally boundaries.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    frames: dict[int, GroundTruthFrame] = {}

    # Strategy 1: CVAT track format (video interpolation mode)
    for track in root.iter("track"):
        label = track.get("label", "").lower()
        if label not in ("ball", "tennis_ball", "tennis ball", "tennisball"):
            continue

        for point_or_box in track:
            tag = point_or_box.tag
            frame_num = int(point_or_box.get("frame", "-1"))
            if frame_num < 0:
                continue

            outside = point_or_box.get("outside", "0")
            if outside == "1":
                continue

            occluded = point_or_box.get("occluded", "0")
            visible = occluded != "1"

            # Check for visibility attribute override
            for attr in point_or_box.iter("attribute"):
                attr_name = (attr.get("name") or "").lower()
                if attr_name in ("visibility", "visible"):
                    val = (attr.text or "").strip().lower()
                    if val in ("occluded", "0", "false", "no"):
                        visible = False
                    elif val in ("visible", "1", "true", "yes"):
                        visible = True

            x: float | None = None
            y: float | None = None

            if tag == "points":
                # CVAT points: "x,y" or "x,y;x2,y2;..."
                pts_text = point_or_box.get("points", "")
                if pts_text:
                    first_pt = pts_text.split(";")[0]
                    parts = first_pt.split(",")
                    if len(parts) >= 2:
                        x = float(parts[0])
                        y = float(parts[1])
            elif tag == "box":
                # Bounding box — use center
                xtl = float(point_or_box.get("xtl", "0"))
                ytl = float(point_or_box.get("ytl", "0"))
                xbr = float(point_or_box.get("xbr", "0"))
                ybr = float(point_or_box.get("ybr", "0"))
                x = (xtl + xbr) / 2.0
                y = (ytl + ybr) / 2.0

            if x is not None and y is not None:
                if max_frame is not None and frame_num > max_frame:
                    continue
                frames[frame_num] = GroundTruthFrame(
                    frame_number=frame_num,
                    x=x,
                    y=y,
                    visible=visible,
                )

    # Strategy 2: CVAT per-image format
    for image_elem in root.iter("image"):
        frame_num = int(image_elem.get("id", "-1"))
        if frame_num < 0:
            continue

        for child in image_elem:
            label = (child.get("label") or "").lower()
            if label not in ("ball", "tennis_ball", "tennis ball", "tennisball"):
                continue

            occluded = child.get("occluded", "0")
            visible = occluded != "1"

            for attr in child.iter("attribute"):
                attr_name = (attr.get("name") or "").lower()
                if attr_name in ("visibility", "visible"):
                    val = (attr.text or "").strip().lower()
                    if val in ("occluded", "0", "false", "no"):
                        visible = False
                    elif val in ("visible", "1", "true", "yes"):
                        visible = True

            x: float | None = None
            y: float | None = None
            tag = child.tag

            if tag == "points":
                pts_text = child.get("points", "")
                if pts_text:
                    first_pt = pts_text.split(";")[0]
                    parts = first_pt.split(",")
                    if len(parts) >= 2:
                        x = float(parts[0])
                        y = float(parts[1])
            elif tag == "box":
                xtl = float(child.get("xtl", "0"))
                ytl = float(child.get("ytl", "0"))
                xbr = float(child.get("xbr", "0"))
                ybr = float(child.get("ybr", "0"))
                x = (xtl + xbr) / 2.0
                y = (ytl + ybr) / 2.0

            if x is not None and y is not None:
                if max_frame is not None and frame_num > max_frame:
                    continue
                frames[frame_num] = GroundTruthFrame(
                    frame_number=frame_num,
                    x=x,
                    y=y,
                    visible=visible,
                )

    rallies = _extract_rallies(sorted(frames.keys()), gap_tolerance)
    return GroundTruthData(frames=frames, rallies=rallies)


def _extract_rallies(
    annotated_frames: list[int],
    gap_tolerance: int,
) -> list[tuple[int, int]]:
    """Infer rally boundaries from annotated frame numbers.

    Consecutive annotated frames (with gaps <= gap_tolerance) form a rally.

    Args:
        annotated_frames: Sorted list of frame numbers with annotations.
        gap_tolerance: Max gap to bridge within a rally.

    Returns:
        List of (start_frame, end_frame) inclusive tuples.
    """
    if not annotated_frames:
        return []

    rallies: list[tuple[int, int]] = []
    start = annotated_frames[0]
    prev = annotated_frames[0]

    for f in annotated_frames[1:]:
        if f - prev > gap_tolerance:
            rallies.append((start, prev))
            start = f
        prev = f

    rallies.append((start, prev))
    return rallies

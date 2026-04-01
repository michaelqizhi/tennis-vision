"""Annotation exporters — CVAT XML 1.1 and COCO JSON formats.

Converts consensus results into formats compatible with common annotation
tools and training pipelines.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent

from src.labeling.consensus import DetectionLabel, FrameConsensus

logger = logging.getLogger(__name__)


def export_cvat_xml(
    consensus: list[FrameConsensus],
    output_path: str,
    *,
    video_filename: str = "video.mp4",
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    task_name: str = "tennis_ball_tracking",
) -> str:
    """Export consensus annotations to CVAT XML 1.1 (video annotation format).

    Creates a ``<points>`` track for the ball with one shape per frame that
    has a detection (consensus or uncertain). Frames with no detection are
    skipped (CVAT interpolates or shows gaps).

    Each point carries attributes:
      - ``label``: consensus / uncertain
      - ``agreeing_models``: comma-separated model names

    Args:
        consensus: List of ``FrameConsensus`` objects.
        output_path: Path to write the XML file.
        video_filename: Name of the source video file.
        width: Video frame width in pixels.
        height: Video frame height in pixels.
        fps: Video frame rate.
        task_name: CVAT task name metadata.

    Returns:
        The ``output_path`` string.
    """
    total_frames = len(consensus)

    # Root element
    root = Element("annotations")

    # Version
    SubElement(root, "version").text = "1.1"

    # Meta block
    meta = SubElement(root, "meta")
    task = SubElement(meta, "task")
    SubElement(task, "id").text = "1"
    SubElement(task, "name").text = task_name
    SubElement(task, "size").text = str(total_frames)
    SubElement(task, "mode").text = "interpolation"
    SubElement(task, "overlap").text = "0"
    SubElement(task, "flipped").text = "False"

    created = SubElement(task, "created")
    created.text = datetime.now(timezone.utc).isoformat()
    updated = SubElement(task, "updated")
    updated.text = datetime.now(timezone.utc).isoformat()

    # Source
    source = SubElement(task, "source")
    source.text = video_filename

    # Labels
    labels_el = SubElement(task, "labels")
    label_el = SubElement(labels_el, "label")
    SubElement(label_el, "name").text = "tennis_ball"
    SubElement(label_el, "type").text = "points"

    attrs = SubElement(label_el, "attributes")
    for attr_name, attr_type, attr_default in [
        ("detection_label", "text", "consensus"),
        ("agreeing_models", "text", ""),
        ("confidence", "number", "0.0"),
    ]:
        attr = SubElement(attrs, "attribute")
        SubElement(attr, "name").text = attr_name
        SubElement(attr, "mutable").text = "True"
        SubElement(attr, "input_type").text = attr_type
        SubElement(attr, "default_value").text = attr_default

    # Segments
    segments = SubElement(task, "segments")
    segment = SubElement(segments, "segment")
    SubElement(segment, "id").text = "1"
    SubElement(segment, "start").text = "0"
    SubElement(segment, "stop").text = str(max(0, total_frames - 1))

    # Original size
    orig_size = SubElement(task, "original_size")
    SubElement(orig_size, "width").text = str(width)
    SubElement(orig_size, "height").text = str(height)

    # Track — one track for the ball across all frames
    track = SubElement(root, "track")
    track.set("id", "0")
    track.set("label", "tennis_ball")

    for fc in consensus:
        if fc.label == DetectionLabel.NO_DETECTION or fc.x is None or fc.y is None:
            continue

        shape = SubElement(track, "points")
        shape.set("frame", str(fc.frame_idx))
        shape.set("outside", "0")
        shape.set("occluded", "0")
        shape.set("keyframe", "1")
        shape.set("points", f"{fc.x:.2f},{fc.y:.2f}")

        attr1 = SubElement(shape, "attribute")
        attr1.set("name", "detection_label")
        attr1.text = fc.label.value

        attr2 = SubElement(shape, "attribute")
        attr2.set("name", "agreeing_models")
        attr2.text = ",".join(fc.agreeing_models)

        attr3 = SubElement(shape, "attribute")
        attr3.set("name", "confidence")
        attr3.text = str(fc.confidence) if fc.confidence is not None else "0.0"

    # Write
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = ElementTree(root)
    indent(tree, space="  ")
    tree.write(str(out), encoding="unicode", xml_declaration=True)

    detected_count = sum(
        1 for fc in consensus if fc.label != DetectionLabel.NO_DETECTION
    )
    logger.info("CVAT XML: %d annotated frames written to %s", detected_count, out)
    return output_path


def export_coco_json(
    consensus: list[FrameConsensus],
    output_path: str,
    *,
    video_filename: str = "video.mp4",
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
) -> str:
    """Export consensus annotations to COCO JSON format.

    Each detected frame becomes an image entry; each detection becomes an
    annotation with a point keypoint. This format is compatible with COCO
    keypoint detection training pipelines.

    Args:
        consensus: List of ``FrameConsensus`` objects.
        output_path: Path to write the JSON file.
        video_filename: Source video filename (used in image file_name field).
        width: Video frame width in pixels.
        height: Video frame height in pixels.
        fps: Video frame rate.

    Returns:
        The ``output_path`` string.
    """
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1

    for fc in consensus:
        # Always create an image entry for every frame
        image_entry = {
            "id": fc.frame_idx,
            "file_name": f"{Path(video_filename).stem}_frame_{fc.frame_idx:06d}.jpg",
            "width": width,
            "height": height,
            "frame_index": fc.frame_idx,
        }
        images.append(image_entry)

        if fc.label == DetectionLabel.NO_DETECTION or fc.x is None or fc.y is None:
            continue

        # Ball annotation as a keypoint
        annotation = {
            "id": ann_id,
            "image_id": fc.frame_idx,
            "category_id": 1,
            "keypoints": [fc.x, fc.y, 2],  # x, y, visibility=2 (visible)
            "num_keypoints": 1,
            "bbox": [
                max(0, fc.x - 5),
                max(0, fc.y - 5),
                10,
                10,
            ],
            "area": 100,
            "iscrowd": 0,
            "attributes": {
                "detection_label": fc.label.value,
                "agreeing_models": fc.agreeing_models,
                "confidence": fc.confidence,
            },
        }
        annotations.append(annotation)
        ann_id += 1

    coco = {
        "info": {
            "description": "Tennis ball tracking annotations",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "contributor": "tennis-vision labeling pipeline",
            "date_created": datetime.now(timezone.utc).isoformat(),
            "video_source": video_filename,
            "fps": fps,
        },
        "licenses": [],
        "categories": [
            {
                "id": 1,
                "name": "tennis_ball",
                "supercategory": "sports_equipment",
                "keypoints": ["center"],
                "skeleton": [],
            }
        ],
        "images": images,
        "annotations": annotations,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(
        "COCO JSON: %d images, %d annotations written to %s",
        len(images),
        len(annotations),
        out,
    )
    return output_path

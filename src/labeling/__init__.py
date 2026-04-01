"""Multi-model consensus labeling pipeline for tennis ball detection.

Includes TrackNet V2, TrackNet V4, Florence-2, YOLO-World detectors, and
YOLOv8-nano player detection with player-proximity filtering. Also provides
consensus engine, benchmark reporting, and CVAT/COCO export.

Run the full pipeline from the command line::

    python -m src.labeling.pipeline <video_path> --output <dir>

Disable player filtering with ``--no-player-filter``.
"""

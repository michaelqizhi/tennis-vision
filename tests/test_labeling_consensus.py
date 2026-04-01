"""Tests for the consensus engine and annotation exporters.

All tests use real logic — no mocks. Tests cover:
  - All-agree, partial-agree, no-agree, and edge cases
  - CVAT XML 1.1 export and re-parsing
  - COCO JSON export and validation
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from xml.etree.ElementTree import parse as parse_xml

import pytest

from src.labeling.consensus import (
    Detection,
    DetectionLabel,
    FrameConsensus,
    compute_consensus,
    compute_consensus_stats,
    serialize_consensus,
    _euclidean_distance,
    _find_largest_cluster,
)
from src.labeling.export import export_cvat_xml, export_coco_json


# ─── Helper fixtures ─────────────────────────────────────────────


def _make_raw(frames: list[dict[str, dict | None]]) -> dict[str, dict[str, dict | None]]:
    """Build raw_detections dict from a list of per-frame model dicts.

    Each element maps model_name → {x, y, conf} | None.
    """
    return {str(i): frame for i, frame in enumerate(frames)}


# ─── Euclidean distance ─────────────────────────────────────────


class TestEuclideanDistance:
    def test_zero_distance(self):
        assert _euclidean_distance(10, 20, 10, 20) == 0.0

    def test_known_distance(self):
        assert _euclidean_distance(0, 0, 3, 4) == 5.0

    def test_diagonal(self):
        d = _euclidean_distance(0, 0, 1, 1)
        assert abs(d - math.sqrt(2)) < 1e-10


# ─── Cluster finding ────────────────────────────────────────────


class TestFindLargestCluster:
    def test_empty(self):
        assert _find_largest_cluster([], 15.0) == []

    def test_single(self):
        dets = [Detection("m1", 100, 200, 0.9)]
        cluster = _find_largest_cluster(dets, 15.0)
        assert len(cluster) == 1

    def test_two_close(self):
        dets = [
            Detection("m1", 100, 200, 0.9),
            Detection("m2", 105, 203, 0.8),
        ]
        cluster = _find_largest_cluster(dets, 15.0)
        assert len(cluster) == 2

    def test_two_far(self):
        dets = [
            Detection("m1", 100, 200, 0.9),
            Detection("m2", 500, 500, 0.8),
        ]
        cluster = _find_largest_cluster(dets, 15.0)
        assert len(cluster) == 1

    def test_three_two_agree(self):
        dets = [
            Detection("m1", 100, 200, 0.9),
            Detection("m2", 108, 204, 0.8),
            Detection("m3", 500, 500, 0.7),
        ]
        cluster = _find_largest_cluster(dets, 15.0)
        assert len(cluster) == 2
        names = {d.model_name for d in cluster}
        assert names == {"m1", "m2"}

    def test_exactly_on_threshold(self):
        """Two detections exactly 15px apart should be in the same cluster."""
        dets = [
            Detection("m1", 0, 0, 0.9),
            Detection("m2", 15, 0, 0.8),
        ]
        cluster = _find_largest_cluster(dets, 15.0)
        assert len(cluster) == 2

    def test_just_over_threshold(self):
        """Two detections 15.01px apart should NOT cluster."""
        dets = [
            Detection("m1", 0, 0, 0.9),
            Detection("m2", 15.01, 0, 0.8),
        ]
        cluster = _find_largest_cluster(dets, 15.0)
        assert len(cluster) == 1


# ─── Consensus computation ──────────────────────────────────────


class TestComputeConsensus:
    def test_no_frames(self):
        result = compute_consensus({})
        assert result == []

    def test_single_frame_no_detection(self):
        raw = _make_raw([{"m1": None, "m2": None}])
        result = compute_consensus(raw)
        assert len(result) == 1
        assert result[0].label == DetectionLabel.NO_DETECTION
        assert result[0].x is None

    def test_single_frame_one_model_detects(self):
        raw = _make_raw([{"m1": {"x": 100, "y": 200, "conf": 0.9}, "m2": None}])
        result = compute_consensus(raw)
        assert len(result) == 1
        assert result[0].label == DetectionLabel.UNCERTAIN
        assert result[0].x == 100
        assert result[0].agreeing_models == ["m1"]

    def test_two_models_agree(self):
        raw = _make_raw([{
            "m1": {"x": 100, "y": 200, "conf": 0.9},
            "m2": {"x": 105, "y": 203, "conf": 0.8},
        }])
        result = compute_consensus(raw)
        assert len(result) == 1
        assert result[0].label == DetectionLabel.CONSENSUS
        assert result[0].x == 102.5  # average
        assert result[0].y == 201.5
        assert set(result[0].agreeing_models) == {"m1", "m2"}

    def test_three_models_all_agree(self):
        raw = _make_raw([{
            "m1": {"x": 100, "y": 200, "conf": 0.9},
            "m2": {"x": 103, "y": 202, "conf": 0.8},
            "m3": {"x": 106, "y": 198, "conf": 0.7},
        }])
        result = compute_consensus(raw)
        assert len(result) == 1
        assert result[0].label == DetectionLabel.CONSENSUS
        assert len(result[0].agreeing_models) == 3

    def test_three_models_two_agree_one_disagrees(self):
        raw = _make_raw([{
            "m1": {"x": 100, "y": 200, "conf": 0.9},
            "m2": {"x": 105, "y": 203, "conf": 0.8},
            "m3": {"x": 500, "y": 500, "conf": 0.7},
        }])
        result = compute_consensus(raw)
        assert len(result) == 1
        assert result[0].label == DetectionLabel.CONSENSUS
        assert set(result[0].agreeing_models) == {"m1", "m2"}
        # All detections should be preserved
        assert len(result[0].all_detections) == 3

    def test_two_models_disagree(self):
        raw = _make_raw([{
            "m1": {"x": 100, "y": 200, "conf": 0.9},
            "m2": {"x": 500, "y": 500, "conf": 0.3},
        }])
        result = compute_consensus(raw)
        assert len(result) == 1
        assert result[0].label == DetectionLabel.UNCERTAIN
        # Should use highest confidence
        assert result[0].x == 100
        assert result[0].agreeing_models == ["m1"]

    def test_multiple_frames_mixed(self):
        raw = _make_raw([
            {"m1": {"x": 10, "y": 20, "conf": 0.9}, "m2": {"x": 12, "y": 22, "conf": 0.8}},  # consensus
            {"m1": None, "m2": None},  # no detection
            {"m1": {"x": 50, "y": 60, "conf": 0.7}, "m2": None},  # uncertain
        ])
        result = compute_consensus(raw)
        assert len(result) == 3
        assert result[0].label == DetectionLabel.CONSENSUS
        assert result[1].label == DetectionLabel.NO_DETECTION
        assert result[2].label == DetectionLabel.UNCERTAIN

    def test_custom_threshold(self):
        """With a small threshold, close-but-not-that-close detections should be uncertain."""
        raw = _make_raw([{
            "m1": {"x": 100, "y": 200, "conf": 0.9},
            "m2": {"x": 110, "y": 200, "conf": 0.8},
        }])
        # 10px apart — within default 15px threshold
        result_default = compute_consensus(raw, threshold_px=15.0)
        assert result_default[0].label == DetectionLabel.CONSENSUS

        # But NOT within 5px threshold
        result_strict = compute_consensus(raw, threshold_px=5.0)
        assert result_strict[0].label == DetectionLabel.UNCERTAIN

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            compute_consensus({}, threshold_px=-1.0)

    def test_sorted_by_frame_index(self):
        """Frames should be returned in frame order even if input dict keys are unordered."""
        raw = {"5": {"m1": None}, "1": {"m1": None}, "3": {"m1": None}}
        result = compute_consensus(raw)
        assert [r.frame_idx for r in result] == [1, 3, 5]

    def test_exactly_on_threshold_boundary(self):
        """Two detections at exactly threshold_px distance should be consensus."""
        raw = _make_raw([{
            "m1": {"x": 0, "y": 0, "conf": 0.9},
            "m2": {"x": 15, "y": 0, "conf": 0.8},
        }])
        result = compute_consensus(raw, threshold_px=15.0)
        assert result[0].label == DetectionLabel.CONSENSUS


# ─── Consensus statistics ───────────────────────────────────────


class TestConsensusStats:
    def test_empty(self):
        stats = compute_consensus_stats([])
        assert stats["total_frames"] == 0
        assert stats["consensus_pct"] == 0.0

    def test_all_consensus(self):
        frames = [
            FrameConsensus(frame_idx=i, label=DetectionLabel.CONSENSUS, x=10, y=20)
            for i in range(10)
        ]
        stats = compute_consensus_stats(frames)
        assert stats["total_frames"] == 10
        assert stats["consensus_count"] == 10
        assert stats["consensus_pct"] == 100.0

    def test_mixed(self):
        frames = [
            FrameConsensus(frame_idx=0, label=DetectionLabel.CONSENSUS, x=10, y=20),
            FrameConsensus(frame_idx=1, label=DetectionLabel.UNCERTAIN, x=10, y=20),
            FrameConsensus(frame_idx=2, label=DetectionLabel.NO_DETECTION),
            FrameConsensus(frame_idx=3, label=DetectionLabel.NO_DETECTION),
        ]
        stats = compute_consensus_stats(frames)
        assert stats["total_frames"] == 4
        assert stats["consensus_count"] == 1
        assert stats["uncertain_count"] == 1
        assert stats["no_detection_count"] == 2
        assert stats["consensus_pct"] == 25.0
        assert stats["uncertain_pct"] == 25.0
        assert stats["no_detection_pct"] == 50.0


# ─── Serialization ──────────────────────────────────────────────


class TestSerializeConsensus:
    def test_serialize_roundtrip(self):
        raw = _make_raw([{
            "m1": {"x": 100, "y": 200, "conf": 0.9},
            "m2": {"x": 105, "y": 203, "conf": 0.8},
        }])
        consensus = compute_consensus(raw)
        serialized = serialize_consensus(consensus)

        assert "0" in serialized
        entry = serialized["0"]
        assert entry["label"] == "consensus"
        assert entry["x"] == 102.5
        assert len(entry["agreeing_models"]) == 2
        assert len(entry["all_detections"]) == 2

    def test_no_detection_serialized(self):
        raw = _make_raw([{"m1": None}])
        consensus = compute_consensus(raw)
        serialized = serialize_consensus(consensus)
        assert serialized["0"]["label"] == "no_detection"
        assert serialized["0"]["x"] is None

    def test_serialized_is_json_compatible(self):
        raw = _make_raw([
            {"m1": {"x": 1, "y": 2, "conf": 0.5}, "m2": {"x": 3, "y": 4, "conf": 0.6}},
            {"m1": None, "m2": None},
        ])
        consensus = compute_consensus(raw)
        serialized = serialize_consensus(consensus)
        # Must not raise
        json_str = json.dumps(serialized)
        reloaded = json.loads(json_str)
        assert reloaded["0"]["label"] in ("consensus", "uncertain")


# ─── CVAT XML Export ─────────────────────────────────────────────


class TestCvatXmlExport:
    def test_basic_export(self, tmp_path):
        consensus = [
            FrameConsensus(0, DetectionLabel.CONSENSUS, 100.5, 200.3, 0.85,
                           ["m1", "m2"], []),
            FrameConsensus(1, DetectionLabel.NO_DETECTION),
            FrameConsensus(2, DetectionLabel.UNCERTAIN, 50.0, 75.0, 0.6,
                           ["m1"], []),
        ]
        out = str(tmp_path / "test.xml")
        result = export_cvat_xml(consensus, out, video_filename="test.mp4",
                                 width=640, height=480, fps=30.0)
        assert result == out
        assert os.path.exists(out)

        # Parse and validate structure
        tree = parse_xml(out)
        root = tree.getroot()
        assert root.tag == "annotations"
        assert root.find("version").text == "1.1"

        # Meta block
        meta = root.find("meta")
        assert meta is not None
        task = meta.find("task")
        assert task.find("size").text == "3"
        assert task.find("mode").text == "interpolation"
        assert task.find("source").text == "test.mp4"

        # Original size
        orig = task.find("original_size")
        assert orig.find("width").text == "640"
        assert orig.find("height").text == "480"

        # Track
        track = root.find("track")
        assert track is not None
        assert track.get("label") == "tennis_ball"

        # Points — only frames 0 and 2 should have points (frame 1 is no_detection)
        points = track.findall("points")
        assert len(points) == 2
        assert points[0].get("frame") == "0"
        assert points[1].get("frame") == "2"

        # Check attributes on first point
        attrs = {a.get("name"): a.text for a in points[0].findall("attribute")}
        assert attrs["detection_label"] == "consensus"
        assert "m1" in attrs["agreeing_models"]

    def test_empty_consensus(self, tmp_path):
        out = str(tmp_path / "empty.xml")
        export_cvat_xml([], out)
        tree = parse_xml(out)
        root = tree.getroot()
        track = root.find("track")
        assert track is not None
        assert len(track.findall("points")) == 0

    def test_all_no_detection(self, tmp_path):
        consensus = [
            FrameConsensus(i, DetectionLabel.NO_DETECTION) for i in range(5)
        ]
        out = str(tmp_path / "nodet.xml")
        export_cvat_xml(consensus, out)
        tree = parse_xml(out)
        track = tree.getroot().find("track")
        assert len(track.findall("points")) == 0

    def test_cvat_xml_loads_cleanly(self, tmp_path):
        """The XML must be well-formed and parseable."""
        consensus = [
            FrameConsensus(i, DetectionLabel.CONSENSUS, float(i * 10), float(i * 20), 0.9,
                           ["m1", "m2"], [])
            for i in range(100)
        ]
        out = str(tmp_path / "large.xml")
        export_cvat_xml(consensus, out, width=1920, height=1080)
        tree = parse_xml(out)  # Must not raise
        points = tree.getroot().find("track").findall("points")
        assert len(points) == 100


# ─── COCO JSON Export ────────────────────────────────────────────


class TestCocoJsonExport:
    def test_basic_export(self, tmp_path):
        consensus = [
            FrameConsensus(0, DetectionLabel.CONSENSUS, 100.5, 200.3, 0.85,
                           ["m1", "m2"], []),
            FrameConsensus(1, DetectionLabel.NO_DETECTION),
            FrameConsensus(2, DetectionLabel.UNCERTAIN, 50.0, 75.0, 0.6,
                           ["m1"], []),
        ]
        out = str(tmp_path / "test.json")
        result = export_coco_json(consensus, out, video_filename="test.mp4",
                                  width=640, height=480, fps=30.0)
        assert result == out
        assert os.path.exists(out)

        with open(out) as f:
            coco = json.load(f)

        # Info
        assert "tennis" in coco["info"]["description"].lower()
        assert coco["info"]["fps"] == 30.0

        # Categories
        assert len(coco["categories"]) == 1
        assert coco["categories"][0]["name"] == "tennis_ball"

        # Images — one per frame (all frames, even no_detection)
        assert len(coco["images"]) == 3
        assert coco["images"][0]["width"] == 640
        assert coco["images"][0]["height"] == 480

        # Annotations — only for detected frames (0 and 2)
        assert len(coco["annotations"]) == 2
        ann0 = coco["annotations"][0]
        assert ann0["image_id"] == 0
        assert ann0["category_id"] == 1
        assert ann0["keypoints"] == [100.5, 200.3, 2]
        assert ann0["attributes"]["detection_label"] == "consensus"

    def test_empty_consensus(self, tmp_path):
        out = str(tmp_path / "empty.json")
        export_coco_json([], out)
        with open(out) as f:
            coco = json.load(f)
        assert len(coco["images"]) == 0
        assert len(coco["annotations"]) == 0

    def test_annotation_ids_unique(self, tmp_path):
        consensus = [
            FrameConsensus(i, DetectionLabel.CONSENSUS, float(i), float(i), 0.9,
                           ["m1", "m2"], [])
            for i in range(50)
        ]
        out = str(tmp_path / "ids.json")
        export_coco_json(consensus, out)
        with open(out) as f:
            coco = json.load(f)
        ids = [a["id"] for a in coco["annotations"]]
        assert len(ids) == len(set(ids)), "Annotation IDs must be unique"

    def test_bbox_non_negative(self, tmp_path):
        """Bounding box coordinates should never be negative."""
        consensus = [
            FrameConsensus(0, DetectionLabel.CONSENSUS, 2.0, 3.0, 0.9,
                           ["m1", "m2"], []),
        ]
        out = str(tmp_path / "bbox.json")
        export_coco_json(consensus, out)
        with open(out) as f:
            coco = json.load(f)
        bbox = coco["annotations"][0]["bbox"]
        assert all(v >= 0 for v in bbox)

    def test_coco_json_valid_json(self, tmp_path):
        """Output must be valid JSON that can be re-loaded."""
        consensus = [
            FrameConsensus(i, DetectionLabel.CONSENSUS, float(i * 10), float(i * 20), 0.9,
                           ["m1"], [])
            for i in range(100)
        ]
        out = str(tmp_path / "valid.json")
        export_coco_json(consensus, out)
        with open(out) as f:
            data = json.load(f)  # Must not raise
        assert len(data["annotations"]) == 100


# ─── Integration: consensus → export ─────────────────────────────


class TestConsensusToExport:
    """End-to-end: raw detections → consensus → export."""

    def test_full_pipeline(self, tmp_path):
        raw = _make_raw([
            {
                "tracknet_v2": {"x": 100, "y": 200, "conf": 0.9},
                "florence_2": {"x": 105, "y": 203, "conf": 0.5},
                "yolo_world": None,
            },
            {
                "tracknet_v2": None,
                "florence_2": None,
                "yolo_world": None,
            },
            {
                "tracknet_v2": {"x": 300, "y": 400, "conf": 0.8},
                "florence_2": None,
                "yolo_world": {"x": 305, "y": 398, "conf": 0.7},
            },
            {
                "tracknet_v2": {"x": 50, "y": 60, "conf": 0.6},
                "florence_2": {"x": 500, "y": 600, "conf": 0.5},
                "yolo_world": None,
            },
        ])

        consensus = compute_consensus(raw, threshold_px=15.0)
        assert len(consensus) == 4
        assert consensus[0].label == DetectionLabel.CONSENSUS  # tracknet+florence agree
        assert consensus[1].label == DetectionLabel.NO_DETECTION
        assert consensus[2].label == DetectionLabel.CONSENSUS  # tracknet+yolo agree
        assert consensus[3].label == DetectionLabel.UNCERTAIN  # tracknet+florence disagree

        stats = compute_consensus_stats(consensus)
        assert stats["total_frames"] == 4
        assert stats["consensus_count"] == 2
        assert stats["consensus_pct"] == 50.0

        # Export both formats
        cvat_path = str(tmp_path / "annotations_cvat.xml")
        coco_path = str(tmp_path / "annotations_coco.json")

        export_cvat_xml(consensus, cvat_path, width=640, height=480)
        export_coco_json(consensus, coco_path, width=640, height=480)

        # Verify CVAT
        tree = parse_xml(cvat_path)
        points = tree.getroot().find("track").findall("points")
        assert len(points) == 3  # frames 0, 2, 3 (frame 1 = no detection)

        # Verify COCO
        with open(coco_path) as f:
            coco = json.load(f)
        assert len(coco["images"]) == 4
        assert len(coco["annotations"]) == 3

"""Download pretrained model weights for Tennis Vision.

Downloads:
  - TrackNetV2 weights from Google Drive
  - TennisCourtDetector weights from Google Drive (for Sprint 2)

Weights are saved to the weights/ directory.
"""

import os
import sys
from pathlib import Path


def download_tracknet_weights(output_dir: Path) -> Path:
    """Download TrackNetV2 pretrained weights.

    Args:
        output_dir: Directory to save weights to.

    Returns:
        Path to the downloaded weights file.
    """
    import gdown

    output_path = output_dir / "tracknet.pt"
    if output_path.exists():
        print(f"TrackNet weights already exist at {output_path}")
        return output_path

    # Google Drive file ID from the yastrebksv/TrackNet repo
    file_id = "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Downloading TrackNet weights to {output_path}...")
    os.makedirs(output_dir, exist_ok=True)
    gdown.download(url, str(output_path), quiet=False)

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Downloaded TrackNet weights: {size_mb:.1f} MB")
    else:
        print("ERROR: Download failed. Check your internet connection.")
        print("You can also manually download from:")
        print(f"  https://drive.google.com/file/d/{file_id}/view")
        print(f"  and save to: {output_path}")
        sys.exit(1)

    return output_path


def download_court_detector_weights(output_dir: Path) -> Path:
    """Download TennisCourtDetector pretrained weights.

    Args:
        output_dir: Directory to save weights to.

    Returns:
        Path to the downloaded weights file.
    """
    import gdown

    output_path = output_dir / "court_detector.pt"
    if output_path.exists():
        print(f"Court detector weights already exist at {output_path}")
        return output_path

    # Google Drive file ID from yastrebksv/TennisCourtDetector README
    file_id = "1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Downloading court detector weights to {output_path}...")
    os.makedirs(output_dir, exist_ok=True)
    gdown.download(url, str(output_path), quiet=False)

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Downloaded court detector weights: {size_mb:.1f} MB")
    else:
        print("ERROR: Download failed. Check your internet connection.")
        print("You can also manually download from:")
        print(f"  https://drive.google.com/file/d/{file_id}/view")
        print(f"  and save to: {output_path}")
        sys.exit(1)

    return output_path


def main() -> None:
    """Download all model weights."""
    weights_dir = Path(__file__).resolve().parent.parent / "weights"
    os.makedirs(weights_dir, exist_ok=True)

    print("=" * 60)
    print("Tennis Vision — Model Weight Downloader")
    print("=" * 60)

    # TrackNet
    print("\n[1/2] TrackNet V2 (Ball Tracking)")
    download_tracknet_weights(weights_dir)

    # Court detector (placeholder for Sprint 2)
    print("\n[2/2] TennisCourtDetector (Court Detection)")
    download_court_detector_weights(weights_dir)

    print("\n" + "=" * 60)
    print("Done! Weights saved to:", weights_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()

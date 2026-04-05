"""Download tennis match videos from YouTube for net detection dataset.

Reads URLs from data/net_detection/video_urls.txt, downloads each at
720p using yt-dlp, and saves to data/net_detection/videos/.
Skips already-downloaded videos.

Usage:
    python scripts/net_dataset/download_videos.py
    python scripts/net_dataset/download_videos.py --max 5  # download first 5 only
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
URLS_FILE = PROJECT_ROOT / "data" / "net_detection" / "video_urls.txt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "net_detection" / "videos"


def parse_urls(path: Path) -> list[str]:
    """Read non-empty, non-comment lines from the URL file."""
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()  # strip inline comments
        if line and line.startswith("http"):
            urls.append(line)
    return urls


def video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else url


def already_downloaded(vid_id: str) -> bool:
    """Check if a video with this ID is already on disk."""
    return any(OUTPUT_DIR.glob(f"*{vid_id}*"))


def download(url: str, idx: int, total: int) -> bool:
    """Download a single video. Returns True on success."""
    vid = video_id(url)
    if already_downloaded(vid):
        print(f"  [{idx}/{total}] SKIP (exists): {vid}")
        return True

    print(f"  [{idx}/{total}] Downloading: {url}")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(OUTPUT_DIR / f"%(id)s.%(ext)s"),
        "--no-playlist",
        "--quiet",
        "--progress",
        "--js-runtimes", "node",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()[:200]}")
        return False
    print(f"    OK: {vid}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download tennis videos for net detection dataset")
    parser.add_argument("--max", type=int, default=None, help="Max videos to download")
    args = parser.parse_args()

    if not URLS_FILE.exists():
        print(f"ERROR: URL file not found: {URLS_FILE}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = parse_urls(URLS_FILE)

    if args.max:
        urls = urls[:args.max]

    print(f"Downloading {len(urls)} videos to {OUTPUT_DIR}\n")

    ok = 0
    for i, url in enumerate(urls, 1):
        if download(url, i, len(urls)):
            ok += 1

    print(f"\nDone: {ok}/{len(urls)} downloaded successfully")
    if ok < len(urls):
        print("Some downloads failed — check URLs and re-run (existing downloads will be skipped)")


if __name__ == "__main__":
    main()

"""Diagnostic script: run Florence-2 on a single frame with multiple task types.

Prints ALL detected objects (bboxes + labels) for each task to reveal what
Florence-2 actually sees, and why <OD> returns zero "ball" matches.

Usage:
    python scripts/diagnose_florence.py
"""

import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

VIDEO_PATH = "tests/fixtures/tennis_test.mp4"
MODEL_ID = "microsoft/Florence-2-large"
DEVICE = "cuda"
FRAME_INDEX = 100  # pick a frame likely to contain a ball in play


def extract_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def run_task(model, processor, pil_img, task: str, text_input: str | None = None):
    """Run a Florence-2 task and return the post-processed result dict."""
    prompt = task if text_input is None else task + text_input
    inputs = processor(text=prompt, images=pil_img, return_tensors="pt")
    inputs = {
        k: v.to(DEVICE, dtype=torch.float16) if v.dtype == torch.float32 else v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        generated_text, task=task, image_size=(pil_img.width, pil_img.height),
    )
    return result, generated_text


def main():
    print("=" * 70)
    print("Florence-2 Diagnostic — Tennis Ball Detection")
    print("=" * 70)

    # Extract frame
    frame_bgr = extract_frame(VIDEO_PATH, FRAME_INDEX)
    print(f"\nFrame shape: {frame_bgr.shape} (H, W, C)")
    frame_rgb = frame_bgr[:, :, ::-1]
    pil_img = Image.fromarray(frame_rgb)
    print(f"PIL image size: {pil_img.size} (W, H), mode: {pil_img.mode}")

    # Load model
    print(f"\nLoading {MODEL_ID} on {DEVICE} (float16) ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    print("Model loaded.\n")

    # ── Task 1: <OD> (generic object detection) ──────────────────────
    print("=" * 70)
    print("TASK 1: <OD> — Generic Object Detection")
    print("=" * 70)
    result, raw = run_task(model, processor, pil_img, "<OD>")
    od = result.get("<OD>", {})
    bboxes = od.get("bboxes", [])
    labels = od.get("labels", [])
    print(f"  Total detections: {len(labels)}")
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        marker = " *** BALL MATCH ***" if "ball" in label.lower() else ""
        print(f"  [{i:3d}] label={label!r:30s}  bbox={bbox}{marker}")
    if not labels:
        print("  (no detections)")
    print(f"\n  Raw output:\n  {raw[:500]}\n")

    # ── Task 2: <CAPTION_TO_PHRASE_GROUNDING> ────────────────────────
    print("=" * 70)
    print("TASK 2: <CAPTION_TO_PHRASE_GROUNDING> — 'tennis ball'")
    print("=" * 70)
    result2, raw2 = run_task(
        model, processor, pil_img,
        "<CAPTION_TO_PHRASE_GROUNDING>", "tennis ball",
    )
    pg = result2.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
    bboxes2 = pg.get("bboxes", [])
    labels2 = pg.get("labels", [])
    print(f"  Total detections: {len(labels2)}")
    for i, (bbox, label) in enumerate(zip(bboxes2, labels2)):
        print(f"  [{i:3d}] label={label!r:30s}  bbox={bbox}")
    if not labels2:
        print("  (no detections)")
    print(f"\n  Raw output:\n  {raw2[:500]}\n")

    # ── Task 3: <OPEN_VOCABULARY_DETECTION> ──────────────────────────
    print("=" * 70)
    print("TASK 3: <OPEN_VOCABULARY_DETECTION> — 'tennis ball'")
    print("=" * 70)
    result3, raw3 = run_task(
        model, processor, pil_img,
        "<OPEN_VOCABULARY_DETECTION>", "tennis ball",
    )
    ovd = result3.get("<OPEN_VOCABULARY_DETECTION>", {})
    bboxes3 = ovd.get("bboxes", [])
    labels3 = ovd.get("bboxes_labels", [])
    print(f"  Total detections: {len(labels3)}")
    for i, (bbox, label) in enumerate(zip(bboxes3, labels3)):
        print(f"  [{i:3d}] label={label!r:30s}  bbox={bbox}")
    if not labels3:
        print("  (no detections)")
    print(f"\n  Raw output:\n  {raw3[:500]}\n")

    # ── Task 4: <DETAILED_CAPTION> to see what it sees ───────────────
    print("=" * 70)
    print("TASK 4: <DETAILED_CAPTION> — what does Florence-2 see?")
    print("=" * 70)
    result4, raw4 = run_task(model, processor, pil_img, "<DETAILED_CAPTION>")
    caption = result4.get("<DETAILED_CAPTION>", "")
    print(f"  Caption: {caption}")
    print(f"\n  Raw output:\n  {raw4[:500]}\n")

    # ── Task 5: <CAPTION_TO_PHRASE_GROUNDING> with richer prompt ─────
    print("=" * 70)
    print("TASK 5: <CAPTION_TO_PHRASE_GROUNDING> — 'a small green tennis ball on the court'")
    print("=" * 70)
    result5, raw5 = run_task(
        model, processor, pil_img,
        "<CAPTION_TO_PHRASE_GROUNDING>",
        "a small green tennis ball on the court",
    )
    pg5 = result5.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
    bboxes5 = pg5.get("bboxes", [])
    labels5 = pg5.get("labels", [])
    print(f"  Total detections: {len(labels5)}")
    for i, (bbox, label) in enumerate(zip(bboxes5, labels5)):
        print(f"  [{i:3d}] label={label!r:30s}  bbox={bbox}")
    if not labels5:
        print("  (no detections)")
    print(f"\n  Raw output:\n  {raw5[:500]}\n")

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  <OD> detections:                        {len(bboxes)}")
    print(f"  <OD> with 'ball' in label:              {sum(1 for l in labels if 'ball' in l.lower())}")
    print(f"  <CAPTION_TO_PHRASE_GROUNDING> 'tennis ball': {len(bboxes2)}")
    print(f"  <OPEN_VOCABULARY_DETECTION> 'tennis ball':   {len(bboxes3)}")
    print(f"  <CAPTION_TO_PHRASE_GROUNDING> rich prompt:   {len(bboxes5)}")


if __name__ == "__main__":
    main()

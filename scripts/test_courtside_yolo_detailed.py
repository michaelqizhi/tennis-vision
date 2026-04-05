"""
Detailed test of CourtSide YOLO model with multiple confidence thresholds.
"""
import cv2
import time
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import matplotlib.pyplot as plt

def test_single_frame(model, frame, frame_idx, conf_threshold=0.05):
    """Test a single frame with given confidence threshold."""
    results = model.predict(frame, conf=conf_threshold, verbose=False)
    
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            cls_name = model.names[cls]
            
            detections.append({
                'class': cls_name,
                'conf': conf,
                'bbox': xyxy
            })
    
    return results, detections

def main():
    print("=" * 80)
    print("CourtSide YOLO - Detailed Analysis")
    print("=" * 80)
    
    # Setup
    video_path = Path('tests/fixtures/real_match_10min.mp4')
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print("\n🔄 Loading model...")
    weights_path = hf_hub_download(
        repo_id="Davidsv/CourtSide-Computer-Vision-v1",
        filename="model.pt"
    )
    model = YOLO(weights_path)
    print("✅ Model loaded!")
    print(f"\n📊 Classes: {list(model.names.values())}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Test multiple frames with very low confidence
    test_frames = [1000, 1500, 2000, 3000, 4500, 6000, 7500, 9000]
    conf_thresholds = [0.01, 0.05, 0.15, 0.25]
    
    print(f"\n🔍 Testing {len(test_frames)} frames with confidence thresholds: {conf_thresholds}")
    
    best_frame = None
    best_detections = []
    best_conf = None
    
    for conf in conf_thresholds:
        print(f"\n{'=' * 80}")
        print(f"Testing with confidence threshold: {conf}")
        print(f"{'=' * 80}")
        
        total_detections = 0
        
        for frame_idx in test_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            results, detections = test_single_frame(model, frame, frame_idx, conf)
            
            if len(detections) > 0:
                print(f"\n📍 Frame {frame_idx} ({frame_idx/fps:.1f}s): {len(detections)} detections")
                for det in detections:
                    print(f"   • {det['class']}: {det['conf']:.3f}")
                
                # Save best frame
                if len(detections) > len(best_detections):
                    best_frame = frame
                    best_detections = detections
                    best_conf = conf
                    best_results = results
                    best_frame_idx = frame_idx
                
                total_detections += len(detections)
        
        print(f"\n   Total detections at conf={conf}: {total_detections}")
    
    cap.release()
    
    # Save annotated result if we found anything
    if best_frame is not None:
        print(f"\n{'=' * 80}")
        print(f"📸 Best Frame: {best_frame_idx} with {len(best_detections)} detections (conf={best_conf})")
        print(f"{'=' * 80}")
        
        for det in best_detections:
            print(f"   • {det['class']}: {det['conf']:.3f}")
            x1, y1, x2, y2 = det['bbox']
            print(f"     BBox: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")
        
        # Save annotated image
        annotated = best_results[0].plot()
        output_path = output_dir / 'courtside_yolo_best.jpg'
        cv2.imwrite(str(output_path), annotated)
        print(f"\n💾 Saved: {output_path}")
        
        # Also save original frame for comparison
        orig_path = output_dir / 'courtside_original_frame.jpg'
        cv2.imwrite(str(orig_path), best_frame)
        print(f"💾 Saved: {orig_path}")
    else:
        print(f"\n❌ No detections found at any confidence level!")
        print(f"\n🤔 Possible reasons:")
        print(f"   1. Model trained on different camera angles (likely broadcast/overhead)")
        print(f"   2. Courtside view is too different from training data")
        print(f"   3. Model expects specific court visibility/framing")
        print(f"\n💡 Let's save a sample frame to inspect:")
        
        # Save a frame anyway to see what the model is working with
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 3000)
        ret, frame = cap.read()
        if ret:
            sample_path = output_dir / 'courtside_sample_frame.jpg'
            cv2.imwrite(str(sample_path), frame)
            print(f"   💾 Saved sample frame: {sample_path}")
        cap.release()
    
    print(f"\n{'=' * 80}")
    print("🎯 FINAL ASSESSMENT")
    print(f"{'=' * 80}")
    
    if len(best_detections) == 0:
        print(f"\n❌ CourtSide YOLOv11 is NOT suitable for courtside amateur footage")
        print(f"\nReason:")
        print(f"   • Model trained on broadcast/overhead camera angles")
        print(f"   • Courtside perspective is fundamentally different")
        print(f"   • Model expects full court visibility from above")
        print(f"\n💡 Recommendations:")
        print(f"   1. Use court line detection (Hough/CNN) instead")
        print(f"   2. Train a custom model on courtside footage")
        print(f"   3. Look for models specifically trained on player/ground-level views")
    else:
        print(f"\n✅ Found {len(best_detections)} detections!")
        print(f"\n🎯 For court geometry extraction:")
        
        # Analyze detection types
        court_zones = [d for d in best_detections if 'court' in d['class'] or 'zone' in d['class'] or 'box' in d['class'] or 'alley' in d['class']]
        
        if len(court_zones) >= 2:
            print(f"   ✅ Detected {len(court_zones)} court zones")
            print(f"   → Can extract corners from zone boundaries")
            print(f"   → Derive homography: zone corners → court template")
        else:
            print(f"   ⚠️  Only {len(court_zones)} court zones detected")
            print(f"   → May not be sufficient for homography")
        
        print(f"\n📊 Detected classes:")
        class_counts = {}
        for det in best_detections:
            cls = det['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        for cls, count in sorted(class_counts.items()):
            print(f"   • {cls}: {count}")
    
    print(f"\n{'=' * 80}\n")

if __name__ == "__main__":
    main()

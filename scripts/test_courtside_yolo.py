"""
Test script for CourtSide YOLOv11 model on amateur courtside footage.
Tests whether the model can detect court zones and derive court keypoints.
"""
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

def main():
    print("=" * 80)
    print("CourtSide YOLOv11 Model Test")
    print("=" * 80)
    
    # Paths
    video_path = Path('tests/fixtures/real_match_10min.mp4')
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Check video exists
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"\n📹 Video: {video_path}")
    
    # Load model
    print("\n🔄 Loading CourtSide YOLOv11 model from HuggingFace...")
    print("   Model: Davidsv/CourtSide-Computer-Vision-v1")
    
    try:
        # Try direct loading first
        model = YOLO('Davidsv/CourtSide-Computer-Vision-v1')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️  Direct loading failed: {e}")
        print("\n🔄 Downloading model weights...")
        
        # Download explicitly from HuggingFace
        try:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(
                repo_id="Davidsv/CourtSide-Computer-Vision-v1",
                filename="model.pt"
            )
            print(f"   Downloaded to: {weights_path}")
            model = YOLO(weights_path)
            print(f"✅ Model loaded successfully!")
        except Exception as e2:
            print(f"❌ Failed to download model: {e2}")
            print("\n💡 Please check the HuggingFace model page:")
            print("   https://huggingface.co/Davidsv/CourtSide-Computer-Vision-v1")
            return
    
    # Print model info
    print(f"\n📊 Model Info:")
    print(f"   Task: {model.task}")
    print(f"   Classes: {model.names}")
    print(f"   Number of classes: {len(model.names)}")
    
    # Open video
    print(f"\n🎬 Opening video...")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Failed to open video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {frame_count}")
    print(f"   Duration: {frame_count/fps:.1f} seconds")
    
    # Test frames
    test_frames = [1500, 3000, 4500]
    print(f"\n🔍 Testing on frames: {test_frames}")
    
    all_detections = []
    inference_times = []
    
    for frame_idx in test_frames:
        print(f"\n{'─' * 80}")
        print(f"Frame {frame_idx} (at {frame_idx/fps:.1f}s)")
        print(f"{'─' * 80}")
        
        # Extract frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ Failed to read frame {frame_idx}")
            continue
        
        # Run inference
        start_time = time.time()
        results = model.predict(frame, conf=0.25, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        inference_times.append(inference_time)
        
        print(f"⏱️  Inference time: {inference_time:.1f}ms")
        
        # Analyze results
        for r in results:
            num_boxes = len(r.boxes)
            print(f"📦 Detected {num_boxes} objects")
            
            if num_boxes == 0:
                print("   (No detections)")
                continue
            
            for i, box in enumerate(r.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                cls_name = model.names[cls]
                
                x1, y1, x2, y2 = [round(x) for x in xyxy]
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                print(f"   [{i+1}] {cls_name}")
                print(f"       Confidence: {conf:.3f}")
                print(f"       BBox: ({x1}, {y1}) -> ({x2}, {y2})")
                print(f"       Size: {bbox_width}x{bbox_height}")
                print(f"       Center: ({center_x}, {center_y})")
                
                all_detections.append({
                    'frame': frame_idx,
                    'class': cls_name,
                    'conf': conf,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y)
                })
        
        # Save annotated frame (save the middle one as main output)
        if frame_idx == 3000:
            annotated = results[0].plot()
            output_path = output_dir / 'courtside_yolo_test.jpg'
            cv2.imwrite(str(output_path), annotated)
            print(f"💾 Saved annotated frame: {output_path}")
    
    cap.release()
    
    # Summary
    print(f"\n{'=' * 80}")
    print("📊 SUMMARY")
    print(f"{'=' * 80}")
    
    print(f"\n⏱️  Performance:")
    print(f"   Average inference time: {sum(inference_times)/len(inference_times):.1f}ms")
    print(f"   Min: {min(inference_times):.1f}ms, Max: {max(inference_times):.1f}ms")
    print(f"   Estimated real-time FPS: {1000/sum(inference_times)*len(inference_times):.1f}")
    
    print(f"\n🎯 Detections:")
    print(f"   Total detections: {len(all_detections)}")
    
    # Group by class
    class_counts = {}
    class_confidences = {}
    for det in all_detections:
        cls = det['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
        if cls not in class_confidences:
            class_confidences[cls] = []
        class_confidences[cls].append(det['conf'])
    
    print(f"\n   Detections by class:")
    for cls, count in sorted(class_counts.items()):
        avg_conf = sum(class_confidences[cls]) / len(class_confidences[cls])
        print(f"      {cls}: {count} detections (avg conf: {avg_conf:.3f})")
    
    # Feasibility assessment
    print(f"\n{'=' * 80}")
    print("🤔 FEASIBILITY ASSESSMENT")
    print(f"{'=' * 80}")
    
    if len(all_detections) == 0:
        print("\n❌ No detections found on any test frame!")
        print("   The model may not be suitable for courtside amateur footage.")
    else:
        print(f"\n✅ Model detected {len(class_counts)} different classes")
        print(f"\n📍 Detected classes:")
        for cls in sorted(class_counts.keys()):
            print(f"   • {cls}")
        
        print(f"\n💡 For court geometry derivation:")
        
        # Check if we can derive keypoints
        court_related = [cls for cls in class_counts.keys() 
                        if any(kw in cls.lower() for kw in 
                              ['court', 'line', 'corner', 'baseline', 'service', 'net', 'sideline'])]
        
        if len(court_related) >= 2:
            print(f"   ✅ Found {len(court_related)} court-related classes")
            print(f"   → May be able to derive court keypoints from zone boundaries")
        else:
            print(f"   ⚠️  Limited court-related detections")
            print(f"   → May need additional processing to extract court corners")
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Analyze zone boundaries to extract corner points")
        print(f"   2. Map detected zones to known court dimensions")
        print(f"   3. Compute homography from zone corners → court template")
        print(f"   4. Test on full video sequence for temporal consistency")
    
    print(f"\n{'=' * 80}")
    print("✅ Test complete!")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    main()

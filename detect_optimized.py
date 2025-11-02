import cv2
import torch
import numpy as np
from pathlib import Path
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image
import os
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings('ignore')

class OWLv2ObjectDetectorOptimized:
    def __init__(self, model_name="google/owlv2-base-patch16-ensemble", device=None,
                 use_fast=True, low_memory=True):
        """
        Initialize OWLv2 model with optimizations for Mac

        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'mps', or 'cpu' (auto-detected if None)
            use_fast: Use fast tokenizer if available
            low_memory: Use 8-bit quantization to reduce memory
        """
        # Auto-detect best device
        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.low_memory = low_memory

        print(f"Loading OWLv2 model: {model_name}")
        print(f"Device: {device}")
        if low_memory:
            print(f"Mode: Low Memory (8-bit quantization)")

        try:
            # Load processor
            self.processor = Owlv2Processor.from_pretrained(
                model_name,
                use_fast=use_fast,
                trust_remote_code=True
            )

            # Load model with memory optimizations
            if low_memory:
                # Use 8-bit quantization for lower memory usage
                try:
                    from transformers import AutoModelForZeroShotObjectDetection
                    self.model = Owlv2ForObjectDetection.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        load_in_8bit=True,
                        device_map=device
                    )
                except:
                    # Fallback to regular loading
                    self.model = Owlv2ForObjectDetection.from_pretrained(
                        model_name,
                        trust_remote_code=True
                    ).to(device)
            else:
                self.model = Owlv2ForObjectDetection.from_pretrained(
                    model_name,
                    trust_remote_code=True
                ).to(device)

            self.model.eval()

            # Enable optimizations
            torch.set_float32_matmul_precision('medium')  # For faster computation

            print("✓ Model loaded successfully")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure you have internet connection to download model")
            print("2. Run: huggingface-cli login")
            print("3. Or set HF token: export HF_TOKEN=your_token_here")
            raise

    def load_guide_images(self, guide_image_paths):
        """Load and resize reference images for guided detection"""
        guide_images = []
        for path in guide_image_paths:
            if os.path.exists(path):
                img = Image.open(path).convert('RGB')
                # Resize guide images for faster processing
                img = img.resize((448, 448), Image.LANCZOS)
                guide_images.append(img)
        return guide_images

    def detect_frame(self, frame_cv2, guide_images, confidence_threshold=0.1):
        """
        Detect objects in a frame using image-based queries

        Args:
            frame_cv2: OpenCV frame (BGR)
            guide_images: List of PIL images for guidance
            confidence_threshold: Minimum confidence score

        Returns:
            detections: List of {bbox, confidence}
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        if not guide_images:
            print("Warning: No guide images provided")
            return []

        detections = []

        # Process each guide image
        for idx, guide_img in enumerate(guide_images):
            try:
                # Prepare inputs
                inputs = self.processor(
                    images=frame_pil,
                    query_images=guide_img,
                    return_tensors="pt"
                )

                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Inference with no_grad to save memory
                with torch.no_grad():
                    outputs = self.model.image_guided_detection(
                        pixel_values=inputs['pixel_values'],
                        query_pixel_values=inputs['query_pixel_values']
                    )

                # Post-process results
                target_sizes = torch.Tensor([frame_pil.size[::-1]])
                results = self.processor.post_process_image_guided_detection(
                    outputs,
                    threshold=confidence_threshold,
                    target_sizes=target_sizes
                )

                # Extract boxes and scores
                if results[0]['boxes'].numel() > 0:
                    boxes = results[0]['boxes'].cpu().numpy()
                    scores = results[0]['scores'].cpu().numpy()

                    # Convert boxes to [x1, y1, x2, y2] format
                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = box
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(score),
                            'class': f'guide_object_{idx}'
                        })

            except Exception as e:
                print(f"Error processing guide image {idx}: {e}")
                continue

        # Non-maximum suppression
        detections = self._nms(detections, iou_threshold=0.5)

        return detections

    def _nms(self, detections, iou_threshold=0.5):
        """Non-maximum suppression to remove overlapping boxes"""
        if not detections:
            return detections

        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        for det in detections:
            is_duplicate = False
            for kept_det in keep:
                iou = self._calculate_iou(det['bbox'], kept_det['bbox'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(det)

        return keep

    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def process_video(self, video_path, guide_image_paths, output_path=None,
                     confidence_threshold=0.1, skip_frames=5, max_frames=None,
                     resize_scale=1.0, batch_size=1):
        """
        Process video and detect objects using guide images

        Args:
            video_path: Path to input video
            guide_image_paths: List of paths to guide images
            output_path: Path to save output video (optional)
            confidence_threshold: Minimum confidence score
            skip_frames: Process every Nth frame
            max_frames: Maximum number of frames to process
            resize_scale: Resize frame to this scale (0.5 = half resolution)
            batch_size: Number of frames to process in batch (1 for memory efficiency)

        Returns:
            detections_per_frame: Dict mapping frame_id to list of detections
        """
        # Load guide images
        guide_images = self.load_guide_images(guide_image_paths)

        if not guide_images:
            raise ValueError(f"No valid guide images found in {guide_image_paths}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Apply resize scale
        original_width, original_height = width, height
        if resize_scale != 1.0:
            width = int(width * resize_scale)
            height = int(height * resize_scale)

        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

        detections_per_frame = {}
        frame_id = 0
        processed_frames = 0

        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}, Resolution: {original_width}x{original_height}")
        print(f"Processing resolution: {width}x{height}")
        print(f"Guide images: {len(guide_images)}")

        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Process every skip_frames-th frame
                if frame_id % skip_frames == 0:
                    # Resize frame if needed
                    if resize_scale != 1.0:
                        resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    else:
                        resized_frame = frame

                    # Run detection
                    detections = self.detect_frame(resized_frame, guide_images, confidence_threshold)

                    # Scale bounding boxes back if resized
                    if resize_scale != 1.0:
                        scale_factor = 1.0 / resize_scale
                        for det in detections:
                            bbox = det['bbox']
                            det['bbox'] = [bbox[0] * scale_factor, bbox[1] * scale_factor,
                                         bbox[2] * scale_factor, bbox[3] * scale_factor]

                    detections_per_frame[frame_id] = detections

                    # Draw bounding boxes on original frame
                    frame_with_boxes = frame.copy()
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        conf_text = f"Conf: {det['confidence']:.2f}"
                        cv2.putText(frame_with_boxes, conf_text, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Add frame info
                    cv2.putText(frame_with_boxes, f"Frame: {frame_id}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(frame_with_boxes, f"Detections: {len(detections)}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    if writer:
                        writer.write(frame_with_boxes)

                    processed_frames += 1

                    if max_frames and processed_frames >= max_frames:
                        break

                    # Clear cache to free memory
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()

                frame_id += 1
                pbar.update(1)

        cap.release()
        if writer:
            writer.release()

        print(f"Processed {processed_frames} frames")
        if output_path:
            print(f"Output video saved: {output_path}")

        return detections_per_frame

    def save_detections(self, detections_per_frame, output_json_path, video_id=None):
        """Save detections to JSON file"""
        detections_by_class = {}

        for frame_id, frame_detections in detections_per_frame.items():
            for det in frame_detections:
                class_id = det.get('class', 'unknown')

                if class_id not in detections_by_class:
                    detections_by_class[class_id] = {
                        'bboxes': []
                    }

                x1, y1, x2, y2 = det['bbox']
                bbox_entry = {
                    'frame': int(frame_id),
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2)
                }
                detections_by_class[class_id]['bboxes'].append(bbox_entry)

        output_data = {
            'video_id': video_id or 'unknown',
            'detections': list(detections_by_class.values())
        }

        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Detections saved to {output_json_path}")


def detect_sample(sample_dir, output_dir="detection_results", skip_frames=5,
                 resize_scale=1.0, low_memory=True):
    """
    Detect objects in a sample directory using guide images

    Args:
        sample_dir: Path to sample directory (e.g., Laptop_0)
        output_dir: Output directory for results
        skip_frames: Process every Nth frame (default: 5)
        resize_scale: Resize frames to this scale (0.5 = half resolution)
        low_memory: Use 8-bit quantization for lower memory usage
    """
    sample_path = Path(sample_dir)
    video_path = sample_path / "drone_video.mp4"
    object_images_dir = sample_path / "object_images"

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    if not object_images_dir.exists():
        print(f"Object images directory not found: {object_images_dir}")
        return

    guide_images = sorted(object_images_dir.glob("*.jpg"))

    if not guide_images:
        print(f"No guide images found in {object_images_dir}")
        return

    output_path = Path(output_dir) / sample_path.name
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize detector with optimizations
    detector = OWLv2ObjectDetectorOptimized(low_memory=low_memory)

    output_video = output_path / "detected_video.mp4"
    output_json = output_path / "detections.json"

    print(f"\nOptimization settings:")
    print(f"  - Skip frames: {skip_frames}")
    print(f"  - Resize scale: {resize_scale}")
    print(f"  - Low memory mode: {low_memory}")

    detections = detector.process_video(
        str(video_path),
        [str(img) for img in guide_images],
        output_path=str(output_video),
        confidence_threshold=0.1,
        skip_frames=skip_frames,
        resize_scale=resize_scale
    )

    video_id = sample_path.name
    detector.save_detections(detections, str(output_json), video_id=video_id)

    print(f"\nResults saved in {output_path}")
    print(f"  - Video: {output_video}")
    print(f"  - JSON: {output_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="OWLv2 Image-Guided Object Detection (Optimized for Mac with 8-bit quantization)"
    )
    parser.add_argument("--sample-dir", type=str, required=True,
                       help="Path to sample directory (e.g., train/samples/Laptop_0)")
    parser.add_argument("--output-dir", type=str, default="detection_results",
                       help="Output directory for results")
    parser.add_argument("--skip-frames", type=int, default=5,
                       help="Process every Nth frame (default: 5)")
    parser.add_argument("--resize-scale", type=float, default=1.0,
                       help="Resize frames to this scale (0.5 = half resolution)")
    parser.add_argument("--no-low-memory", action="store_true",
                       help="Disable 8-bit quantization (uses more memory)")

    args = parser.parse_args()

    detect_sample(
        args.sample_dir,
        args.output_dir,
        skip_frames=args.skip_frames,
        resize_scale=args.resize_scale,
        low_memory=not args.no_low_memory
    )

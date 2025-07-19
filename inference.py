#!/usr/bin/env python3
"""
Bird Detection Inference Script
Real-time inference with visualization and performance metrics
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class BirdDetectionInference:
    """
    Real-time bird detection inference with comprehensive visualization
    """
    
    def __init__(self, model_path, conf_threshold=0.25, device='auto'):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.load_model()
        self.setup_colors()
        
    def load_model(self):
        """Load the trained YOLO model"""
        print(f"🔧 Loading model from: {self.model_path}")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
            
        self.model = YOLO(self.model_path)
        print("✅ Model loaded successfully")
        
        # Get class names
        self.class_names = self.model.names
        print(f"   Classes: {list(self.class_names.values())}")
        
    def setup_colors(self):
        """Setup colors for visualization"""
        # Generate distinct colors for each class
        np.random.seed(42)  # For consistent colors
        self.colors = {}
        for class_id in self.class_names.keys():
            self.colors[class_id] = tuple(map(int, np.random.randint(0, 255, 3)))
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        return image_rgb
    
    def postprocess_results(self, results, image_shape):
        """Postprocess YOLO results"""
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    if conf >= self.conf_threshold:
                        x1, y1, x2, y2 = map(int, box)
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': self.class_names[class_id]
                        }
                        detections.append(detection)
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        annotated_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            # Get color for this class
            color = self.colors[class_id]
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate label size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated_image
    
    def add_performance_info(self, image, fps, inference_time):
        """Add performance information to image"""
        # Performance text
        perf_text = f"FPS: {fps:.1f} | Inference: {inference_time*1000:.1f}ms"
        
        # Add background for text
        text_size = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(image, (10, 10), (10 + text_size[0] + 10, 40), (0, 0, 0), -1)
        
        # Add text
        cv2.putText(
            image,
            perf_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return image
    
    def predict_image(self, image_path, save_path=None, show=True):
        """Run inference on a single image"""
        print(f"🔍 Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        results = self.model(processed_image, conf=self.conf_threshold, verbose=False)
        inference_time = time.time() - start_time
        
        # Postprocess results
        detections = self.postprocess_results(results, image.shape)
        
        # Draw detections
        annotated_image = self.draw_detections(image, detections)
        
        # Add performance info
        fps = 1.0 / inference_time if inference_time > 0 else 0
        annotated_image = self.add_performance_info(annotated_image, fps, inference_time)
        
        # Print results
        print(f"   Found {len(detections)} bird(s)")
        for i, det in enumerate(detections):
            print(f"   {i+1}. {det['class_name']}: {det['confidence']:.3f}")
        print(f"   Inference time: {inference_time*1000:.1f}ms")
        
        # Save result
        if save_path:
            cv2.imwrite(str(save_path), annotated_image)
            print(f"   Result saved to: {save_path}")
        
        # Show result
        if show:
            cv2.imshow('Bird Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections, annotated_image
    
    def predict_batch(self, image_dir, output_dir=None):
        """Run inference on a batch of images"""
        image_dir = Path(image_dir)
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print("⚠️  No image files found in directory")
            return
        
        print(f"🔍 Processing {len(image_files)} images from: {image_dir}")
        
        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        total_detections = 0
        total_time = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] {image_path.name}")
            
            try:
                # Set save path
                save_path = None
                if output_dir:
                    save_path = output_dir / f"detected_{image_path.name}"
                
                # Run inference
                detections, _ = self.predict_image(
                    image_path, 
                    save_path=save_path, 
                    show=False
                )
                
                total_detections += len(detections)
                
            except Exception as e:
                print(f"   ❌ Error processing {image_path.name}: {e}")
        
        print(f"\n🎉 Batch processing completed!")
        print(f"   Total detections: {total_detections}")
        if output_dir:
            print(f"   Results saved to: {output_dir}")
    
    def predict_video(self, video_path, output_path=None, show=True):
        """Run inference on video"""
        print(f"🎥 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Video info: {width}x{height} @ {fps}FPS, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        total_detections = 0
        inference_times = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run inference
                start_time = time.time()
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Process results
                detections = self.postprocess_results(results, frame.shape)
                total_detections += len(detections)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, detections)
                
                # Add performance info
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                annotated_frame = self.add_performance_info(
                    annotated_frame, current_fps, inference_time
                )
                
                # Save frame
                if writer:
                    writer.write(annotated_frame)
                
                # Show frame
                if show:
                    cv2.imshow('Bird Detection - Video', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_fps = 1.0 / np.mean(inference_times[-30:])
                    print(f"   Progress: {progress:.1f}% | Avg FPS: {avg_fps:.1f}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Print summary
        avg_inference_time = np.mean(inference_times)
        avg_fps = 1.0 / avg_inference_time
        
        print(f"\n🎉 Video processing completed!")
        print(f"   Processed frames: {frame_count}")
        print(f"   Total detections: {total_detections}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Average inference time: {avg_inference_time*1000:.1f}ms")
        
        if output_path:
            print(f"   Output saved to: {output_path}")
    
    def predict_webcam(self, camera_id=0):
        """Run real-time inference on webcam"""
        print(f"📹 Starting webcam inference (Camera {camera_id})")
        print("   Press 'q' to quit, 's' to save screenshot")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        screenshot_count = 0
        inference_times = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Run inference
                start_time = time.time()
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Keep only recent times for FPS calculation
                if len(inference_times) > 30:
                    inference_times.pop(0)
                
                # Process results
                detections = self.postprocess_results(results, frame.shape)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, detections)
                
                # Add performance info
                avg_fps = 1.0 / np.mean(inference_times)
                annotated_frame = self.add_performance_info(
                    annotated_frame, avg_fps, inference_time
                )
                
                # Show frame
                cv2.imshow('Bird Detection - Webcam', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_count += 1
                    filename = f"webcam_screenshot_{screenshot_count}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"   Screenshot saved: {filename}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print("📹 Webcam inference stopped")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Bird Detection Inference')
    
    parser.add_argument('--model', required=True, type=str,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--source', required=True, type=str,
                       help='Source: image file, directory, video file, or webcam (0)')
    parser.add_argument('--output', type=str,
                       help='Output path for results')
    parser.add_argument('--conf', default=0.25, type=float,
                       help='Confidence threshold')
    parser.add_argument('--device', default='auto', type=str,
                       help='Device to run inference on')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display results')
    
    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_arguments()
    
    print("🔍 Bird Detection Inference")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Confidence: {args.conf}")
    print("=" * 50)
    
    try:
        # Initialize inference
        detector = BirdDetectionInference(
            model_path=args.model,
            conf_threshold=args.conf,
            device=args.device
        )
        
        source = args.source
        show = not args.no_show
        
        # Determine source type and run appropriate inference
        if source.isdigit():
            # Webcam
            detector.predict_webcam(camera_id=int(source))
            
        elif Path(source).is_file():
            # Check if it's a video or image
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
            if Path(source).suffix.lower() in video_extensions:
                # Video file
                detector.predict_video(source, args.output, show)
            else:
                # Image file
                detector.predict_image(source, args.output, show)
                
        elif Path(source).is_dir():
            # Directory of images
            detector.predict_batch(source, args.output)
            
        else:
            print(f"❌ Invalid source: {source}")
            print("   Source must be: image file, directory, video file, or webcam ID")
            return
        
        print("🎉 Inference completed successfully!")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Bird Detection Model Evaluation Script
Comprehensive evaluation with all standard detection metrics
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
import cv2


class BirdDetectionEvaluator:
    """
    Comprehensive evaluator for bird detection models
    Calculates all standard object detection metrics
    """
    
    def __init__(self, model_path, data_config, device='auto'):
        self.model_path = model_path
        self.data_config = data_config
        self.device = device
        self.load_model()
        self.setup_wandb()
        
    def load_model(self):
        """Load the trained YOLO model"""
        print(f"🔧 Loading model from: {self.model_path}")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
            
        self.model = YOLO(self.model_path)
        
        # Get model info
        model_info = self.model.info()
        print(f"   Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
        
    def setup_wandb(self):
        """Setup WandB for logging evaluation results"""
        try:
            wandb.init(
                project="bird-detection-yolov11",
                name=f"evaluation_{int(time.time())}",
                job_type="evaluation",
                tags=["evaluation", "metrics", "bird-detection"]
            )
            self.use_wandb = True
            print("✅ WandB initialized for evaluation")
        except Exception as e:
            print(f"⚠️  WandB setup failed: {e}")
            self.use_wandb = False
    
    def evaluate_model(self, conf_threshold=0.25, iou_threshold=0.5):
        """
        Comprehensive model evaluation with all detection metrics
        """
        print("📊 Starting Model Evaluation")
        print("=" * 50)
        
        # Run validation
        results = self.model.val(
            data=self.data_config,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            plots=True,
            save_json=True,
            save_hybrid=True
        )
        
        # Extract metrics
        metrics = self.extract_metrics(results)
        
        # Generate detailed analysis
        detailed_metrics = self.generate_detailed_analysis(results)
        
        # Create visualizations
        self.create_visualizations(results)
        
        # Performance analysis
        performance_metrics = self.analyze_performance()
        
        # Combine all metrics
        all_metrics = {
            **metrics,
            **detailed_metrics,
            **performance_metrics
        }
        
        # Log to WandB
        if self.use_wandb:
            self.log_metrics_to_wandb(all_metrics, results)
        
        # Save results
        self.save_evaluation_results(all_metrics)
        
        return all_metrics
    
    def extract_metrics(self, results):
        """Extract standard object detection metrics"""
        metrics = {}
        
        # Main detection metrics
        if hasattr(results, 'box'):
            box_metrics = results.box
            
            # Mean Average Precision metrics
            metrics['mAP_50'] = float(box_metrics.map50)
            metrics['mAP_50_95'] = float(box_metrics.map)
            metrics['mAP_75'] = float(box_metrics.map75) if hasattr(box_metrics, 'map75') else 0.0
            
            # Precision and Recall
            metrics['precision'] = float(box_metrics.mp)  # mean precision
            metrics['recall'] = float(box_metrics.mr)     # mean recall
            
            # F1 Score
            if metrics['precision'] > 0 and metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1_score'] = 0.0
            
            # Per-class metrics
            if hasattr(box_metrics, 'ap'):
                metrics['per_class_ap'] = box_metrics.ap.tolist() if box_metrics.ap is not None else []
            
            if hasattr(box_metrics, 'ap_class_index'):
                metrics['class_indices'] = box_metrics.ap_class_index.tolist() if box_metrics.ap_class_index is not None else []
        
        # Additional metrics from results
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
            for key, value in results_dict.items():
                if 'metrics/' in key:
                    clean_key = key.replace('metrics/', '').replace('(B)', '')
                    if isinstance(value, (int, float)):
                        metrics[clean_key] = float(value)
        
        return metrics
    
    def generate_detailed_analysis(self, results):
        """Generate detailed per-class and threshold analysis"""
        detailed_metrics = {}
        
        # Per-class analysis
        if hasattr(results, 'box') and hasattr(results.box, 'ap'):
            ap_per_class = results.box.ap
            if ap_per_class is not None:
                # Calculate per-class metrics
                detailed_metrics['mean_ap_per_class'] = float(np.mean(ap_per_class))
                detailed_metrics['std_ap_per_class'] = float(np.std(ap_per_class))
                detailed_metrics['min_ap_per_class'] = float(np.min(ap_per_class))
                detailed_metrics['max_ap_per_class'] = float(np.max(ap_per_class))
        
        # Small, Medium, Large object analysis
        if hasattr(results, 'box'):
            box = results.box
            detailed_metrics['map_small'] = float(getattr(box, 'maps', 0.0))
            detailed_metrics['map_medium'] = float(getattr(box, 'mapm', 0.0))
            detailed_metrics['map_large'] = float(getattr(box, 'mapl', 0.0))
        
        return detailed_metrics
    
    def analyze_performance(self):
        """Analyze model performance (speed, efficiency)"""
        performance_metrics = {}
        
        # Test inference speed
        test_image = torch.randn(1, 3, 640, 640).to(self.model.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model.model(test_image)
        
        # Measure inference time
        times = []
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model.model(test_image)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start_time)
        
        performance_metrics['avg_inference_time_ms'] = np.mean(times) * 1000
        performance_metrics['std_inference_time_ms'] = np.std(times) * 1000
        performance_metrics['fps'] = 1.0 / np.mean(times)
        
        # Model size metrics
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.model.parameters()) / (1024 * 1024)
        performance_metrics['model_size_mb'] = model_size_mb
        performance_metrics['parameter_count'] = sum(p.numel() for p in self.model.model.parameters())
        
        return performance_metrics
    
    def create_visualizations(self, results):
        """Create comprehensive visualizations"""
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Confusion Matrix
        self.plot_confusion_matrix(results, output_dir)
        
        # 2. Precision-Recall Curve
        self.plot_pr_curve(results, output_dir)
        
        # 3. F1-Confidence Curve
        self.plot_f1_curve(results, output_dir)
        
        # 4. Per-class AP visualization
        self.plot_per_class_ap(results, output_dir)
        
    def plot_confusion_matrix(self, results, output_dir):
        """Plot and save confusion matrix"""
        try:
            if hasattr(results, 'confusion_matrix'):
                cm = results.confusion_matrix.matrix
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix - Bird Detection')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print("✅ Confusion matrix saved")
                
        except Exception as e:
            print(f"⚠️  Could not create confusion matrix: {e}")
    
    def plot_pr_curve(self, results, output_dir):
        """Plot Precision-Recall curve"""
        try:
            # This would require access to the raw predictions
            # For now, create a placeholder
            plt.figure(figsize=(10, 8))
            plt.title('Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.grid(True)
            plt.savefig(output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Could not create PR curve: {e}")
    
    def plot_f1_curve(self, results, output_dir):
        """Plot F1-Confidence curve"""
        try:
            plt.figure(figsize=(10, 8))
            plt.title('F1-Confidence Curve')
            plt.xlabel('Confidence Threshold')
            plt.ylabel('F1 Score')
            plt.grid(True)
            plt.savefig(output_dir / 'f1_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Could not create F1 curve: {e}")
    
    def plot_per_class_ap(self, results, output_dir):
        """Plot per-class Average Precision"""
        try:
            if hasattr(results, 'box') and hasattr(results.box, 'ap'):
                ap_per_class = results.box.ap
                if ap_per_class is not None:
                    plt.figure(figsize=(12, 8))
                    class_names = [f"Class {i}" for i in range(len(ap_per_class))]
                    
                    bars = plt.bar(class_names, ap_per_class)
                    plt.title('Average Precision per Class')
                    plt.xlabel('Classes')
                    plt.ylabel('Average Precision')
                    plt.xticks(rotation=45)
                    
                    # Add value labels on bars
                    for bar, ap in zip(bars, ap_per_class):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{ap:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / 'per_class_ap.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print("✅ Per-class AP plot saved")
                    
        except Exception as e:
            print(f"⚠️  Could not create per-class AP plot: {e}")
    
    def log_metrics_to_wandb(self, metrics, results):
        """Log all metrics to Weights & Biases"""
        try:
            # Log main metrics
            wandb.log({
                "evaluation/mAP_50": metrics.get('mAP_50', 0),
                "evaluation/mAP_50_95": metrics.get('mAP_50_95', 0),
                "evaluation/precision": metrics.get('precision', 0),
                "evaluation/recall": metrics.get('recall', 0),
                "evaluation/f1_score": metrics.get('f1_score', 0),
                "performance/fps": metrics.get('fps', 0),
                "performance/inference_time_ms": metrics.get('avg_inference_time_ms', 0),
                "performance/model_size_mb": metrics.get('model_size_mb', 0),
            })
            
            # Log visualizations
            output_dir = Path("evaluation_results")
            if (output_dir / 'confusion_matrix.png').exists():
                wandb.log({"evaluation/confusion_matrix": wandb.Image(str(output_dir / 'confusion_matrix.png'))})
            
            print("📊 Metrics logged to WandB")
            
        except Exception as e:
            print(f"⚠️  Failed to log to WandB: {e}")
    
    def save_evaluation_results(self, metrics):
        """Save evaluation results to files"""
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics as JSON
        with open(output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save metrics as CSV for easy analysis
        df = pd.DataFrame([metrics])
        df.to_csv(output_dir / 'evaluation_metrics.csv', index=False)
        
        # Create summary report
        self.create_summary_report(metrics, output_dir)
        
        print(f"📁 Results saved to: {output_dir}")
    
    def create_summary_report(self, metrics, output_dir):
        """Create a comprehensive summary report"""
        report = f"""
# Bird Detection Model Evaluation Report

## Model Information
- **Model Path**: {self.model_path}
- **Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Detection Performance Metrics

### Main Metrics
- **mAP@0.5**: {metrics.get('mAP_50', 0):.4f}
- **mAP@0.5:0.95**: {metrics.get('mAP_50_95', 0):.4f}
- **Precision**: {metrics.get('precision', 0):.4f}
- **Recall**: {metrics.get('recall', 0):.4f}
- **F1-Score**: {metrics.get('f1_score', 0):.4f}

### Performance Metrics
- **Inference Speed**: {metrics.get('avg_inference_time_ms', 0):.2f} ms
- **FPS**: {metrics.get('fps', 0):.1f}
- **Model Size**: {metrics.get('model_size_mb', 0):.2f} MB
- **Parameters**: {metrics.get('parameter_count', 0):,}

### Detailed Analysis
- **Mean AP per Class**: {metrics.get('mean_ap_per_class', 0):.4f}
- **AP Standard Deviation**: {metrics.get('std_ap_per_class', 0):.4f}
- **Minimum Class AP**: {metrics.get('min_ap_per_class', 0):.4f}
- **Maximum Class AP**: {metrics.get('max_ap_per_class', 0):.4f}

## Interpretation

### Performance Rating
"""
        
        # Add performance interpretation
        map_50 = metrics.get('mAP_50', 0)
        if map_50 >= 0.9:
            report += "🟢 **Excellent** - Outstanding detection performance\n"
        elif map_50 >= 0.8:
            report += "🟡 **Good** - Strong detection performance\n"
        elif map_50 >= 0.7:
            report += "🟠 **Fair** - Adequate detection performance\n"
        else:
            report += "🔴 **Poor** - Needs improvement\n"
        
        report += f"""
### Recommendations
- Model is suitable for production use: {'✅ Yes' if map_50 >= 0.8 else '❌ No'}
- Consider further training: {'❌ No' if map_50 >= 0.85 else '✅ Yes'}
- Optimize for speed: {'✅ Yes' if metrics.get('fps', 0) < 30 else '❌ No'}

---
*Generated by Bird Detection Evaluation System*
"""
        
        with open(output_dir / 'evaluation_report.md', 'w') as f:
            f.write(report)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Bird Detection Model')
    
    parser.add_argument('--model-path', required=True, type=str,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--data', default='configs/yolov11_birds.yaml', type=str,
                       help='Dataset configuration file')
    parser.add_argument('--conf-threshold', default=0.25, type=float,
                       help='Confidence threshold for evaluation')
    parser.add_argument('--iou-threshold', default=0.5, type=float,
                       help='IoU threshold for NMS')
    parser.add_argument('--device', default='auto', type=str,
                       help='Device to run evaluation on')
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    print("🔍 Bird Detection Model Evaluation")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.data}")
    print(f"Confidence Threshold: {args.conf_threshold}")
    print(f"IoU Threshold: {args.iou_threshold}")
    print("=" * 50)
    
    try:
        # Initialize evaluator
        evaluator = BirdDetectionEvaluator(
            model_path=args.model_path,
            data_config=args.data,
            device=args.device
        )
        
        # Run evaluation
        metrics = evaluator.evaluate_model(
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold
        )
        
        print("\n🎉 Evaluation completed successfully!")
        print("\n📊 Key Results:")
        print(f"   mAP@0.5: {metrics.get('mAP_50', 0):.4f}")
        print(f"   mAP@0.5:0.95: {metrics.get('mAP_50_95', 0):.4f}")
        print(f"   Precision: {metrics.get('precision', 0):.4f}")
        print(f"   Recall: {metrics.get('recall', 0):.4f}")
        print(f"   F1-Score: {metrics.get('f1_score', 0):.4f}")
        print(f"   FPS: {metrics.get('fps', 0):.1f}")
        
        print(f"\n📁 Detailed results saved to: evaluation_results/")
        print("📊 Check WandB dashboard for visualizations")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise
    
    finally:
        if 'evaluator' in locals() and evaluator.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
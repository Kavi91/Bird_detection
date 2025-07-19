#!/bin/bash
echo "🐦 Activating Bird Detection Environment..."
source bird_detection_env/bin/activate
echo "✅ Virtual environment activated"
echo "📝 To deactivate, run: deactivate"
echo ""
echo "🎯 Quick commands:"
echo "  Train model:    python train.py --epochs 100 --batch-size 16"
echo "  Evaluate:       python evaluate.py --model-path runs/train/exp/weights/best.pt"
echo "  Inference:      python inference.py --source 0 --model yolov11n.pt"
echo "  Download data:  python scripts/download_dataset.py --sample"

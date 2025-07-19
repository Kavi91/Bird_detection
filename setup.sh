#!/bin/bash

# Bird Detection Project Setup Script - CUB-200-2011 Dataset
# Complete automated setup for YOLOv11 bird detection with CUB dataset

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_NAME="Bird_detection"
VENV_NAME="bird_detection_env"
PYTHON_VERSION="3.8"
CUB_DATASET_URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

echo -e "${BLUE}🐦 Bird Detection Project Setup - CUB-200-2011 Dataset${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "${CYAN}Setting up complete YOLOv11 bird detection project...${NC}"
echo -e "${CYAN}This will create a virtual environment and install all dependencies${NC}"
echo -e "${CYAN}Dataset: CUB-200-2011 (200 bird species, 11,788 images)${NC}"
echo -e "${BLUE}============================================================${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}🔧 $1${NC}"
}

# Check if Python 3 is installed
check_python() {
    print_info "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
        print_status "Python $PYTHON_VER found"
        
        # Check if version is >= 3.8
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python version is compatible (>= 3.8)"
        else
            print_error "Python version must be >= 3.8"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        echo "Please install Python 3.8 or higher"
        exit 1
    fi
}

# Check if pip is available
check_pip() {
    print_info "Checking pip installation..."
    
    if command -v pip3 &> /dev/null; then
        print_status "pip3 found"
    else
        print_error "pip3 is not installed"
        echo "Installing pip3..."
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update
            sudo apt-get install -y python3-pip
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
            python3 get-pip.py
            rm get-pip.py
        fi
    fi
}

# Check if venv module is available
check_venv() {
    print_info "Checking python3-venv..."
    
    if python3 -m venv --help &> /dev/null; then
        print_status "python3-venv is available"
    else
        print_warning "python3-venv not found, installing..."
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update
            sudo apt-get install -y python3-venv
        fi
    fi
}

# Create project directory structure for CUB dataset
create_project_structure() {
    print_info "Creating CUB-200-2011 project directory structure..."
    
    # Create all necessary directories
    mkdir -p data/{train,val,test}/{images,labels}
    mkdir -p data/downloads
    mkdir -p scripts
    mkdir -p configs
    mkdir -p runs/{train,val,detect}
    mkdir -p evaluation_results
    mkdir -p visualization_results
    mkdir -p models
    mkdir -p notebooks
    mkdir -p docs
    mkdir -p wandb
    
    print_status "Project structure created for CUB dataset"
}

# Create virtual environment
create_virtual_environment() {
    print_info "Creating virtual environment: $VENV_NAME"
    
    # Remove existing virtual environment if it exists
    if [ -d "$VENV_NAME" ]; then
        print_warning "Removing existing virtual environment..."
        rm -rf "$VENV_NAME"
    fi
    
    # Create new virtual environment
    python3 -m venv "$VENV_NAME"
    print_status "Virtual environment created: $VENV_NAME"
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    print_status "Virtual environment activated"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip
    print_status "pip upgraded"
}

# Install PyTorch with CUDA support
install_pytorch() {
    print_info "Installing PyTorch with CUDA support..."
    
    # Check if NVIDIA GPU is available
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected, installing PyTorch with CUDA support"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_warning "No NVIDIA GPU detected, installing CPU-only PyTorch"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_status "PyTorch installed"
}

# Install core dependencies for bird detection
install_dependencies() {
    print_info "Installing core dependencies..."
    
    # Create requirements.txt content specific to this project
    cat > requirements.txt << EOF
# Core ML and Computer Vision
ultralytics>=8.0.196
opencv-python>=4.8.0
Pillow>=9.5.0

# Experiment tracking and visualization
wandb>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0

# Data manipulation and analysis
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0

# Utilities
tqdm>=4.65.0
PyYAML>=6.0
requests>=2.31.0
psutil>=5.9.0

# Dataset handling
roboflow>=1.1.0

# Development and notebooks
jupyter>=1.0.0
ipywidgets>=8.0.0
black>=23.0.0
flake8>=6.0.0

# Optional dependencies for advanced features
albumentations>=1.3.0
onnx>=1.14.0
onnxruntime>=1.15.0
EOF

    # Install dependencies
    pip install -r requirements.txt
    print_status "Core dependencies installed"
}

# Create configuration files for CUB dataset
create_config_files() {
    print_info "Creating CUB-200-2011 configuration files..."
    
    # Create .gitignore
    cat > .gitignore << EOF
# Virtual Environment
bird_detection_env/
venv/
env/
.venv/

# Dataset files (large)
data/CUB_200_2011/
data/downloads/
*.tgz
*.tar.gz
*.zip

# Generated data
data/train/
data/val/
data/test/
*.jpg
*.jpeg
*.png
*.mp4
*.avi
*.mov

# Model files
models/*.pt
*.onnx
*.engine
*.trt

# Results and logs
runs/
wandb/
evaluation_results/
visualization_results/
*.log
logs/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store
Thumbs.db

# Environment variables
.env
.env.local

# Temporary files
tmp/
temp/
*.tmp
*.temp
*.cache
EOF

    # Create environment activation script
    cat > activate_env.sh << EOF
#!/bin/bash
echo "🐦 Activating Bird Detection Environment..."
echo "CUB-200-2011 Dataset Project"
echo "================================"
source bird_detection_env/bin/activate
echo "✅ Virtual environment activated"
echo "📝 To deactivate, run: deactivate"
echo ""
echo "🎯 Quick commands:"
echo "  Setup check:      python verify_setup.py"
echo "  Download dataset: python scripts/download_cub_dataset.py"
echo "  Train model:      python train.py --data configs/cub_birds.yaml --epochs 100 --batch-size 32"
echo "  Test model:       python test.py --model-path runs/train/exp/weights/best.pt"
echo "  Run inference:    python inference.py --source 0 --model runs/train/exp/weights/best.pt"
echo ""
echo "📊 WandB Dashboard: https://wandb.ai/"
echo "📚 Dataset Info: CUB-200-2011 (200 species, 11,788 images)"
EOF
    chmod +x activate_env.sh

    # Create dataset configuration directory
    mkdir -p configs
    
    # Create full CUB-200-2011 configuration (200 classes)
    cat > configs/cub_birds.yaml << EOF
# CUB-200-2011 Bird Dataset Configuration - Full 200 Classes
path: data
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 200

# Class names (all 200 CUB-200-2011 species)
names:
  0: '001.Black footed Albatross'
  1: '002.Laysan Albatross'
  2: '003.Sooty Albatross'
  3: '004.Groove billed Ani'
  4: '005.Crested Auklet'
  5: '006.Least Auklet'
  6: '007.Parakeet Auklet'
  7: '008.Rhinoceros Auklet'
  8: '009.Brewer Blackbird'
  9: '010.Red winged Blackbird'
  10: '011.Rusty Blackbird'
  11: '012.Yellow headed Blackbird'
  12: '013.Bobolink'
  13: '014.Indigo Bunting'
  14: '015.Lazuli Bunting'
  15: '016.Painted Bunting'
  16: '017.Cardinal'
  17: '018.Spotted Catbird'
  18: '019.Gray Catbird'
  19: '020.Yellow breasted Chat'
  20: '021.Eastern Towhee'
  21: '022.Chuck will Widow'
  22: '023.Brandt Cormorant'
  23: '024.Red faced Cormorant'
  24: '025.Pelagic Cormorant'
  25: '026.Bronzed Cowbird'
  26: '027.Shiny Cowbird'
  27: '028.Brown Creeper'
  28: '029.American Crow'
  29: '030.Fish Crow'
  30: '031.Black billed Cuckoo'
  31: '032.Mangrove Cuckoo'
  32: '033.Yellow billed Cuckoo'
  33: '034.Gray crowned Rosy Finch'
  34: '035.Purple Finch'
  35: '036.Northern Flicker'
  36: '037.Acadian Flycatcher'
  37: '038.Great Crested Flycatcher'
  38: '039.Least Flycatcher'
  39: '040.Olive sided Flycatcher'
  40: '041.Scissor tailed Flycatcher'
  41: '042.Vermilion Flycatcher'
  42: '043.Yellow bellied Flycatcher'
  43: '044.Frigatebird'
  44: '045.Northern Fulmar'
  45: '046.Gadwall'
  46: '047.American Goldfinch'
  47: '048.European Goldfinch'
  48: '049.Boat tailed Grackle'
  49: '050.Eared Grebe'
  50: '051.Horned Grebe'
  51: '052.Pied billed Grebe'
  52: '053.Western Grebe'
  53: '054.Blue Grosbeak'
  54: '055.Evening Grosbeak'
  55: '056.Pine Grosbeak'
  56: '057.Rose breasted Grosbeak'
  57: '058.Pigeon Guillemot'
  58: '059.California Gull'
  59: '060.Glaucous winged Gull'
  60: '061.Heermann Gull'
  61: '062.Herring Gull'
  62: '063.Ivory Gull'
  63: '064.Ring billed Gull'
  64: '065.Slaty backed Gull'
  65: '066.Western Gull'
  66: '067.Anna Hummingbird'
  67: '068.Ruby throated Hummingbird'
  68: '069.Rufous Hummingbird'
  69: '070.Green Violetear'
  70: '071.Long tailed Jaeger'
  71: '072.Pomarine Jaeger'
  72: '073.Blue Jay'
  73: '074.Florida Jay'
  74: '075.Green Jay'
  75: '076.Dark eyed Junco'
  76: '077.Tropical Kingbird'
  77: '078.Gray Kingbird'
  78: '079.Belted Kingfisher'
  79: '080.Green Kingfisher'
  80: '081.Pied Kingfisher'
  81: '082.Ringed Kingfisher'
  82: '083.White breasted Kingfisher'
  83: '084.Red legged Kittiwake'
  84: '085.Horned Lark'
  85: '086.Pacific Loon'
  86: '087.Mallard'
  87: '088.Western Meadowlark'
  88: '089.Hooded Merganser'
  89: '090.Red breasted Merganser'
  90: '091.Mockingbird'
  91: '092.Nighthawk'
  92: '093.Clark Nutcracker'
  93: '094.White breasted Nuthatch'
  94: '095.Baltimore Oriole'
  95: '096.Hooded Oriole'
  96: '097.Orchard Oriole'
  97: '098.Scott Oriole'
  98: '099.Ovenbird'
  99: '100.Brown Pelican'
  100: '101.White Pelican'
  101: '102.Western Wood Pewee'
  102: '103.Sayornis'
  103: '104.American Pipit'
  104: '105.Whip poor Will'
  105: '106.Horned Puffin'
  106: '107.Common Raven'
  107: '108.White necked Raven'
  108: '109.American Redstart'
  109: '110.Geococcyx'
  110: '111.Loggerhead Shrike'
  111: '112.Great Grey Shrike'
  112: '113.Baird Sparrow'
  113: '114.Black throated Sparrow'
  114: '115.Brewer Sparrow'
  115: '116.Chipping Sparrow'
  116: '117.Clay colored Sparrow'
  117: '118.House Sparrow'
  118: '119.Field Sparrow'
  119: '120.Fox Sparrow'
  120: '121.Grasshopper Sparrow'
  121: '122.Harris Sparrow'
  122: '123.Henslow Sparrow'
  123: '124.Le Conte Sparrow'
  124: '125.Lincoln Sparrow'
  125: '126.Nelson Sharp tailed Sparrow'
  126: '127.Savannah Sparrow'
  127: '128.Seaside Sparrow'
  128: '129.Song Sparrow'
  129: '130.Tree Sparrow'
  130: '131.Vesper Sparrow'
  131: '132.White crowned Sparrow'
  132: '133.White throated Sparrow'
  133: '134.Cape Glossy Starling'
  134: '135.Bank Swallow'
  135: '136.Barn Swallow'
  136: '137.Cliff Swallow'
  137: '138.Tree Swallow'
  138: '139.Scarlet Tanager'
  139: '140.Summer Tanager'
  140: '141.Artic Tern'
  141: '142.Black Tern'
  142: '143.Caspian Tern'
  143: '144.Common Tern'
  144: '145.Elegant Tern'
  145: '146.Forsters Tern'
  146: '147.Least Tern'
  147: '148.Green tailed Towhee'
  148: '149.Brown Thrasher'
  149: '150.Sage Thrasher'
  150: '151.Black capped Vireo'
  151: '152.Blue headed Vireo'
  152: '153.Philadelphia Vireo'
  153: '154.Red eyed Vireo'
  154: '155.Warbling Vireo'
  155: '156.White eyed Vireo'
  156: '157.Yellow throated Vireo'
  157: '158.Bay breasted Warbler'
  158: '159.Black and white Warbler'
  159: '160.Black throated Blue Warbler'
  160: '161.Blue winged Warbler'
  161: '162.Canada Warbler'
  162: '163.Cape May Warbler'
  163: '164.Cerulean Warbler'
  164: '165.Chestnut sided Warbler'
  165: '166.Golden winged Warbler'
  166: '167.Hooded Warbler'
  167: '168.Kentucky Warbler'
  168: '169.Magnolia Warbler'
  169: '170.Mourning Warbler'
  170: '171.Myrtle Warbler'
  171: '172.Nashville Warbler'
  172: '173.Orange crowned Warbler'
  173: '174.Palm Warbler'
  174: '175.Pine Warbler'
  175: '176.Prairie Warbler'
  176: '177.Prothonotary Warbler'
  177: '178.Swainson Warbler'
  178: '179.Tennessee Warbler'
  179: '180.Wilson Warbler'
  180: '181.Worm eating Warbler'
  181: '182.Yellow Warbler'
  182: '183.Northern Waterthrush'
  183: '184.Louisiana Waterthrush'
  184: '185.Bohemian Waxwing'
  185: '186.Cedar Waxwing'
  186: '187.American Three toed Woodpecker'
  187: '188.Pileated Woodpecker'
  188: '189.Red bellied Woodpecker'
  189: '190.Red cockaded Woodpecker'
  190: '191.Red headed Woodpecker'
  191: '192.Downy Woodpecker'
  192: '193.Bewick Wren'
  193: '194.Cactus Wren'
  194: '195.Carolina Wren'
  195: '196.House Wren'
  196: '197.Marsh Wren'
  197: '198.Rock Wren'
  198: '199.Winter Wren'
  199: '200.Common Yellowthroat'
EOF

    # Create simplified configuration for testing (first 10 classes)
    cat > configs/cub_birds_simple.yaml << EOF
# CUB-200-2011 Bird Dataset Configuration - Simplified (10 Classes)
path: data
train: train/images
val: val/images
test: test/images

# Number of classes (subset for quick testing)
nc: 10

# Class names (first 10 species)
names:
  0: '001.Black footed Albatross'
  1: '002.Laysan Albatross'
  2: '003.Sooty Albatross'
  3: '004.Groove billed Ani'
  4: '005.Crested Auklet'
  5: '006.Least Auklet'
  6: '007.Parakeet Auklet'
  7: '008.Rhinoceros Auklet'
  8: '009.Brewer Blackbird'
  9: '010.Red winged Blackbird'
EOF

    print_status "CUB-200-2011 configuration files created"
}

# Create CUB dataset download script
create_download_script() {
    print_info "Creating CUB dataset download script..."
    
    mkdir -p scripts
    cat > scripts/download_cub_dataset.py << 'EOF'
#!/usr/bin/env python3
"""
CUB-200-2011 Dataset Download and Conversion Script
Downloads the CUB dataset and converts it to YOLO format
"""

import os
import requests
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

class CUBDatasetDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.downloads_dir = self.data_dir / "downloads"
        self.cub_url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
        self.dataset_file = "CUB_200_2011.tgz"
        
    def download_dataset(self):
        """Download CUB-200-2011 dataset"""
        print("📥 Downloading CUB-200-2011 dataset...")
        
        # Create downloads directory
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_path = self.downloads_dir / self.dataset_file
        
        # Check if already downloaded
        if dataset_path.exists():
            print(f"✅ Dataset already downloaded: {dataset_path}")
            return dataset_path
        
        # Download with progress bar
        try:
            response = requests.get(self.cub_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dataset_path, 'wb') as file:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Download completed: {dataset_path}")
            return dataset_path
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            if dataset_path.exists():
                dataset_path.unlink()
            raise
    
    def extract_dataset(self, dataset_path):
        """Extract CUB dataset"""
        print("📂 Extracting CUB-200-2011 dataset...")
        
        extract_path = self.data_dir / "CUB_200_2011"
        
        if extract_path.exists():
            print(f"✅ Dataset already extracted: {extract_path}")
            return extract_path
        
        try:
            with tarfile.open(dataset_path, 'r:gz') as tar:
                tar.extractall(self.data_dir)
            
            print(f"✅ Extraction completed: {extract_path}")
            return extract_path
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            raise
    
    def convert_to_yolo(self, cub_path):
        """Convert CUB annotations to YOLO format"""
        print("🔄 Converting CUB dataset to YOLO format...")
        
        # This is a placeholder - the actual conversion would be done
        # by the existing convert_cub_to_yolo.py script
        convert_script = Path("convert_cub_to_yolo.py")
        
        if convert_script.exists():
            print("✅ Running existing conversion script...")
            os.system(f"python {convert_script}")
        else:
            print("⚠️  Conversion script not found. Please run convert_cub_to_yolo.py manually")
            print("   Expected structure:")
            print("   data/train/images/ - Training images")
            print("   data/train/labels/ - Training labels (YOLO format)")
            print("   data/val/images/ - Validation images")
            print("   data/val/labels/ - Validation labels (YOLO format)")
    
    def verify_dataset(self):
        """Verify dataset structure"""
        print("🔍 Verifying dataset structure...")
        
        required_dirs = [
            "data/train/images",
            "data/train/labels", 
            "data/val/images",
            "data/val/labels"
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists():
                count = len(list(path.glob("*")))
                print(f"✅ {dir_path}: {count} files")
            else:
                print(f"❌ Missing: {dir_path}")
        
        # Count total images
        train_images = len(list(Path("data/train/images").glob("*.jpg"))) if Path("data/train/images").exists() else 0
        val_images = len(list(Path("data/val/images").glob("*.jpg"))) if Path("data/val/images").exists() else 0
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Training images: {train_images}")
        print(f"   Validation images: {val_images}")
        print(f"   Total: {train_images + val_images}")
        
        if train_images > 4000 and val_images > 1000:
            print("✅ Dataset appears to be properly set up!")
        else:
            print("⚠️  Dataset might not be fully converted. Check conversion process.")

def main():
    parser = argparse.ArgumentParser(description='Download and setup CUB-200-2011 dataset')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--skip-download', action='store_true', help='Skip download if file exists')
    args = parser.parse_args()
    
    print("🐦 CUB-200-2011 Dataset Setup")
    print("=" * 40)
    
    downloader = CUBDatasetDownloader(args.data_dir)
    
    try:
        # Download dataset
        if not args.skip_download:
            dataset_path = downloader.download_dataset()
        else:
            dataset_path = downloader.downloads_dir / downloader.dataset_file
        
        # Extract dataset
        cub_path = downloader.extract_dataset(dataset_path)
        
        # Convert to YOLO format
        downloader.convert_to_yolo(cub_path)
        
        # Verify setup
        downloader.verify_dataset()
        
        print("\n🎉 CUB-200-2011 dataset setup completed!")
        print("\nNext steps:")
        print("1. python train.py --data configs/cub_birds.yaml --epochs 100")
        print("2. Check WandB dashboard for training progress")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("\nManual setup instructions:")
        print("1. Download CUB_200_2011.tgz from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html")
        print("2. Extract to data/CUB_200_2011/")
        print("3. Run convert_cub_to_yolo.py")

if __name__ == "__main__":
    main()
EOF
    chmod +x scripts/download_cub_dataset.py
    
    print_status "CUB dataset download script created"
}

# Test installation with CUB dataset
test_installation() {
    print_info "Testing installation..."
    
    # Test Python imports
    python3 -c "
import torch
import ultralytics
import cv2
import matplotlib
import numpy as np
import pandas as pd
print('✅ All core packages imported successfully')

# Test GPU
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  CUDA not available, will use CPU')

# Test YOLOv11 (this will download the model)
from ultralytics import YOLO
print('📥 Downloading YOLOv11 model...')
model = YOLO('yolo11n.pt')
print('✅ YOLOv11 model loaded successfully')

# Move model to models directory
import shutil
import os
os.makedirs('models', exist_ok=True)
if os.path.exists('yolo11n.pt'):
    shutil.move('yolo11n.pt', 'models/yolo11n.pt')
    print('✅ Model moved to models/yolo11n.pt')
"
    
    if [ $? -eq 0 ]; then
        print_status "Installation test passed"
    else
        print_error "Installation test failed"
        return 1
    fi
}

# Create quick start script
create_quick_start() {
    print_info "Creating quick start script..."
    
    cat > quick_start.py << EOF
#!/usr/bin/env python3
"""
Quick start script for CUB-200-2011 bird detection project
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def main():
    print("🐦 CUB-200-2011 Bird Detection Quick Start")
    print("=" * 50)
    
    # Test GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  Using CPU")
    
    # Load model
    print("📥 Loading YOLOv11 model...")
    model_path = "models/yolo11n.pt" if Path("models/yolo11n.pt").exists() else "yolo11n.pt"
    model = YOLO(model_path)
    print("✅ Model loaded successfully")
    
    # Test with sample image
    print("🔍 Testing inference...")
    
    # Create a dummy image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Run inference
    results = model(test_image, verbose=False)
    print(f"✅ Inference completed")
    print(f"   Detections: {len(results[0].boxes) if results[0].boxes is not None else 0}")
    
    # Check dataset structure
    print("\n📊 Checking dataset structure...")
    data_dirs = ["data/train/images", "data/val/images", "data/test/images"]
    
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            count = len(list(Path(data_dir).glob("*.jpg")))
            print(f"   {data_dir}: {count} images")
        else:
            print(f"   {data_dir}: Not found")
    
    print("\n🎉 Setup test completed successfully!")
    print("\nNext steps:")
    if not Path("data/train/images").exists() or len(list(Path("data/train/images").glob("*.jpg"))) == 0:
        print("1. python scripts/download_cub_dataset.py")
        print("2. python train.py --data configs/cub_birds.yaml --epochs 100 --batch-size 32")
    else:
        print("1. python train.py --data configs/cub_birds.yaml --epochs 100 --batch-size 32")
    print("3. python test.py --model-path runs/train/exp/weights/best.pt")
    print("4. python inference.py --source 0 --model runs/train/exp/weights/best.pt")
    
    print(f"\n📚 Dataset: CUB-200-2011 (200 bird species)")
    print(f"📊 WandB: https://wandb.ai/")

if __name__ == "__main__":
    main()
EOF

    print_status "Quick start script created"
}

# Print completion message
print_completion() {
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}🎉 CUB-200-2011 BIRD DETECTION SETUP COMPLETED! 🎉${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo -e "${CYAN}📁 PROJECT STRUCTURE:${NC}"
    echo -e "   📂 data/              - CUB-200-2011 dataset location"
    echo -e "   📂 scripts/           - Dataset download and utilities"
    echo -e "   📂 configs/           - YAML configurations (200 classes)"
    echo -e "   📂 models/            - Pre-trained YOLO models"
    echo -e "   📂 runs/              - Training results and logs"
    echo -e "   🐍 bird_detection_env/ - Virtual environment"
    echo ""
    echo -e "${CYAN}🚀 TO START WORKING:${NC}"
    echo -e "${YELLOW}   ./activate_env.sh${NC}"
    echo ""
    echo -e "${CYAN}📋 NEXT STEPS:${NC}"
    echo -e "   ${GREEN}1.${NC} Activate environment:  ${YELLOW}./activate_env.sh${NC}"
    echo -e "   ${GREEN}2.${NC} Test setup:           ${YELLOW}python quick_start.py${NC}"
    echo -e "   ${GREEN}3.${NC} Setup WandB:          ${YELLOW}wandb login${NC}"
    echo -e "   ${GREEN}4.${NC} Download CUB dataset: ${YELLOW}python scripts/download_cub_dataset.py${NC}"
    echo -e "   ${GREEN}5.${NC} Train model:          ${YELLOW}python train.py --data configs/cub_birds.yaml --epochs 100 --batch-size 32${NC}"
    echo -e "   ${GREEN}6.${NC} Evaluate model:       ${YELLOW}python test.py --model-path runs/train/exp/weights/best.pt${NC}"
    echo -e "   ${GREEN}7.${NC} Run inference:        ${YELLOW}python inference.py --source 0 --model runs/train/exp/weights/best.pt${NC}"
    echo ""
    echo -e "${CYAN}📊 DATASET INFO:${NC}"
    echo -e "   🐦 CUB-200-2011: 200 bird species"
    echo -e "   📸 Total images: 11,788"
    echo -e "   🏋️  Training: ~4,816 images"
    echo -e "   ✅ Validation: ~1,206 images"
    echo -e "   🧪 Test: ~5,766 images"
    echo ""
    echo -e "${CYAN}💡 HARDWARE OPTIMIZATION:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        echo -e "   🔥 GPU: $GPU_NAME"
        echo -e "   💾 VRAM: ${GPU_MEMORY}MB"
        
        if [ "$GPU_MEMORY" -gt 15000 ]; then
            echo -e "   💪 Recommended batch size: 64-128"
        elif [ "$GPU_MEMORY" -gt 11000 ]; then
            echo -e "   💪 Recommended batch size: 32-64"
        elif [ "$GPU_MEMORY" -gt 7000 ]; then
            echo -e "   💪 Recommended batch size: 16-32"
        else
            echo -e "   💪 Recommended batch size: 8-16"
        fi
    else
        echo -e "   🖥️  CPU-only setup detected"
        echo -e "   💪 Recommended batch size: 4-8"
    fi
    echo ""
    echo -e "${CYAN}🔗 USEFUL LINKS:${NC}"
    echo -e "   📚 YOLOv11 Docs: https://docs.ultralytics.com/"
    echo -e "   📊 WandB Dashboard: https://wandb.ai/"
    echo -e "   🐦 CUB Dataset: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html"
    echo -e "   📖 Paper: https://authors.library.caltech.edu/27452/"
    echo ""
    echo -e "${GREEN}Happy Bird Species Detection! 🐦✨${NC}"
    echo -e "${YELLOW}Perfect for university research projects and academic papers!${NC}"
}

# Main execution
main() {
    # Check prerequisites
    check_python
    check_pip
    check_venv
    
    # Setup project
    create_project_structure
    create_virtual_environment
    
    # Install dependencies
    install_pytorch
    install_dependencies
    
    # Configuration and scripts
    create_config_files
    create_download_script
    
    # Test and finalize
    test_installation
    create_quick_start
    
    # Deactivate virtual environment
    deactivate
    
    print_completion
}

# Run main function
main

echo -e "${BLUE}Setup script completed. CUB-200-2011 project is ready!${NC}"
#!/usr/bin/env python3
"""
System validation script - checks if all requirements are met
"""
import sys
import subprocess
from pathlib import Path


def print_section(title):
    """Print section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_imports():
    """Check if key modules can be imported"""
    print("\nChecking Python packages...")
    
    required_modules = {
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'yaml': 'PyYAML',
        'psutil': 'psutil',
    }
    
    optional_modules = {
        'torch': 'PyTorch',
        'mediapipe': 'MediaPipe',
        'GPUtil': 'GPUtil',
        'SpoutGL': 'SpoutGL (Windows only)',
        'NDIlib': 'NDI (ndi-python)',
    }
    
    all_good = True
    
    # Check required
    for module, name in required_modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - REQUIRED")
            all_good = False
    
    # Check optional
    print("\nOptional packages:")
    for module, name in optional_modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"○ {name} - Optional")
    
    return all_good


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA support...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✓ CUDA is available")
            print(f"  Devices: {device_count}")
            
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1024**3
                print(f"  GPU {i}: {name} ({vram_gb:.1f}GB)")
            
            return True
        else:
            print("○ CUDA not available (will use CPU)")
            print("  Install PyTorch with CUDA for GPU acceleration")
            return True  # Not a failure, just a warning
            
    except ImportError:
        print("○ PyTorch not installed")
        print("  Install with: pip install torch torchvision")
        return True


def check_project_structure():
    """Check if project structure is correct"""
    print("\nChecking project structure...")
    
    required_paths = [
        'src',
        'src/animation',
        'src/capture',
        'src/tracking',
        'src/output',
        'src/pipeline',
        'src/utils',
        'models',
        'scripts',
        'docs',
        'tests',
        'config.yaml',
        'main.py',
        'requirements.txt',
    ]
    
    all_good = True
    base_path = Path(__file__).parent.parent
    
    for path in required_paths:
        full_path = base_path / path
        if full_path.exists():
            print(f"✓ {path}")
        else:
            print(f"✗ {path} - MISSING")
            all_good = False
    
    return all_good


def check_config():
    """Check if configuration loads"""
    print("\nChecking configuration...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.utils import Config
        
        config = Config()
        print("✓ Configuration loaded successfully")
        
        # Check key settings
        capture_fps = config.get('capture.fps')
        target_fps = config.get('performance.target_fps')
        model = config.get('animation.model')
        
        print(f"  Capture FPS: {capture_fps}")
        print(f"  Target FPS: {target_fps}")
        print(f"  Animation Model: {model}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def check_webcam():
    """Check if webcam is accessible"""
    print("\nChecking webcam...")
    
    try:
        import cv2
        
        # Try to open default camera
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                h, w = frame.shape[:2]
                print(f"✓ Webcam is accessible")
                print(f"  Resolution: {w}x{h}")
                return True
            else:
                print("○ Webcam opened but couldn't read frame")
                return True
        else:
            print("○ No webcam detected (optional for some use cases)")
            return True
            
    except Exception as e:
        print(f"○ Webcam check failed: {e}")
        return True


def run_basic_test():
    """Run basic import test"""
    print("\nRunning basic import test...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src.utils import Config, setup_logger
        from src.capture import WebcamCapture
        from src.tracking import MediaPipeTracker
        from src.animation import BaseAnimationModel, LivePortraitModel
        from src.output import DisplayOutput
        from src.pipeline import AnimatorPipeline
        
        print("✓ All core modules imported successfully")
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


def main():
    """Main validation"""
    print("\n" + "="*60)
    print(" Stream Motion Animator - System Validation")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'Python Packages': check_imports(),
        'CUDA Support': check_cuda(),
        'Project Structure': check_project_structure(),
        'Configuration': check_config(),
        'Webcam': check_webcam(),
        'Import Test': run_basic_test(),
    }
    
    # Summary
    print_section("Validation Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check:.<40} {status}")
    
    print("\n" + "="*60)
    print(f" Result: {passed}/{total} checks passed")
    print("="*60)
    
    if passed == total:
        print("\n✓ System is ready!")
        print("\nNext steps:")
        print("  1. Download AI models: python scripts/download_models.py")
        print("  2. Prepare portrait image")
        print("  3. Run: python main.py --image your_portrait.jpg")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the output above.")
        print("\nSee docs/SETUP.md for detailed setup instructions.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

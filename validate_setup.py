"""
Setup Validation Script
=======================

This script checks if all dependencies and hardware are properly configured.
Run this before running the main application to identify issues early.
"""

import sys
import platform
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"[X] Python version {version.major}.{version.minor} is too old. Need 3.8+")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_imports():
    """Check if all required modules can be imported."""
    modules = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('sounddevice', 'sounddevice'),
        ('ultralytics', 'ultralytics'),
        ('usb', 'pyusb'),
    ]
    
    all_ok = True
    for module_name, package_name in modules:
        try:
            __import__(module_name)
            print(f"[OK] {package_name}")
        except ImportError:
            print(f"[X] {package_name} not installed. Run: pip install {package_name}")
            all_ok = False
    
    return all_ok

def check_audio_doa():
    """Check if audio_doa module can be imported."""
    usb_mic_dir = Path(__file__).parent / 'usb_4_mic_array'
    if not usb_mic_dir.exists():
        print(f"[X] USB mic array directory not found: {usb_mic_dir}")
        return False
    
    sys.path.insert(0, str(usb_mic_dir))
    try:
        import audio_doa
        print("[OK] audio_doa module")
        return True
    except ImportError as e:
        print(f"[X] Cannot import audio_doa: {e}")
        return False

def check_yolo_model():
    """Check if YOLO model exists."""
    # Check for YOLOv10n first (default), then fallback to YOLOv8n
    model_paths = [Path('yolov10n.pt'), Path('yolov8n.pt'), Path('yolov10n.engine')]
    for model_path in model_paths:
        if model_path.exists():
            print(f"[OK] YOLO model found: {model_path}")
            return True
    print(f"[!] YOLO model not found (checked: yolov10n.pt, yolov8n.pt, yolov10n.engine)")
    print("    (Will be auto-downloaded on first run)")
    return True  # Not critical, will auto-download

def check_camera():
    """Check if camera is available."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            print("[OK] Camera available")
            return True
        else:
            print("[!] Camera not accessible (may need to connect or check permissions)")
            return False
    except Exception as e:
        print(f"[!] Camera check failed: {e}")
        return False

def check_microphone():
    """Check if microphone devices are available."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
        if input_devices:
            print(f"[OK] Microphone devices found: {len(input_devices)}")
            for idx, dev in enumerate(input_devices):
                if 'ReSpeaker' in dev['name']:
                    print(f"    [OK] ReSpeaker found: {dev['name']} (index {idx})")
                    return True
            print("[!] ReSpeaker not found. Available devices:")
            for idx, dev in enumerate(input_devices):
                print(f"    - [{idx}] {dev['name']}")
            return False
        else:
            print("[X] No microphone devices found")
            return False
    except Exception as e:
        print(f"[!] Microphone check failed: {e}")
        return False

def check_respeaker_usb():
    """Check if ReSpeaker USB device is accessible."""
    try:
        import usb.core
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if dev is not None:
            print("[OK] ReSpeaker USB device found")
            return True
        else:
            print("[!] ReSpeaker USB device not found (may need USB driver on Windows)")
            print("    See usb_4_mic_array/README.md for driver installation")
            return False
    except Exception as e:
        print(f"[!] USB check failed: {e}")
        return False

def check_platform():
    """Check platform-specific requirements."""
    system = platform.system()
    print(f"\nPlatform: {system} {platform.release()}")
    
    if system == 'Windows':
        print("[!] Windows detected: Ensure libusb-win32 driver is installed via Zadig")
        print("    See usb_4_mic_array/README.md for details")
    elif system == 'Linux':
        print("[i] Linux detected: May need sudo or udev rules for USB access")
    elif system == 'Darwin':
        print("[i] macOS detected: Should work without additional setup")

def main():
    """Run all checks."""
    print("=" * 60)
    print("Audio-Visual Fusion Setup Validation")
    print("=" * 60)
    print()
    
    checks = []
    
    print("1. Python Version:")
    checks.append(check_python_version())
    print()
    
    print("2. Python Dependencies:")
    checks.append(check_imports())
    print()
    
    print("3. Audio DOA Module:")
    checks.append(check_audio_doa())
    print()
    
    print("4. YOLO Model:")
    checks.append(check_yolo_model())
    print()
    
    print("5. Camera:")
    checks.append(check_camera())
    print()
    
    print("6. Microphone:")
    checks.append(check_microphone())
    print()
    
    print("7. ReSpeaker USB:")
    checks.append(check_respeaker_usb())
    print()
    
    check_platform()
    print()
    
    # Summary
    print("=" * 60)
    critical_checks = checks[:4]  # Python, imports, audio_doa, yolo
    if all(critical_checks):
        print("[OK] Basic setup looks good!")
        print("[!] Some hardware checks may need attention, but software is ready")
    else:
        print("[X] Critical issues found. Please fix them before running the application.")
    print("=" * 60)

if __name__ == "__main__":
    main()


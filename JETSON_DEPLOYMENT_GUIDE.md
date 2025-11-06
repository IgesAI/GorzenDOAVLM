# Complete Jetson Deployment Guide: Audio-Visual Fusion with YOLOv10

## Overview

This guide provides step-by-step instructions to deploy the Audio-Visual Fusion system on NVIDIA Jetson devices (Nano, Xavier, Orin, etc.) with YOLOv10 optimized for edge AI.

---

## Prerequisites

### Hardware Requirements
- **NVIDIA Jetson device** (Nano, Xavier NX, AGX Xavier, Orin Nano, AGX Orin)
- **MicroSD card** (64GB+ recommended) or **NVMe SSD** (for Orin)
- **Power supply** (adequate for your Jetson model)
- **ReSpeaker USB Mic Array v2.1**
- **USB Camera** (compatible with Jetson)
- **USB-C cable** (for initial setup)
- **Keyboard, mouse, monitor** (or SSH access)

### Software Requirements
- **JetPack 5.x** (recommended) or **JetPack 4.6+**
- **Internet connection** (for package installation)

---

## Part 1: Jetson Initial Setup

### Step 1.1: Flash JetPack OS

1. **Download JetPack SDK**
   - Visit: https://developer.nvidia.com/embedded/jetpack
   - Download JetPack 5.1.2 (latest) or compatible version
   - Extract and run SDK Manager

2. **Flash OS Image**
   - Connect Jetson via USB-C (Recovery Mode)
   - Follow SDK Manager wizard
   - Select: **JetPack 5.x** (or 4.6+)
   - Components to install:
     - ✅ CUDA
     - ✅ cuDNN
     - ✅ TensorRT
     - ✅ VisionWorks
     - ✅ OpenCV (with CUDA support)

3. **Complete Initial Setup**
   - Boot Jetson
   - Complete Ubuntu setup wizard
   - Set username/password
   - Configure network (WiFi or Ethernet)

### Step 1.2: System Updates

```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

### Step 1.3: Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check GPU
nvidia-smi

# Set power mode (for maximum performance)
sudo nvpmodel -m 0  # MAXN mode (if available)
sudo jetson_clocks  # Set clocks to maximum
```

**Note**: Jetson Nano default mode is 5W. For better performance:
```bash
sudo nvpmodel -m 0  # 10W mode (Nano)
```

---

## Part 2: Python Environment Setup

### Step 2.1: Install System Dependencies

```bash
# Update package list
sudo apt update

# Essential build tools
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-setuptools \
    git \
    cmake \
    build-essential \
    libatlas-base-dev \
    libopencv-dev \
    python3-opencv \
    libusb-1.0-0-dev \
    libffi-dev \
    libssl-dev

# Audio libraries
sudo apt install -y \
    portaudio19-dev \
    python3-pyaudio \
    alsa-utils

# USB permissions (for ReSpeaker)
sudo usermod -a -G dialout $USER
sudo usermod -a -G audio $USER
```

### Step 2.2: Create Python Virtual Environment

```bash
# Navigate to project directory
cd ~
mkdir -p projects
cd projects

# Clone or transfer your project
# If using git:
# git clone <your-repo-url> sound_yolo_project
# Or transfer via SCP/USB

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 2.3: Install PyTorch (Jetson Optimized)

**Important**: Use NVIDIA's pre-built PyTorch wheels for Jetson (not PyPI).

```bash
# For JetPack 5.x (Python 3.10)
wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl

pip install torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl

# Install torchvision
pip install torchvision --no-deps

# Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Alternative**: If above doesn't work, use NVIDIA's installation script:
```bash
# Download and run NVIDIA's PyTorch installer
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.13.0-cp38-cp38-linux_aarch64.whl
pip install torch-1.13.0-cp38-cp38-linux_aarch64.whl
```

### Step 2.4: Install Project Dependencies

```bash
# Make sure you're in virtual environment
source venv/bin/activate

# Navigate to project
cd sound_yolo_project

# Install core dependencies
pip install numpy opencv-python-headless

# Install Ultralytics (YOLOv10)
pip install ultralytics

# Install audio dependencies
pip install sounddevice pyaudio

# Install USB dependencies
pip install pyusb==1.0.2 click==7.0

# Install additional utilities
pip install pathlib typing-extensions
```

**Note**: If `opencv-python-headless` fails, use system OpenCV:
```bash
# Use system OpenCV (already installed)
# Just ensure Python can import cv2
python3 -c "import cv2; print(cv2.__version__)"
```

---

## Part 3: Project Setup

### Step 3.1: Transfer Project Files

**Option A: Using Git**
```bash
cd ~/projects
git clone <your-repo-url> sound_yolo_project
cd sound_yolo_project
```

**Option B: Using SCP (from your PC)**
```bash
# On your PC:
scp -r sound_yolo_project/ jetson@<jetson-ip>:~/projects/

# On Jetson:
cd ~/projects/sound_yolo_project
```

**Option C: Using USB Drive**
```bash
# Mount USB drive
sudo mkdir /mnt/usb
sudo mount /dev/sda1 /mnt/usb  # Adjust device name
cp -r /mnt/usb/sound_yolo_project ~/projects/
```

### Step 3.2: Verify Project Structure

```bash
cd ~/projects/sound_yolo_project
ls -la

# Should see:
# - audio_vision_fusion.py
# - requirements.txt
# - usb_4_mic_array/
# - README.md
# - etc.
```

### Step 3.3: Install Project Dependencies

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Install from requirements.txt
pip install -r requirements.txt

# Verify key imports
python3 -c "
from ultralytics import YOLO
import cv2
import numpy as np
import sounddevice as sd
print('All imports successful!')
"
```

### Step 3.4: Configure USB Permissions for ReSpeaker

```bash
# Create udev rules for ReSpeaker
sudo nano /etc/udev/rules.d/99-respeaker.rules
```

Add this content:
```
# ReSpeaker USB Mic Array
SUBSYSTEM=="usb", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", MODE="0666", GROUP="plugdev"
```

Save and reload:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger

# Reconnect ReSpeaker USB device
```

---

## Part 4: YOLOv10 Model Setup

### Step 4.1: Download YOLOv10 Model

```bash
cd ~/projects/sound_yolo_project

# Activate virtual environment
source venv/bin/activate

# Model will auto-download on first run, or download manually:
python3 << EOF
from ultralytics import YOLO
model = YOLO('yolov10n.pt')  # Will download if not present
print(f"Model loaded: {len(model.names)} classes")
EOF
```

### Step 4.2: Optimize Model for Jetson (TensorRT)

**Important**: TensorRT optimization significantly improves inference speed on Jetson.

```bash
# Export to TensorRT format
python3 << EOF
from ultralytics import YOLO

# Load model
model = YOLO('yolov10n.pt')

# Export to TensorRT (this will take a few minutes)
model.export(
    format='engine',  # TensorRT engine
    device=0,         # GPU
    half=True,        # FP16 precision
    workspace=4       # 4GB workspace
)

print("TensorRT engine created successfully!")
EOF
```

This creates `yolov10n.engine` - use this instead of `.pt` for faster inference.

**Update code to use TensorRT**:
```python
# In audio_vision_fusion.py, change:
YOLO_MODEL_PATH = 'yolov10n.engine'  # Use TensorRT engine
```

---

## Part 5: Hardware Configuration

### Step 5.1: Test Camera

```bash
# List video devices
ls -la /dev/video*

# Test camera with v4l2
v4l2-ctl --list-devices

# Test with OpenCV
python3 << EOF
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Camera opened successfully!")
    ret, frame = cap.read()
    if ret:
        print(f"Frame size: {frame.shape}")
    cap.release()
else:
    print("Failed to open camera")
EOF
```

### Step 5.2: Test ReSpeaker Microphone

```bash
# List audio devices
python3 << EOF
import sounddevice as sd
devices = sd.query_devices()
for i, dev in enumerate(devices):
    if dev['max_input_channels'] > 0:
        print(f"[{i}] {dev['name']}")
EOF

# Test USB device
python3 << EOF
import usb.core
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
if dev is not None:
    print("ReSpeaker USB device found!")
    print(f"Manufacturer: {dev.manufacturer}")
    print(f"Product: {dev.product}")
else:
    print("ReSpeaker not found - check USB connection")
EOF
```

### Step 5.3: Test Audio DOA Module

```bash
cd ~/projects/sound_yolo_project
source venv/bin/activate

python3 << EOF
import sys
from pathlib import Path
sys.path.insert(0, str(Path('usb_4_mic_array')))

try:
    import audio_doa
    doa = audio_doa.ReSpeakerDOA()
    print(f"DOA angle: {doa.doa()}°")
    print(f"Voice active: {doa.voice_active()}")
    print("ReSpeaker DOA module working!")
except Exception as e:
    print(f"Error: {e}")
EOF
```

---

## Part 6: Run Application

### Step 6.1: Validate Setup

```bash
cd ~/projects/sound_yolo_project
source venv/bin/activate

# Run validation script
python3 validate_setup.py
```

### Step 6.2: Configure Settings

Edit `audio_vision_fusion.py` or use environment variables:

```bash
# Set optimal settings for Jetson
export CAMERA_INDEX=0
export FRAME_WIDTH=640        # Lower resolution for Jetson
export FRAME_HEIGHT=480
export YOLO_MODEL_PATH=yolov10n.engine  # Use TensorRT
export YOLO_CONFIDENCE_THRESHOLD=0.3
```

Or edit `audio_vision_fusion.py`:
```python
FRAME_WIDTH = 640   # Optimized for Jetson
FRAME_HEIGHT = 480
YOLO_MODEL_PATH = 'yolov10n.engine'  # TensorRT engine
```

### Step 6.3: Set Jetson Performance Mode

```bash
# Set to maximum performance (before running)
sudo nvpmodel -m 0       # Maximum power mode
sudo jetson_clocks        # Maximum clocks

# Check status
sudo nvpmodel -q
```

### Step 6.4: Run Application

```bash
cd ~/projects/sound_yolo_project
source venv/bin/activate

# Run with logging
python3 audio_vision_fusion.py

# Or run in background
nohup python3 audio_vision_fusion.py > output.log 2>&1 &
```

### Step 6.5: Monitor Performance

```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi

# Monitor CPU/GPU temperature
watch -n 1 cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Check logs
tail -f audio_vision_fusion.log
```

---

## Part 7: Optimization for Jetson

### Step 7.1: Enable TensorRT Inference

The code already supports TensorRT. After exporting the engine:

```python
# In audio_vision_fusion.py, the model will automatically use TensorRT
# if a .engine file is provided
```

### Step 7.2: Reduce Resolution for Better Performance

```python
# In audio_vision_fusion.py
FRAME_WIDTH = 640   # Instead of 1280
FRAME_HEIGHT = 480  # Instead of 720
```

### Step 7.3: Use FP16 Precision

TensorRT engine already uses FP16. If using `.pt` model:

```python
# In initialize_yolo method, add:
results = self.model(frame, conf=YOLO_CONFIDENCE_THRESHOLD, 
                    verbose=False, half=True)  # FP16
```

### Step 7.4: Frame Skipping (Optional)

For very resource-constrained scenarios:

```python
# In run() method, add frame skipping
frame_count = 0
while True:
    ret, frame = self.cap.read()
    frame_count += 1
    
    # Process every 2nd frame
    if frame_count % 2 == 0:
        results = self.model(frame, ...)
```

---

## Part 8: Troubleshooting

### Issue: CUDA Out of Memory

**Solution**:
```bash
# Reduce batch size (already 1 in code)
# Reduce resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Use TensorRT engine (smaller memory footprint)
```

### Issue: Low FPS

**Solutions**:
1. Use TensorRT engine instead of `.pt`
2. Reduce resolution (640x480)
3. Enable jetson_clocks
4. Check power mode (nvpmodel -m 0)

### Issue: ReSpeaker Not Detected

**Solutions**:
```bash
# Check USB connection
lsusb | grep 2886

# Check permissions
groups  # Should include 'plugdev' and 'dialout'

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Issue: Camera Not Working

**Solutions**:
```bash
# Check device
ls -la /dev/video*

# Test with v4l2
v4l2-ctl --device=/dev/video0 --list-formats

# Try different index
export CAMERA_INDEX=1  # or 2, etc.
```

### Issue: Import Errors

**Solutions**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python path
python3 -c "import sys; print(sys.path)"
```

---

## Part 9: Performance Benchmarks

### Expected Performance (Jetson Nano)

| Configuration | FPS | GPU Memory | Notes |
|---------------|-----|------------|-------|
| YOLOv10n PT (1280x720) | 5-10 | ~1.5GB | CPU bottleneck |
| YOLOv10n PT (640x480) | 15-25 | ~800MB | Balanced |
| YOLOv10n Engine (640x480) | 30-50 | ~600MB | **Recommended** |
| YOLOv10n Engine (480x360) | 50-70 | ~400MB | Maximum speed |

### Expected Performance (Jetson Xavier/Orin)

| Configuration | FPS | GPU Memory |
|---------------|-----|------------|
| YOLOv10n Engine (1280x720) | 50-80 | ~1GB |
| YOLOv10n Engine (640x480) | 80-120 | ~600MB |

---

## Part 10: Auto-Start on Boot (Optional)

### Create Systemd Service

```bash
sudo nano /etc/systemd/system/audio-vision-fusion.service
```

Add content:
```ini
[Unit]
Description=Audio-Visual Fusion with YOLOv10
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/projects/sound_yolo_project
Environment="PATH=/home/YOUR_USERNAME/projects/sound_yolo_project/venv/bin"
ExecStart=/home/YOUR_USERNAME/projects/sound_yolo_project/venv/bin/python3 audio_vision_fusion.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable audio-vision-fusion.service
sudo systemctl start audio-vision-fusion.service

# Check status
sudo systemctl status audio-vision-fusion.service
```

---

## Quick Start Checklist

- [ ] JetPack flashed and updated
- [ ] CUDA verified (`nvcc --version`)
- [ ] Python virtual environment created
- [ ] PyTorch installed (Jetson-optimized)
- [ ] Project dependencies installed
- [ ] ReSpeaker USB permissions configured
- [ ] Camera tested
- [ ] YOLOv10n model downloaded
- [ ] TensorRT engine exported (optional but recommended)
- [ ] Validation script passes
- [ ] Application runs successfully

---

## Summary

Your project is now configured for Jetson with:
- ✅ YOLOv10n (latest, fastest model)
- ✅ TensorRT optimization ready
- ✅ Jetson-specific optimizations
- ✅ Complete deployment guide

**Key Differences from PC**:
1. Use Jetson-optimized PyTorch wheels (not PyPI)
2. Lower resolution (640x480 recommended)
3. Use TensorRT engine for best performance
4. Set jetson_clocks for maximum performance
5. Monitor GPU memory and temperature

**Performance Tips**:
- Use TensorRT `.engine` format (2-3x faster)
- Keep resolution at 640x480 or lower
- Enable maximum power mode
- Monitor GPU temperature

---

## Additional Resources

- **NVIDIA Jetson Documentation**: https://docs.nvidia.com/jetson/
- **Ultralytics YOLOv10**: https://github.com/ultralytics/ultralytics
- **JetPack Downloads**: https://developer.nvidia.com/embedded/jetpack
- **TensorRT Optimization**: https://developer.nvidia.com/tensorrt

---

**Ready to deploy!** Follow the steps above in order, and you'll have a fully optimized Audio-Visual Fusion system running on your Jetson.


# GorzenDOAVLM - Direction of Arrival Vision-Language Model

**Gorzen Engineering Solutions**  
Audio-Visual Fusion System for Intelligent Object Tracking

## Overview

GorzenDOAVLM is an advanced audio-visual fusion system that combines:
- **YOLOv10n Object Detection** with NVIDIA CUDA acceleration
- **ReSpeaker 4-Mic Array** for Direction of Arrival (DOA) estimation
- **IoU-based Object Tracking** with persistent lock-on capability
- **Progressive Confidence Visualization** for intuitive feedback

## Key Features

### 🎯 Intelligent Tracking
- **IoU Matching**: Tracks objects across frames using Intersection over Union (>0.5 threshold)
- **Persistent Lock-On**: Maintains focus on sound-emitting objects
- **Automatic Reset**: Releases lock when objects leave the frame

### 🎤 Advanced Audio Processing
- **DOA Detection**: Real-time direction estimation using ReSpeaker microphone array
- **Angular Validation**: ±20° tolerance for directional accuracy
- **Debounced Detection**: 0.3s minimum gap prevents noise spam

### 📊 Visual Feedback
- **Progressive Loading Bar**: Shows confidence building from 0-100%
- **Color Gradient**: Green → Orange → Red as confidence increases
- **Lock Indicator**: Visual "🔒 LOCKED" status on tracked objects

### ⚡ Performance Optimized
- **CUDA Acceleration**: Runs on NVIDIA Jetson (tested on Orin)
- **15-30 FPS**: Real-time processing on edge devices
- **TensorRT Support**: Export models for maximum performance

## Hardware Requirements

- **Computing**: NVIDIA Jetson Orin (or compatible CUDA device)
- **Audio**: ReSpeaker USB Mic Array v2.1 (4-mic)
- **Camera**: USB webcam (720p or higher recommended)
- **OS**: Ubuntu 20.04+ with JetPack (for Jetson)

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/IgesAI/GorzenDOAVLM.git
cd GorzenDOAVLM
```

### 2. Set Up Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install CUDA PyTorch (Jetson Only)
```bash
./install_pytorch_jetson.sh
```

### 4. Verify Setup
```bash
python3 validate_setup.py
```

## Quick Start

### Run the System
```bash
python3 audio_vision_fusion.py
```

### Configuration (Optional)
```bash
# Adjust sensitivity
export MAX_SOUND_HITS=5              # Hits needed for 100% confidence
export SOUND_HIT_DEBOUNCE=0.3        # Min seconds between hits
export DOA_LOCK_TOLERANCE=20         # Max angle difference (degrees)
export TRACKING_IOU_THRESHOLD=0.5    # IoU threshold for tracking
```

## How It Works

1. **Detection**: YOLO detects all objects in camera view
2. **Sound Trigger**: ReSpeaker detects sound direction (DOA)
3. **Object Matching**: System finds object closest to sound direction
4. **Lock Creation**: Begins tracking with 20% initial confidence
5. **Confidence Building**: Each subsequent sound adds 20% (debounced)
6. **Full Lock**: At 100%, object is fully locked with red bounding box
7. **Persistent Tracking**: IoU matching maintains lock across frames
8. **Auto Reset**: Lock releases when object leaves frame

## Research Validation

This implementation has been validated against **34 research papers** (2015-2025) covering:
- Audio-visual fusion architectures
- Object tracking methodologies  
- DOA estimation techniques
- Multi-modal synchronization

See `IMPLEMENTATION_AUDIT.md` for full research references.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_SOUND_HITS` | 5 | Number of sound detections for 100% confidence |
| `SOUND_HIT_DEBOUNCE` | 0.3s | Minimum time between sound hits |
| `TRACKING_IOU_THRESHOLD` | 0.5 | IoU threshold for object matching |
| `DOA_LOCK_TOLERANCE` | 20° | Max angle difference for directional lock |
| `SOUND_HISTORY_WINDOW` | 2.0s | Time window for sound history |
| `YOLO_CONFIDENCE_THRESHOLD` | 0.25 | YOLO detection confidence threshold |

## Performance Tuning

### Adjust Lock Speed
```bash
export SOUND_HIT_DEBOUNCE=0.1  # Fast lock (~0.5s)
export SOUND_HIT_DEBOUNCE=0.5  # Slow lock (~2.5s)
```

### Adjust Sensitivity
```bash
export MAX_SOUND_HITS=3        # Easier to lock (60% threshold)
export MAX_SOUND_HITS=10       # Harder to lock (requires more hits)
```

### Adjust Directional Tolerance
```bash
export DOA_LOCK_TOLERANCE=10   # Stricter (within 10°)
export DOA_LOCK_TOLERANCE=30   # More lenient (within 30°)
```

## Documentation

- **[README.md](README.md)**: Original project documentation
- **[JETSON_QUICK_START.md](JETSON_QUICK_START.md)**: Jetson setup guide
- **[IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md)**: Research validation
- **[TRACKING_ENHANCEMENT.md](TRACKING_ENHANCEMENT.md)**: Feature documentation
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)**: AI development guide

## Troubleshooting

### CUDA Not Available
```bash
# Verify PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall CUDA PyTorch (Jetson)
./install_pytorch_jetson.sh
```

### ReSpeaker Not Detected
```bash
# List audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Check USB connection
lsusb | grep -i respeaker
```

### Low FPS
```bash
# Export to TensorRT for better performance
python3 export_tensorrt.py
```

## Contributing

This project is maintained by **Gorzen Engineering Solutions**.

For issues or feature requests, please contact:
- **Email**: nathan.gorzen@gmail.com
- **Website**: GorzenEngineering.com
- **GitHub**: @dabsmalone

## License

See individual component licenses:
- ByteTrack: MIT License (see `bytetrack/LICENSE`)
- ReSpeaker: MIT License (see `usb_4_mic_array/LICENSE`)

## Acknowledgments

- YOLOv10: Ultralytics team
- ByteTrack: Yifu Zhang et al.
- ReSpeaker: Seeed Studio
- Research community for audio-visual fusion methodologies

---

**Gorzen Engineering Solutions** | Michigan, USA  
*Intelligent Systems for the Future*

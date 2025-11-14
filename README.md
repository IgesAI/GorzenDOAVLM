# Audio-Visual Fusion: YOLO + ReSpeaker DOA# Audio-Visual Fusion: YOLO + ReSpeaker DOA



A real-time audio-visual fusion system that combines YOLO object detection with ReSpeaker microphone array Direction of Arrival (DOA) to highlight objects based on sound direction.A real-time audio-visual fusion system that combines YOLO object detection with ReSpeaker microphone array Direction of Arrival (DOA) to highlight objects based on sound direction.



## 🎯 Overview## Overview



This project detects objects using YOLOv10 and highlights them based on sound direction from a ReSpeaker microphone array. Perfect for:This project detects objects in a camera feed using YOLOv10, and when sound is detected from a specific direction via the ReSpeaker USB Mic Array, it highlights the object closest to that sound direction. This enables applications like:

- Sound-triggered object tracking

- Voice-activated object detection  - **Sound-triggered object tracking**: Highlight objects that made a sound

- Audio-visual scene analysis- **Voice-activated object detection**: Find objects near a speaking person

- **Audio-visual scene analysis**: Correlate audio events with visual detections

## ✨ Features

## Features

- ✅ Real-time YOLO object detection with CUDA acceleration

- ✅ ReSpeaker USB Mic Array DOA integration- ✅ Real-time YOLO object detection with configurable confidence thresholds

- ✅ Persistent object tracking with confidence visualization- ✅ ReSpeaker USB Mic Array DOA (Direction of Arrival) integration

- ✅ **Performance monitoring** - FPS, latency, GPU metrics overlay- ✅ Voice Activity Detection (VAD) and speech detection

- ✅ **Spatial audio heatmap** - Visual sound direction overlay- ✅ Automatic object highlighting based on sound direction

- ✅ **Configuration hot-reload** - Update settings without restart (press `r`)- ✅ **Persistent object tracking** with confidence visualization

- ✅ Cross-platform support (Windows, Linux, macOS)- ✅ **Performance monitoring** - FPS, latency, GPU metrics overlay

- ✅ **Spatial audio heatmap** - Visual sound direction overlay

> 🆕 **v2.0**: See [ENHANCEMENTS.md](ENHANCEMENTS.md) for new features- ✅ **Configuration hot-reload** - Update settings without restart

- ✅ Cross-platform support (Windows, Linux, macOS)

## 📦 Requirements- ✅ CUDA acceleration on NVIDIA Jetson and desktop GPUs

- ✅ Comprehensive logging and error handling

### Hardware- ✅ Graceful shutdown and resource cleanup

- **ReSpeaker USB Mic Array v2.1** ([Seeed Studio](https://www.seeedstudio.com/ReSpeaker-Mic-Array-v2.0-p-3053.html))- ✅ Configurable via environment variables

- **USB Camera** (webcam)

- **Computer** with USB ports (Jetson Nano, desktop PC, etc.)> 🆕 **New in v2.0**: See [ENHANCEMENTS.md](ENHANCEMENTS.md) for details on performance monitoring, audio heatmap, and hot-reload features.



### Software## Requirements

- Python 3.8+

- CUDA-enabled GPU (optional, for acceleration)### Hardware



## 🚀 Quick Start- **ReSpeaker USB Mic Array v2.1** (or compatible)

  - Available at [Seeed Studio](https://www.seeedstudio.com/ReSpeaker-Mic-Array-v2.0-p-3053.html)

### 1. Install Dependencies- **USB Camera** (webcam or USB camera)

- **Computer** with USB ports

```bash

# Create virtual environment### Software

python3 -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate- Python 3.8 or higher

- ReSpeaker USB Mic Array firmware (included in `usb_4_mic_array/`)

# Install packages

pip install -r requirements.txt### Platform-Specific Requirements

```

#### Windows

### 2. Platform-Specific Setup- Install **libusb-win32** driver using [Zadig](https://zadig.akeo.ie/)

  - See `usb_4_mic_array/README.md` for detailed instructions

**Windows:**  - Install driver for both `SEEED DFU` and `SEEED Control` devices

```bash

# Install libusb driver using Zadig: https://zadig.akeo.ie/#### Linux

# See usb_4_mic_array/README.md for details- USB permissions may require:

```  - Running with `sudo`, OR

  - Adding udev rules for USB access

**Linux:**  - Adding user to appropriate groups (e.g., `plugdev`)

```bash

# May need USB permissions (run with sudo or add udev rules)#### macOS

```- Should work without additional setup



**NVIDIA Jetson:**## Installation

```bash

# Install PyTorch with CUDA support### 1. Clone Repository

./install_pytorch_jetson.sh

``````bash

git clone <repository-url>

### 3. Validate Setupcd sound_yolo_project

```

```bash

python3 validate_setup.py### 2. Create Virtual Environment (Recommended)

```

```bash

This checks:python -m venv venv

- Camera connection

- ReSpeaker microphone# On Windows:

- YOLO modelvenv\Scripts\activate

- CUDA availability

# On Linux/macOS:

### 4. Run Applicationsource venv/bin/activate

```

```bash

python3 audio_vision_fusion.py### 3. Install Dependencies

```

```bash

**Controls:**pip install -r requirements.txt

- Press `q` to quit```

- Press `r` to reload configuration

### 4. Download YOLO Model

## ⚙️ Configuration

The YOLOv10 model will be automatically downloaded on first run if not present. Alternatively, download manually:

### Quick Config

```bash

```bash# The model file should be named yolov10n.pt in the project root

# Copy example configuration# Ultralytics will download it automatically if missing

cp .env.example .env```



# Edit settings## Configuration

nano .env

```### Environment Variables



### Key SettingsYou can configure the system using environment variables:



```bash```bash

# Camera# Camera settings

CAMERA_INDEX=0export CAMERA_INDEX=0                    # Camera device index

FRAME_WIDTH=1280export FRAME_WIDTH=1280                  # Frame width in pixels

FRAME_HEIGHT=720export FRAME_HEIGHT=720                  # Frame height in pixels

CAMERA_FOV=78export CAMERA_FOV=78                     # Camera horizontal FOV in degrees



# Detection# YOLO settings

YOLO_CONFIDENCE_THRESHOLD=0.25export YOLO_MODEL_PATH=yolov10n.pt       # Path to YOLO model file (default)

SOUND_THRESHOLD_DB=-30export YOLO_CONFIDENCE_THRESHOLD=0.25    # Detection confidence threshold (0-1)



# Tracking# Audio settings

MAX_SOUND_HITS=5export SOUND_SAMPLE_RATE=16000           # Audio sample rate (Hz)

DOA_LOCK_TOLERANCE=20export SOUND_CHUNK_DURATION=0.1          # Audio chunk duration (seconds)

TRACKING_IOU_THRESHOLD=0.5export SOUND_THRESHOLD_DB=-30            # Sound detection threshold (dBFS)



# Features# DOA settings

ENABLE_PERFORMANCE_MONITOR=Trueexport DOA_OFFSET=0                      # DOA angle offset for calibration (degrees)

ENABLE_AUDIO_HEATMAP=Trueexport DOA_FLIP=True                     # Mirror left/right if orientation reversed

```export DOA_SNAP_TOLERANCE=10             # DOA angle change tolerance (degrees)

export TRIGGER_HOLD_TIME=0.5             # Trigger hold duration (seconds)

**Hot-reload:** Edit `.env` and press `r` during runtime!export MIC_DEVICE_NAME="ReSpeaker"      # Microphone device name substring

```

## 🎮 Usage

### Configuration File (Alternative)

### Basic Operation

You can also edit the configuration constants directly in `audio_vision_fusion.py`:

1. **Start application** - Objects detected in camera view

2. **Make sound** - Microphone detects direction```python

3. **Object highlighted** - Closest object to sound directionCAMERA_INDEX = 0

4. **Confidence builds** - Multiple detections = higher confidenceFRAME_WIDTH = 1280

5. **Lock maintained** - Object tracked until it leaves frameFRAME_HEIGHT = 720

# ... etc

### Visual Feedback```



- **Green box**: Detected object## Usage

- **Red box**: Sound-locked object

- **Loading bar**: Sound confidence (0-100%)### Basic Usage

- **[LOCKED]**: Object is being tracked

- **Yellow overlay**: Performance metrics (top-left)```bash

- **Color heatmap**: Audio direction visualizationpython audio_vision_fusion.py

```

## 🔧 Troubleshooting

### Calibration

### No Camera Feed

```bash1. **DOA Offset Calibration**: If your ReSpeaker and camera are not perfectly aligned:

# Test camera index   - Adjust `DOA_OFFSET` environment variable

CAMERA_INDEX=1 python3 audio_vision_fusion.py   - Positive values rotate clockwise, negative counter-clockwise

```

2. **DOA Flip**: If left/right are reversed:

### No Audio Detection   - Set `DOA_FLIP=True` (default) or `DOA_FLIP=False`

```bash

# Check microphone3. **Camera FOV**: Measure or look up your camera's horizontal field of view:

python3 -c "from usb_4_mic_array.audio_doa import ReSpeakerDOA; doa = ReSpeakerDOA(); print('✓ ReSpeaker OK')"   - Set `CAMERA_FOV` to match your camera's specifications

   - Common values: 60-90 degrees for webcams

# Lower threshold

SOUND_THRESHOLD_DB=-40 python3 audio_vision_fusion.py### Controls

```

- **Press 'q'**: Quit the application gracefully

### Low FPS- **Ctrl+C**: Also triggers graceful shutdown

```bash

# Disable features## Output

ENABLE_PERFORMANCE_MONITOR=False ENABLE_AUDIO_HEATMAP=False python3 audio_vision_fusion.py

### Visual Output

# Lower resolution

FRAME_WIDTH=640 FRAME_HEIGHT=480 python3 audio_vision_fusion.py- **Green boxes**: Regular object detections

```- **Red boxes**: Objects highlighted based on sound direction

- **Status overlay**: Shows current DOA angle and trigger status

### CUDA Not Available (Jetson)

```bash### Logging

# Install PyTorch with CUDA

./install_pytorch_jetson.shLogs are written to:

- **Console**: Real-time status and error messages

# Verify- **File**: `audio_vision_fusion.log` (detailed logs with timestamps)

python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

```## Troubleshooting



## 📊 Performance### ReSpeaker Not Found



### Expected Performance```

Error: ReSpeaker USB Mic Array not found

| Platform | FPS | Latency | Notes |```

|----------|-----|---------|-------|

| Jetson Nano | 18-25 | 45-55ms | With all features |**Solutions:**

| Desktop GPU | 25-30+ | 35-45ms | RTX 2060+ |- Check USB connection

| CPU Only | 8-12 | 100-150ms | Not recommended |- Verify device is powered (LEDs should be on)

- On Windows: Install libusb-win32 driver via Zadig

### Optimization- On Linux: Try running with `sudo` or check USB permissions



**For Speed:**### Camera Not Opening

```bash

FRAME_WIDTH=640```

FRAME_HEIGHT=480Error: Failed to open camera at index 0

ENABLE_AUDIO_HEATMAP=False```

```

**Solutions:**

**For Accuracy:**- Check camera connection

```bash- Try different `CAMERA_INDEX` values (0, 1, 2, etc.)

YOLO_CONFIDENCE_THRESHOLD=0.4- Verify camera permissions

DOA_LOCK_TOLERANCE=10- Check if another application is using the camera

TRACKING_IOU_THRESHOLD=0.6

```### Model Not Found



**For Visualization:**```

```bashError: YOLO model file not found

HEATMAP_INTENSITY=0.6```

HEATMAP_DECAY_RATE=0.98

```**Solutions:**

- Model will auto-download on first run

## 📚 Documentation- Or manually download from Ultralytics

- Check `YOLO_MODEL_PATH` environment variable

- **[ENHANCEMENTS.md](ENHANCEMENTS.md)** - Feature guide for v2.0

- **[.env.example](.env.example)** - Full configuration reference### Audio Not Working

- **[test_enhancements.py](test_enhancements.py)** - Test suite

- Check microphone permissions

## 🛠️ Development- Verify `MIC_DEVICE_NAME` matches your device name

- Run `python -c "import sounddevice; print(sounddevice.query_devices())"` to list devices

### Project Structure- Check audio sample rate compatibility



```### Poor DOA Accuracy

sound_yolo_project/

├── audio_vision_fusion.py       # Main application- Calibrate `DOA_OFFSET` for your setup

├── requirements.txt             # Python dependencies- Adjust `DOA_SNAP_TOLERANCE` to reduce jitter

├── .env.example                 # Configuration template- Ensure ReSpeaker is firmly mounted

├── validate_setup.py            # Hardware validation- Check for acoustic reflections/echoes

├── test_enhancements.py         # Feature tests

├── install_pytorch_jetson.sh    # Jetson CUDA setup## Architecture

├── usb_4_mic_array/            # ReSpeaker driver

│   ├── audio_doa.py            # DOA interface```

│   └── ...audio_vision_fusion.py    # Main application (class-based, with logging)

└── yolov10n.pt                  # YOLO model weights├── AudioVisionFusion     # Main class orchestrating all components

```│   ├── initialize_audio()     # Setup ReSpeaker DOA

│   ├── initialize_yolo()      # Load YOLO model

### Running Tests│   ├── initialize_camera()    # Setup camera capture

│   └── run()                  # Main processing loop

```bash│

# Validate hardware setupusb_4_mic_array/

python3 validate_setup.py├── audio_doa.py          # ReSpeaker DOA wrapper

├── tuning.py             # Seeed library for USB communication

# Test new features└── audio_energy.py       # Alternative audio trigger (unused but available)

python3 test_enhancements.py```

```

## Project Structure

### Key Components

```

**AudioVisionFusion** - Main classsound_yolo_project/

- `initialize_audio()` - Setup ReSpeaker├── audio_vision_fusion.py    # Main application

- `initialize_yolo()` - Load YOLO model├── requirements.txt          # Python dependencies

- `initialize_camera()` - Setup video capture├── README.md                 # This file

- `run()` - Main processing loop├── AUDIT_REPORT.md           # Detailed code audit

├── yolov10n.pt              # YOLO model (auto-downloaded)

**PerformanceMonitor** - Metrics tracking├── audio_vision_fusion.log   # Log file (generated)

- FPS calculation│

- Latency measurement├── usb_4_mic_array/          # ReSpeaker integration

- GPU/CPU monitoring│   ├── audio_doa.py         # DOA interface wrapper

│   ├── audio_energy.py      # Alternative audio trigger

**Tracking System** - Object persistence│   ├── tuning.py            # USB communication

- IoU-based tracking│   └── README.md             # ReSpeaker documentation

- Sound confidence building│

- Lock-on mechanism└── bytetrack/               # Multi-object tracking library (optional, unused)

    └── ...                  # BYTETrack codebase

## 🔬 Technical Details```



### Audio Processing## Performance



- Sample rate: 16kHz### Typical Performance

- Chunk duration: 0.1s

- VAD threshold: Configurable (default -30 dBFS)- **FPS**: 15-30 FPS (depends on camera and GPU)

- DOA resolution: 1° (360° coverage)- **Latency**: <100ms audio-to-visual feedback

- Trigger hold: 0.5s (configurable)- **CPU Usage**: Moderate (depends on YOLO model size)

- **Memory**: ~500MB-2GB (depends on model)

### Object Detection

### Optimization Tips

- Model: YOLOv10n (fastest, 6.2M params)

- Input: RGB frames- Use YOLOv10n model (default) or export to TensorRT engine for fastest inference

- Inference: CUDA-accelerated- Reduce frame resolution for better FPS

- Post-processing: NMS, confidence filtering- Use GPU acceleration if available (CUDA)

- Adjust `SOUND_CHUNK_DURATION` for different latency/accuracy tradeoffs

### Tracking Algorithm

## Development

- Method: IoU (Intersection over Union)

- Threshold: 0.5 (configurable)### Code Quality

- Persistence: Maintained until object leaves frame

- Confidence: Builds with repeated sound detections- Comprehensive error handling

- Logging throughout

## 🤝 Contributing- Type hints where applicable

- Resource cleanup on shutdown

Contributions welcome! Please:- Platform detection and warnings

1. Test on your hardware setup

2. Validate with `test_enhancements.py`### Testing

3. Document configuration changes

4. Submit PR with descriptionRun basic functionality test:



## 📝 License```bash

python -c "from audio_vision_fusion import AudioVisionFusion; print('Import successful')"

[Your License Here]```



## 🙏 Acknowledgments### Contributing



- **Ultralytics YOLO** - Object detection1. Follow existing code style

- **Seeed ReSpeaker** - Microphone array2. Add logging for new features

- **OpenCV** - Computer vision3. Include error handling

- **PyTorch** - Deep learning framework4. Update documentation



## 📞 Support## License



**Common Issues:**Check individual component licenses:

- USB permissions → See usb_4_mic_array/README.md- YOLO: AGPL-3.0 (Ultralytics)

- CUDA not found → Run `./install_pytorch_jetson.sh` (Jetson)- ReSpeaker: See `usb_4_mic_array/LICENSE`

- Low FPS → Lower resolution or disable features- BYTETrack: See `bytetrack/LICENSE`

- No sound detection → Adjust `SOUND_THRESHOLD_DB`

## Credits

**Getting Help:**

1. Check logs in `audio_vision_fusion.log`- **YOLO**: [Ultralytics](https://github.com/ultralytics/ultralytics)

2. Run `validate_setup.py` for diagnostics- **ReSpeaker**: [Seeed Studio](https://www.seeedstudio.com/)

3. Review [ENHANCEMENTS.md](ENHANCEMENTS.md) for features- **BYTETrack**: [ByteTrack](https://github.com/ifzhang/ByteTrack) (included but unused)

4. Open GitHub issue with logs

## Support

---

For issues:

**Made with ❤️ for audio-visual intelligence**1. Check `audio_vision_fusion.log` for detailed error messages

2. Verify hardware connections
3. Check platform-specific requirements
4. Review configuration settings

## Future Enhancements

Potential improvements:
- [ ] Multi-object tracking integration (BYTETrack)
- [ ] Configuration file (YAML/JSON)
- [ ] Web interface for remote monitoring
- [ ] Recording/saving functionality
- [ ] Multi-camera support
- [ ] Calibration wizard
- [ ] Performance metrics dashboard

## Changelog

### v2.0 (Current)
- Complete refactor with class-based architecture
- Comprehensive error handling and logging
- Graceful shutdown and resource cleanup
- Platform detection and warnings
- Configuration validation
- Improved documentation

### v1.0 (Original)
- Basic YOLO + ReSpeaker integration
- Simple audio-visual fusion
- Basic highlighting functionality



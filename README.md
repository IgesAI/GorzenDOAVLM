# Audio-Visual Fusion: YOLO + ReSpeaker DOA

A real-time audio-visual fusion system that combines YOLO object detection with ReSpeaker microphone array Direction of Arrival (DOA) to highlight objects based on sound direction.

## Overview

This project detects objects in a camera feed using YOLOv10, and when sound is detected from a specific direction via the ReSpeaker USB Mic Array, it highlights the object closest to that sound direction. This enables applications like:

- **Sound-triggered object tracking**: Highlight objects that made a sound
- **Voice-activated object detection**: Find objects near a speaking person
- **Audio-visual scene analysis**: Correlate audio events with visual detections

## Features

- ✅ Real-time YOLO object detection with configurable confidence thresholds
- ✅ ReSpeaker USB Mic Array DOA (Direction of Arrival) integration
- ✅ Voice Activity Detection (VAD) and speech detection
- ✅ Automatic object highlighting based on sound direction
- ✅ Cross-platform support (Windows, Linux, macOS)
- ✅ Comprehensive logging and error handling
- ✅ Graceful shutdown and resource cleanup
- ✅ Configurable via environment variables

## Requirements

### Hardware

- **ReSpeaker USB Mic Array v2.1** (or compatible)
  - Available at [Seeed Studio](https://www.seeedstudio.com/ReSpeaker-Mic-Array-v2.0-p-3053.html)
- **USB Camera** (webcam or USB camera)
- **Computer** with USB ports

### Software

- Python 3.8 or higher
- ReSpeaker USB Mic Array firmware (included in `usb_4_mic_array/`)

### Platform-Specific Requirements

#### Windows
- Install **libusb-win32** driver using [Zadig](https://zadig.akeo.ie/)
  - See `usb_4_mic_array/README.md` for detailed instructions
  - Install driver for both `SEEED DFU` and `SEEED Control` devices

#### Linux
- USB permissions may require:
  - Running with `sudo`, OR
  - Adding udev rules for USB access
  - Adding user to appropriate groups (e.g., `plugdev`)

#### macOS
- Should work without additional setup

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd sound_yolo_project
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLO Model

The YOLOv10 model will be automatically downloaded on first run if not present. Alternatively, download manually:

```bash
# The model file should be named yolov10n.pt in the project root
# Ultralytics will download it automatically if missing
```

## Configuration

### Environment Variables

You can configure the system using environment variables:

```bash
# Camera settings
export CAMERA_INDEX=0                    # Camera device index
export FRAME_WIDTH=1280                  # Frame width in pixels
export FRAME_HEIGHT=720                  # Frame height in pixels
export CAMERA_FOV=78                     # Camera horizontal FOV in degrees

# YOLO settings
export YOLO_MODEL_PATH=yolov10n.pt       # Path to YOLO model file (default)
export YOLO_CONFIDENCE_THRESHOLD=0.25    # Detection confidence threshold (0-1)

# Audio settings
export SOUND_SAMPLE_RATE=16000           # Audio sample rate (Hz)
export SOUND_CHUNK_DURATION=0.1          # Audio chunk duration (seconds)
export SOUND_THRESHOLD_DB=-30            # Sound detection threshold (dBFS)

# DOA settings
export DOA_OFFSET=0                      # DOA angle offset for calibration (degrees)
export DOA_FLIP=True                     # Mirror left/right if orientation reversed
export DOA_SNAP_TOLERANCE=10             # DOA angle change tolerance (degrees)
export TRIGGER_HOLD_TIME=0.5             # Trigger hold duration (seconds)
export MIC_DEVICE_NAME="ReSpeaker"      # Microphone device name substring
```

### Configuration File (Alternative)

You can also edit the configuration constants directly in `audio_vision_fusion.py`:

```python
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# ... etc
```

## Usage

### Basic Usage

```bash
python audio_vision_fusion.py
```

### Calibration

1. **DOA Offset Calibration**: If your ReSpeaker and camera are not perfectly aligned:
   - Adjust `DOA_OFFSET` environment variable
   - Positive values rotate clockwise, negative counter-clockwise

2. **DOA Flip**: If left/right are reversed:
   - Set `DOA_FLIP=True` (default) or `DOA_FLIP=False`

3. **Camera FOV**: Measure or look up your camera's horizontal field of view:
   - Set `CAMERA_FOV` to match your camera's specifications
   - Common values: 60-90 degrees for webcams

### Controls

- **Press 'q'**: Quit the application gracefully
- **Ctrl+C**: Also triggers graceful shutdown

## Output

### Visual Output

- **Green boxes**: Regular object detections
- **Red boxes**: Objects highlighted based on sound direction
- **Status overlay**: Shows current DOA angle and trigger status

### Logging

Logs are written to:
- **Console**: Real-time status and error messages
- **File**: `audio_vision_fusion.log` (detailed logs with timestamps)

## Troubleshooting

### ReSpeaker Not Found

```
Error: ReSpeaker USB Mic Array not found
```

**Solutions:**
- Check USB connection
- Verify device is powered (LEDs should be on)
- On Windows: Install libusb-win32 driver via Zadig
- On Linux: Try running with `sudo` or check USB permissions

### Camera Not Opening

```
Error: Failed to open camera at index 0
```

**Solutions:**
- Check camera connection
- Try different `CAMERA_INDEX` values (0, 1, 2, etc.)
- Verify camera permissions
- Check if another application is using the camera

### Model Not Found

```
Error: YOLO model file not found
```

**Solutions:**
- Model will auto-download on first run
- Or manually download from Ultralytics
- Check `YOLO_MODEL_PATH` environment variable

### Audio Not Working

- Check microphone permissions
- Verify `MIC_DEVICE_NAME` matches your device name
- Run `python -c "import sounddevice; print(sounddevice.query_devices())"` to list devices
- Check audio sample rate compatibility

### Poor DOA Accuracy

- Calibrate `DOA_OFFSET` for your setup
- Adjust `DOA_SNAP_TOLERANCE` to reduce jitter
- Ensure ReSpeaker is firmly mounted
- Check for acoustic reflections/echoes

## Architecture

```
audio_vision_fusion.py    # Main application (class-based, with logging)
├── AudioVisionFusion     # Main class orchestrating all components
│   ├── initialize_audio()     # Setup ReSpeaker DOA
│   ├── initialize_yolo()      # Load YOLO model
│   ├── initialize_camera()    # Setup camera capture
│   └── run()                  # Main processing loop
│
usb_4_mic_array/
├── audio_doa.py          # ReSpeaker DOA wrapper
├── tuning.py             # Seeed library for USB communication
└── audio_energy.py       # Alternative audio trigger (unused but available)
```

## Project Structure

```
sound_yolo_project/
├── audio_vision_fusion.py    # Main application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── AUDIT_REPORT.md           # Detailed code audit
├── yolov10n.pt              # YOLO model (auto-downloaded)
├── audio_vision_fusion.log   # Log file (generated)
│
├── usb_4_mic_array/          # ReSpeaker integration
│   ├── audio_doa.py         # DOA interface wrapper
│   ├── audio_energy.py      # Alternative audio trigger
│   ├── tuning.py            # USB communication
│   └── README.md             # ReSpeaker documentation
│
└── bytetrack/               # Multi-object tracking library (optional, unused)
    └── ...                  # BYTETrack codebase
```

## Performance

### Typical Performance

- **FPS**: 15-30 FPS (depends on camera and GPU)
- **Latency**: <100ms audio-to-visual feedback
- **CPU Usage**: Moderate (depends on YOLO model size)
- **Memory**: ~500MB-2GB (depends on model)

### Optimization Tips

- Use YOLOv10n model (default) or export to TensorRT engine for fastest inference
- Reduce frame resolution for better FPS
- Use GPU acceleration if available (CUDA)
- Adjust `SOUND_CHUNK_DURATION` for different latency/accuracy tradeoffs

## Development

### Code Quality

- Comprehensive error handling
- Logging throughout
- Type hints where applicable
- Resource cleanup on shutdown
- Platform detection and warnings

### Testing

Run basic functionality test:

```bash
python -c "from audio_vision_fusion import AudioVisionFusion; print('Import successful')"
```

### Contributing

1. Follow existing code style
2. Add logging for new features
3. Include error handling
4. Update documentation

## License

Check individual component licenses:
- YOLO: AGPL-3.0 (Ultralytics)
- ReSpeaker: See `usb_4_mic_array/LICENSE`
- BYTETrack: See `bytetrack/LICENSE`

## Credits

- **YOLO**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **ReSpeaker**: [Seeed Studio](https://www.seeedstudio.com/)
- **BYTETrack**: [ByteTrack](https://github.com/ifzhang/ByteTrack) (included but unused)

## Support

For issues:
1. Check `audio_vision_fusion.log` for detailed error messages
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


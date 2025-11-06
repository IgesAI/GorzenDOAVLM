# AI Agent Instructions for Sound-YOLO Project

## Project Overview

This is an audio-visual fusion system that combines YOLO object detection with ReSpeaker microphone array direction detection. The system highlights objects in camera feed based on sound direction, with **persistent tracking and confidence visualization**.

### Key Features
- Real-time YOLOv10n object detection with CUDA acceleration
- ReSpeaker 4-mic array for Direction of Arrival (DOA)
- **Object tracking with lock-on mechanism** - tracks objects across frames using IoU
- **Sound confidence loading bar** - visual indicator showing accumulated sound detections
- **Persistent tracking** - maintains lock until object leaves frame
- Configurable sensitivity and tracking thresholds

## Key Components & Architecture

1. **Core Components**:
   - `audio_vision_fusion.py`: Main integration module
   - `usb_4_mic_array/audio_doa.py`: ReSpeaker DOA interface
   - YOLOv10 model (auto-downloaded as `yolov10n.pt`)

2. **Data Flow**:
   ```
   [ReSpeaker Mic Array] → audio_doa.py → Direction of Arrival (θ)
                                      ↘
   [USB Camera] → YOLO Detection    →  audio_vision_fusion.py → Tracking + Visualization
                                      ↗
   [IoU Tracker] → Persistent lock on sound source
   ```

3. **Object Tracking**:
   - Uses IoU (Intersection over Union) to match objects between frames
   - Accumulates sound detections to build confidence (0-100%)
   - Locks onto objects with repeated sound detections
   - Resets when object leaves frame or IoU < threshold

## Development Workflows

### Environment Setup

1. **Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # (or venv\Scripts\activate on Windows)
   pip install -r requirements.txt
   ```

2. **CUDA/PyTorch Setup (CRITICAL for Jetson)**:
   - **Never** install PyTorch from PyPI on Jetson - it's CPU-only!
   - Run `./install_pytorch_jetson.sh` to install NVIDIA's CUDA-enabled wheel
   - Verifies: JetPack version, downloads correct wheel, installs torchvision
   - YOLO will automatically use `device='cuda:0'` when CUDA available
   - Check: `python3 -c "import torch; print(torch.cuda.is_available())"`

3. **Hardware Configuration**:
   - ReSpeaker USB Mic Array v2.1 must be connected
   - USB Camera must be connected
   - Windows requires libusb-win32 driver via Zadig
   - Linux may need USB permissions (sudo or udev rules)

### Testing & Validation

1. Use `validate_setup.py` to verify hardware connections and dependencies
2. Use `benchmark_yolo_models.py` for performance testing
3. Key test points in `audio_vision_fusion.py`:
   - Camera feed capture
   - YOLO detection accuracy
   - Audio DOA reliability
   - System latency

## Project Conventions

1. **CUDA Device Management**:
   - Device selection: `self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'`
   - Pass to YOLO: `model(frame, device=self.device)`
   - Don't use `torch.device()` object - use string format for ultralytics
   - Log GPU info at startup for debugging

2. **Logging**:
   - Use the configured logger from `audio_vision_fusion.py`
   - Include contextual information in log messages
   - Example: `logger.info(f"DOA angle detected: {angle}°")`

3. **Error Handling**:
   - Hardware errors are wrapped in descriptive RuntimeErrors
   - USB device errors include vendor/product IDs in messages
   - Graceful fallbacks when audio/video sources fail

4. **Configuration**:
   - Use environment variables for runtime configuration
   - Default values in code should match `README.md`
   - Document new settings in both code and README

## Integration Points

1. **ReSpeaker Interface**:
   - Use `ReSpeakerDOA` class from `audio_doa.py`
   - Check `voice_active()` before using `doa()`
   - Handle USB reconnection scenarios

2. **YOLO Integration**:
   - Model path: `yolov10n.pt` in project root
   - Use ultralytics.YOLO for inference
   - Pass `device=self.device` to all inference calls
   - Confidence threshold configurable via env var

## Common Tasks

1. **Adding New Detection Features**:
   - Extend `process_frame()` in `audio_vision_fusion.py`
   - Add configuration to environment variables
   - Update visualization in `draw_detections()`

2. **Optimizing Performance**:
   - Profile using `benchmark_yolo_models.py`
   - Consider TensorRT export using `export_tensorrt.py`
   - Balance YOLO model size vs accuracy
   - Ensure CUDA is being used (check logs for "Using GPU")

## Critical Dependencies

- **PyTorch**: Must be Jetson-specific wheel (not PyPI)
- **ByteTrack**: Has torch==1.13.0 dependency conflict - install with `--no-deps` if needed
- **OpenCV**: Prefer system package on Jetson (`apt install python3-opencv`)
- **pyusb**: Version 1.0.2 required for ReSpeaker compatibility
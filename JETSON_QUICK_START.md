# Quick Jetson Setup Script

This script automates most of the Jetson setup process.

## Usage

```bash
# Make executable
chmod +x jetson_setup.sh

# Run (will prompt for sudo password)
./jetson_setup.sh
```

## Manual Setup

If you prefer step-by-step, follow `JETSON_DEPLOYMENT_GUIDE.md` instead.

---

## Pre-Flight Checklist

Before running, ensure:
- [ ] JetPack is flashed (CUDA, cuDNN, TensorRT installed)
- [ ] Internet connection active
- [ ] User has sudo privileges
- [ ] ReSpeaker connected via USB
- [ ] Camera connected

---

## After Setup

1. **Export TensorRT model**:
   ```bash
   source venv/bin/activate
   python3 export_tensorrt.py
   ```

2. **Run validation**:
   ```bash
   python3 validate_setup.py
   ```

3. **Start application**:
   ```bash
   sudo jetson_clocks  # Maximum performance
   python3 audio_vision_fusion.py
   ```

---

## Performance Optimization

Edit `audio_vision_fusion.py`:
```python
FRAME_WIDTH = 640   # Jetson-optimized
FRAME_HEIGHT = 480
YOLO_MODEL_PATH = 'yolov10n.engine'  # Use TensorRT
```

---

See `JETSON_DEPLOYMENT_GUIDE.md` for complete details.


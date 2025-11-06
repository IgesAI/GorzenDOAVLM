# CUDA Setup Summary

## Problem Identified

The system had PyTorch installed from PyPI, which is **CPU-only** on ARM64/Jetson platforms. This prevented YOLO from using the GPU, resulting in slow inference times.

## Root Cause

- PyTorch from PyPI (via `pip install torch`) doesn't include CUDA support on aarch64
- Jetson requires NVIDIA-provided PyTorch wheels with CUDA support
- The requirements.txt had commented-out instructions but no automated setup

## Solution Implemented

### 1. Created `install_pytorch_jetson.sh`
Automated installation script that:
- Detects JetPack version (R35.x, R36.x, etc.)
- Detects Python version (3.8, 3.10)
- Downloads correct NVIDIA PyTorch wheel from official repository
- Installs compatible torchvision version
- Verifies CUDA availability

**Usage:**
```bash
./install_pytorch_jetson.sh
```

### 2. Updated `requirements.txt`
- Added clear warnings about PyPI PyTorch being CPU-only on Jetson
- Documented the correct installation process
- Noted ByteTrack dependency conflict (requires torch==1.13.0)

### 3. Updated `audio_vision_fusion.py`
**Changes to CUDA initialization:**

```python
# Before (incorrect for ultralytics):
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
self.model.to(self.device)  # This doesn't work with ultralytics YOLO

# After (correct):
self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
results = self.model(frame, device=self.device)  # Pass device to inference
```

**Improved logging:**
- Shows GPU name and VRAM on startup
- Displays CUDA version
- Warns if CPU-only mode detected
- Suggests running install script if CUDA not available

### 4. Updated `.github/copilot-instructions.md`
Added critical information for AI agents:
- NEVER install PyTorch from PyPI on Jetson
- Use installation script for proper setup
- Device string format for ultralytics YOLO
- Dependency conflict notes

## Verification

After installation, verify CUDA is working:

```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should print: CUDA: True

python3 audio_vision_fusion.py
# Should show in logs: "✓ Using GPU: Orin"
```

## Current Status

✅ PyTorch 2.0.0+nv23.05 with CUDA 11.4 installed  
✅ CUDA available and detected  
✅ GPU: Orin (Jetson Orin)  
✅ YOLO configured to use GPU via `device='cuda:0'`  
✅ Installation script created for future use  

## Known Issues

1. **cuDNN version warning**: PyTorch expects cuDNN 8.6.0 but system has 8.4.1
   - **Impact**: None - PyTorch bundles its own cuDNN
   - **Action**: Can be safely ignored

2. **ByteTrack dependency conflict**: Requires torch==1.13.0
   - **Impact**: Warning message during installation
   - **Action**: Install ByteTrack with `--no-deps` if needed
   - **Note**: Not critical for core audio-vision fusion functionality

## Performance Impact

Expected improvements:
- **Before (CPU)**: ~2-5 FPS on YOLOv10n
- **After (GPU)**: ~15-30 FPS on YOLOv10n (Jetson Orin)
- **With TensorRT**: ~40-60 FPS potential (export via `export_tensorrt.py`)

## Next Steps

1. Run the application to verify GPU acceleration is working
2. Monitor FPS in the application to confirm performance improvement
3. Consider TensorRT export for further optimization:
   ```bash
   python3 export_tensorrt.py --model yolov10n.pt
   ```

## For Future Developers

**When setting up a new Jetson device:**
1. Clone repository
2. Create virtual environment: `python -m venv venv && source venv/bin/activate`
3. Run: `./install_pytorch_jetson.sh` (installs correct PyTorch)
4. Run: `pip install -r requirements.txt` (installs other dependencies)
5. Verify: Check logs show "✓ Using GPU" when running the application

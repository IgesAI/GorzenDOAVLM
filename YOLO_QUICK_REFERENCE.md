# Quick Reference: YOLO Model Selection

## TL;DR - Is YOLOv8n Overkill?

**No, YOLOv8n is already the smallest YOLOv8 variant** (~6MB, ~3.2M parameters)

**However**, if you want:
- **Faster inference**: Use **YOLOv10n** (2-3x faster, drop-in replacement)
- **Ultra-fast**: Use **RTMDet-Tiny** (5x faster, more setup)
- **Aerospace-specific**: Use **SenseLite** (better for small objects)

---

## Quick Model Comparison

| Model | Speed | GPU Memory | Best For |
|-------|-------|------------|----------|
| **yolov8n.pt** (current) | 200-300 FPS | 500MB-1GB | ✅ Good balance |
| **yolov10n.pt** ⭐ | 400-700 FPS | 500MB-1GB | 🚀 **Recommended upgrade** |
| **RTMDet-Tiny** ⭐ | 1000+ FPS | 200-400MB | ⚡ **Fastest** |
| **yolov5n.pt** | 200-300 FPS | 400-800MB | Stable production |

---

## Easy Upgrade: Switch to YOLOv10n

### Option 1: Environment Variable
```bash
export YOLO_MODEL_PATH=yolov10n.pt
python audio_vision_fusion.py
```

### Option 2: Edit Code
```python
# In audio_vision_fusion.py line 61
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'yolov10n.pt')
```

**That's it!** Model will auto-download on first run.

---

## Benchmark Your Models

Run the benchmark script to compare models:

```bash
python benchmark_yolo_models.py
```

This will test:
- yolov8n.pt (current)
- yolov10n.pt (newest)
- yolov8s.pt (for comparison)
- yolov5n.pt (alternative)

---

## If GPU Still Strained

### 1. Reduce Resolution (Easiest)
```python
FRAME_WIDTH = 640   # Instead of 1280
FRAME_HEIGHT = 480  # Instead of 720
```

### 2. Use FP16 Precision
```python
# In audio_vision_fusion.py, modify inference call
results = self.model(frame, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False, half=True)
```

### 3. Skip Frames
```python
# Process every 2nd frame
if frame_count % 2 == 0:
    results = self.model(frame)
```

---

## Aerospace/Manufacturing Specific

### For Small Objects (Aircraft Components, UAVs)
- **SenseLite**: Best for aerial/small object detection
- **HierLight-YOLO**: Good for UAV photography
- **AeroLight**: Aerospace-focused

### For Manufacturing Quality Control
- **YOLOv10n**: Good for general QC
- **RTMDet-Tiny**: Best for high-speed production lines

---

## What GPU Do You Have?

**Check your GPU:**
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Recommendations by GPU:**
- **< 2GB VRAM**: RTMDet-Tiny or reduce resolution
- **2-4GB VRAM**: YOLOv10n or YOLOv8n
- **4-8GB VRAM**: YOLOv10n or YOLOv8s
- **> 8GB VRAM**: Any model, use YOLOv8s+ for accuracy

---

## Next Steps

1. **Try YOLOv10n** (easiest upgrade, just change model name)
2. **Run benchmark** (`python benchmark_yolo_models.py`)
3. **Check GPU usage** in logs when running
4. **Reduce resolution** if needed
5. **Consider specialized models** if aerospace/manufacturing specific

---

## Full Analysis

See `YOLO_MODEL_ANALYSIS.md` for:
- Detailed model specifications
- Performance benchmarks
- Migration guides
- Custom training tips


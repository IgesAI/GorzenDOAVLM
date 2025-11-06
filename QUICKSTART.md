# Quick Start Guide

## Prerequisites Check

Before running, ensure you have:

1. **ReSpeaker USB Mic Array** connected and powered (LEDs should be on)
2. **Camera** connected
3. **Python 3.8+** installed
4. **Dependencies** installed: `pip install -r requirements.txt`

## Platform-Specific Setup

### Windows
```bash
# Install libusb-win32 driver using Zadig
# See: usb_4_mic_array/README.md for details
```

### Linux
```bash
# May need to run with sudo or configure udev rules
sudo python audio_vision_fusion.py
```

### macOS
```bash
# Should work without additional setup
python audio_vision_fusion.py
```

## First Run

1. **Activate virtual environment** (if using one):
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

2. **Run the application**:
   ```bash
   python audio_vision_fusion.py
   ```

3. **YOLO model** will be auto-downloaded on first run if not present

4. **Press 'q'** to quit gracefully

## Verification

Check that everything is working:

1. **Check logs**: `audio_vision_fusion.log` should show successful initialization
2. **Camera**: Video window should open showing camera feed
3. **Audio**: Make a sound - objects should highlight in red
4. **DOA**: Status overlay should show DOA angle when sound detected

## Troubleshooting

See `README.md` for detailed troubleshooting guide.

## Configuration

Edit configuration in `audio_vision_fusion.py` or use environment variables (see `README.md`).


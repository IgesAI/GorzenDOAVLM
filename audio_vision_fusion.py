"""
Audio-Visual Fusion: YOLO Object Detection with ReSpeaker DOA
=============================================================

This module combines YOLO object detection with ReSpeaker microphone array
Direction of Arrival (DOA) to highlight objects based on sound direction.

Requirements:
- ReSpeaker USB Mic Array v2.1 connected
- Camera connected
- YOLOv10 model file (yolov10n.pt) - auto-downloads on first run
"""

import cv2
import numpy as np
import time
import threading
import sounddevice as sd
from ultralytics import YOLO
import os
import sys
import signal
import logging
import platform
from pathlib import Path
from typing import Optional, Tuple
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_vision_fusion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add USB mic array directory to path
USB_MIC_DIR = Path(__file__).parent / 'usb_4_mic_array'
sys.path.insert(0, str(USB_MIC_DIR))
try:
    import audio_doa
except ImportError as e:
    logger.error(f"Failed to import audio_doa module: {e}")
    logger.error(f"Please ensure '{USB_MIC_DIR}' exists and contains audio_doa.py")
    sys.exit(1)


# ==================== Configuration ====================
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', '1280'))
FRAME_HEIGHT = int(os.getenv('FRAME_HEIGHT', '720'))
CAMERA_FOV = float(os.getenv('CAMERA_FOV', '78'))  # Horizontal field-of-view in degrees
# Model options:
# - yolov10n.pt: Default (newest, fastest, recommended for Jetson)
# - yolov8n.pt: Alternative (slightly slower)
# - yolov5n.pt: Older but stable
# - yolov8s.pt: Larger, more accurate (if GPU allows)
# See YOLO_MODEL_ANALYSIS.md for detailed comparison
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'yolov10n.pt')
SOUND_SAMPLE_RATE = int(os.getenv('SOUND_SAMPLE_RATE', '16000'))
SOUND_CHUNK_DURATION = float(os.getenv('SOUND_CHUNK_DURATION', '0.1'))
SOUND_THRESHOLD_DB = float(os.getenv('SOUND_THRESHOLD_DB', '-30'))
DOA_OFFSET = float(os.getenv('DOA_OFFSET', '0'))
DOA_FLIP = os.getenv('DOA_FLIP', 'True').lower() == 'true'
DOA_SNAP_TOLERANCE = float(os.getenv('DOA_SNAP_TOLERANCE', '10'))
TRIGGER_HOLD_TIME = float(os.getenv('TRIGGER_HOLD_TIME', '0.5'))
MIC_DEVICE_NAME = os.getenv('MIC_DEVICE_NAME', 'ReSpeaker')
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv('YOLO_CONFIDENCE_THRESHOLD', '0.25'))

# Object tracking configuration
MAX_SOUND_HITS = int(os.getenv('MAX_SOUND_HITS', '5'))  # Number of sound hits for 100% confidence
TRACKING_IOU_THRESHOLD = float(os.getenv('TRACKING_IOU_THRESHOLD', '0.5'))  # IoU threshold for tracking
SOUND_HISTORY_WINDOW = float(os.getenv('SOUND_HISTORY_WINDOW', '2.0'))  # Seconds to keep sound history
DOA_LOCK_TOLERANCE = float(os.getenv('DOA_LOCK_TOLERANCE', '20'))  # Max angle difference to lock (degrees)
SOUND_HIT_DEBOUNCE = float(os.getenv('SOUND_HIT_DEBOUNCE', '0.3'))  # Minimum seconds between sound hits
# =======================================================


class AudioVisionFusion:
    """Main class for audio-visual fusion system."""
    
    def __init__(self):
        """Initialize the audio-visual fusion system."""
        self.validate_config()
        self.setup_platform_checks()
        
        # Shared state for audio trigger
        self.last_trigger_time = 0.0
        self.last_doa_angle: Optional[float] = None
        self.audio_trigger_lock = threading.Lock()
        self.audio_active = True
        
        # Object tracking state
        self.locked_object_id = None  # ID of locked object (tracking ID)
        self.locked_object_bbox = None  # Last known bbox (x1, y1, x2, y2)
        self.sound_detections = []  # List of (timestamp, doa_angle) for locked object
        self.sound_confidence = 0.0  # 0.0 to 1.0, how many sounds detected
        self.max_sound_hits = MAX_SOUND_HITS  # Number of hits to reach 100% confidence
        
        # Resources
        self.audio_doa_instance: Optional[audio_doa.ReSpeakerDOA] = None
        self.audio_thread: Optional[threading.Thread] = None
        self.model: Optional[YOLO] = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def validate_config(self):
        """Validate configuration parameters."""
        assert CAMERA_FOV > 0 and CAMERA_FOV <= 180, f"Invalid FOV: {CAMERA_FOV} (must be 0-180 degrees)"
        assert FRAME_WIDTH > 0 and FRAME_HEIGHT > 0, f"Invalid frame dimensions: {FRAME_WIDTH}x{FRAME_HEIGHT}"
        assert SOUND_THRESHOLD_DB <= 0, f"Sound threshold should be <= 0 dBFS, got {SOUND_THRESHOLD_DB}"
        assert SOUND_CHUNK_DURATION > 0, f"Sound chunk duration must be positive, got {SOUND_CHUNK_DURATION}"
        assert TRIGGER_HOLD_TIME > 0, f"Trigger hold time must be positive, got {TRIGGER_HOLD_TIME}"
        assert DOA_SNAP_TOLERANCE >= 0, f"DOA snap tolerance must be non-negative, got {DOA_SNAP_TOLERANCE}"
        assert 0 <= YOLO_CONFIDENCE_THRESHOLD <= 1, f"YOLO confidence must be 0-1, got {YOLO_CONFIDENCE_THRESHOLD}"
        logger.info("Configuration validation passed")
    
    def setup_platform_checks(self):
        """Perform platform-specific checks and warnings."""
        system = platform.system()
        logger.info(f"Running on {system} {platform.release()}")
        
        if system == 'Windows':
            logger.warning("Windows detected: Ensure libusb-win32 driver is installed for ReSpeaker")
            logger.warning("Use Zadig tool to install driver: https://zadig.akeo.ie/")
            logger.warning("See usb_4_mic_array/README.md for details")
        elif system == 'Linux':
            logger.info("Linux detected: Ensure udev rules are set up for USB access")
            logger.info("You may need to run with sudo or add user to appropriate groups")
            # Check if running on Jetson
            if 'jetson' in platform.platform().lower() or Path('/etc/nv_tegra_release').exists():
                logger.info("NVIDIA Jetson detected - optimizations enabled")
                logger.info("Recommendation: Use TensorRT engine for best performance")
                logger.info("  Export with: python3 export_tensorrt.py")
        elif system == 'Darwin':
            logger.info("macOS detected: USB access should work without additional setup")
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully."""
        logger.info("\nReceived shutdown signal, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def rms_dbfs(self, audio_samples: np.ndarray) -> float:
        """Compute RMS level in dBFS from audio sample array.
        
        Args:
            audio_samples: Audio samples as numpy array
            
        Returns:
            RMS level in dBFS
        """
        rms = np.sqrt(np.mean(np.square(audio_samples), dtype=np.float64))
        if rms <= 1e-9:
            return -float('inf')
        return 20 * np.log10(rms)
    
    def audio_capture_thread(self):
        """Thread function to capture audio and detect sound direction."""
        logger.info("Audio capture thread started")
        
        # Find input device index and info for ReSpeaker Mic Array
        device_index = None
        device_info = None
        devices = sd.query_devices()
        
        logger.info(f"Searching for microphone device containing '{MIC_DEVICE_NAME}'...")
        for idx, dev in enumerate(devices):
            if dev.get('max_input_channels', 0) > 0 and MIC_DEVICE_NAME.lower() in dev['name'].lower():
                device_index = idx
                device_info = dev
                logger.info(f"Found microphone device: {dev['name']} (index {idx})")
                break
        
        if device_index is None:
            logger.error(f"Microphone device with name containing '{MIC_DEVICE_NAME}' not found.")
            logger.error("Available input devices:")
            for idx, dev in enumerate(devices):
                if dev.get('max_input_channels', 0) > 0:
                    logger.error(f"  [{idx}] {dev['name']}")
            self.audio_active = False
            return
        
        # Use device default samplerate/channels if available
        channels_count = int(device_info.get('max_input_channels', 1))
        samplerate = int(device_info.get('default_samplerate', SOUND_SAMPLE_RATE))
        frames_per_chunk = int(samplerate * SOUND_CHUNK_DURATION)
        
        logger.info(f"Audio settings: {channels_count} channels, {samplerate} Hz, {frames_per_chunk} frames/chunk")
        
        try:
            with sd.InputStream(device=device_index, channels=channels_count, 
                              samplerate=samplerate, dtype='float32') as stream:
                overflow_count = 0
                while self.audio_active:
                    # Read one chunk of audio data
                    audio_data, overflowed = stream.read(frames_per_chunk)
                    if overflowed:
                        overflow_count += 1
                        if overflow_count % 10 == 0:  # Log every 10th overflow
                            logger.warning(f"Audio buffer overflow detected ({overflow_count} total)")
                    
                    # Compute sound level (dBFS) for the chunk (use channel 0)
                    level_db = self.rms_dbfs(audio_data[:, 0])
                    
                    # Check ReSpeaker's voice activity and speech detection status
                    try:
                        voice = self.audio_doa_instance.voice_active()
                        speech = self.audio_doa_instance.speech_detected()
                    except Exception as e:
                        logger.error(f"Error reading DOA status: {e}")
                        continue
                    
                    # Determine if this chunk represents a sound event
                    if level_db > SOUND_THRESHOLD_DB or voice or speech:
                        current_time = time.time()
                        try:
                            doa_angle = self.audio_doa_instance.doa()
                        except Exception as e:
                            logger.error(f"Error reading DOA angle: {e}")
                            continue
                        
                        # Flip left/right if device orientation is mirrored
                        if DOA_FLIP:
                            doa_angle = (-doa_angle) % 360
                        
                        # Adjust DOA angle with offset so that 0° corresponds to camera forward
                        doa_angle = (doa_angle + DOA_OFFSET) % 360
                        
                        # Convert absolute DOA to relative angle (-180 to 180) w.r.t. camera center
                        doa_rel = doa_angle if doa_angle <= 180 else doa_angle - 360
                        
                        with self.audio_trigger_lock:
                            # If no recent trigger (or first trigger), accept new DOA immediately
                            if self.last_doa_angle is None or (current_time - self.last_trigger_time) > TRIGGER_HOLD_TIME:
                                self.last_doa_angle = doa_rel
                                logger.debug(f"New DOA trigger: {doa_rel:.1f}°")
                            else:
                                # If continuing trigger, only update DOA if change exceeds tolerance
                                if abs(doa_rel - self.last_doa_angle) >= DOA_SNAP_TOLERANCE:
                                    self.last_doa_angle = doa_rel
                                    logger.debug(f"DOA updated: {doa_rel:.1f}°")
                            
                            # Update trigger time (extend the trigger hold period)
                            self.last_trigger_time = current_time
                            
        except Exception as e:
            logger.error(f"Audio capture error: {e}", exc_info=True)
            self.audio_active = False
        
        logger.info("Audio capture thread stopped")
    
    def initialize_audio(self):
        """Initialize ReSpeaker DOA interface and start audio thread."""
        logger.info("Initializing ReSpeaker DOA interface...")
        try:
            self.audio_doa_instance = audio_doa.ReSpeakerDOA()
            logger.info("ReSpeaker DOA interface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ReSpeaker audio DOA interface: {e}")
            logger.error("Ensure the ReSpeaker is connected and accessible.")
            if platform.system() == 'Linux':
                logger.error("On Linux, you may need sudo or udev rules for USB access.")
            elif platform.system() == 'Windows':
                logger.error("On Windows, ensure libusb-win32 driver is installed via Zadig.")
            raise
        
        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self.audio_capture_thread, daemon=True)
        self.audio_thread.start()
        logger.info("Audio capture thread started")
        
        # Give thread a moment to initialize
        time.sleep(0.1)
    
    def initialize_yolo(self):
        """Initialize YOLO model (YOLOv10 optimized for Jetson)."""
        logger.info(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
        
        # Check if model file exists
        model_path = Path(YOLO_MODEL_PATH)
        
        # Check for TensorRT engine first (preferred for Jetson)
        engine_path = model_path.with_suffix('.engine')
        if engine_path.exists() and model_path.suffix == '.pt':
            logger.info(f"TensorRT engine found: {engine_path}")
            logger.info("Using TensorRT engine for optimal Jetson performance")
            model_path = engine_path
        
        if not model_path.exists():
            # Try to auto-download if it's a standard model name
            if YOLO_MODEL_PATH in ['yolov10n.pt', 'yolov8n.pt', 'yolov5n.pt']:
                logger.info(f"Model not found locally, will attempt to download...")
            else:
                error_msg = f"YOLO model file not found: {YOLO_MODEL_PATH}"
                logger.error(error_msg)
                logger.error("Please download the model or update YOLO_MODEL_PATH")
                raise FileNotFoundError(error_msg)
        
        try:
            self.model = YOLO(str(model_path))
            logger.info(f"YOLO model loaded successfully (classes: {len(self.model.names)})")
            
            # Detect and configure device (CUDA vs CPU)
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Selected device: {self.device}")

            # Log GPU info if CUDA available
            if torch.cuda.is_available():
                try:
                    logger.info(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
                    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info(f"  GPU VRAM: {total_vram:.1f} GB")
                    logger.info(f"  CUDA version: {torch.version.cuda}")
                    # Jetson-specific warnings
                    if total_vram < 4:
                        logger.warning("  Low VRAM detected - consider using 640x480 resolution")
                        logger.warning("  For best performance, export to TensorRT engine")
                except Exception as e:
                    logger.debug(f"Unable to query GPU properties: {e}")
            else:
                logger.warning("⚠ CUDA not available, using CPU (will be slow)")
                logger.warning("  Ensure PyTorch with CUDA is installed")
                logger.warning("  On Jetson: run ./install_pytorch_jetson.sh")
                
            # Check if TensorRT engine is being used
            if model_path.suffix == '.engine':
                logger.info("✓ Using TensorRT engine (optimal for Jetson)")
            elif model_path.suffix == '.pt':
                logger.info("⚠ Using PyTorch model - consider exporting to TensorRT:")
                logger.info(f"  python3 export_tensorrt.py --model {model_path.name}")
                
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            logger.error("Model options: yolov10n.pt, yolov8n.pt, yolov5n.pt")
            logger.error("Or use TensorRT engine: yolov10n.engine")
            raise
    
    def initialize_camera(self):
        """Initialize camera capture."""
        logger.info(f"Initializing camera at index {CAMERA_INDEX}...")
        
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            error_msg = f"Failed to open camera at index {CAMERA_INDEX}"
            logger.error(error_msg)
            logger.error("Please check camera connection and permissions")
            raise RuntimeError(error_msg)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        # Verify settings were applied
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width != FRAME_WIDTH or actual_height != FRAME_HEIGHT:
            logger.warning(f"Camera resolution is {actual_width}x{actual_height}, requested {FRAME_WIDTH}x{FRAME_HEIGHT}")
        else:
            logger.info(f"Camera resolution set to {actual_width}x{actual_height}")
        
        # Get FPS
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera FPS: {fps}")
    
    def calculate_object_angle(self, center_x: float, frame_width: int) -> float:
        """Calculate object angle relative to camera center.
        
        Args:
            center_x: X coordinate of object center
            frame_width: Width of the frame
            
        Returns:
            Angle in degrees relative to camera center (-FOV/2 to +FOV/2)
        """
        # Normalize to -1 to +1 (left to right)
        normalized = (center_x - frame_width / 2) / (frame_width / 2)
        # Scale to FOV range
        return normalized * (CAMERA_FOV / 2)
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def find_closest_object(self, detections, current_doa: float) -> Optional[int]:
        """Find object closest to DOA angle, with tracking support.
        
        Args:
            detections: YOLO detection results
            current_doa: Current DOA angle in degrees
            
        Returns:
            Index of closest object or None
        """
        if len(detections.boxes.cls) == 0:
            # No objects detected - check if locked object is still in frame
            if self.locked_object_id is not None:
                logger.debug("Locked object out of frame - resetting")
                self.locked_object_id = None
                self.locked_object_bbox = None
                self.sound_detections = []
                self.sound_confidence = 0.0
            return None
        
        boxes_xyxy = detections.boxes.xyxy.cpu().numpy()
        centers_x = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
        rel_angles = np.array([self.calculate_object_angle(cx, FRAME_WIDTH) for cx in centers_x])
        
        # Find detection with minimum angular difference to the sound direction
        diffs = np.abs(rel_angles - current_doa)
        closest_idx = int(np.argmin(diffs))
        closest_angle_diff = diffs[closest_idx]
        
        # Only lock onto objects within reasonable angular range
        if closest_angle_diff > DOA_LOCK_TOLERANCE:
            logger.debug(f"Closest object at {closest_angle_diff:.1f}° away - too far from DOA (tolerance: {DOA_LOCK_TOLERANCE}°)")
            # If we had a locked object but sound is now far away, keep tracking but don't increment
            if self.locked_object_id is not None and self.locked_object_bbox is not None:
                # Try to maintain tracking of existing object
                best_iou = 0.0
                best_idx = None
                
                for i, bbox in enumerate(boxes_xyxy):
                    iou = self.calculate_iou(self.locked_object_bbox, bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                
                if best_iou > TRACKING_IOU_THRESHOLD:
                    self.locked_object_bbox = boxes_xyxy[best_idx]
                    logger.debug(f"Maintaining lock without new sound (IoU: {best_iou:.2f})")
                    return best_idx
                else:
                    logger.debug("Lost locked object - resetting")
                    self.locked_object_id = None
                    self.locked_object_bbox = None
                    self.sound_detections = []
                    self.sound_confidence = 0.0
            return None
        
        # Check if we have a locked object and try to find it in current frame
        current_time = time.time()
        
        if self.locked_object_id is not None and self.locked_object_bbox is not None:
            # Try to match locked object with current detections using IoU
            best_iou = 0.0
            best_idx = None
            
            for i, bbox in enumerate(boxes_xyxy):
                iou = self.calculate_iou(self.locked_object_bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            # If we found a good match (IoU > threshold), continue tracking
            if best_iou > TRACKING_IOU_THRESHOLD:
                self.locked_object_bbox = boxes_xyxy[best_idx]
                
                # Check if the new sound is from the locked object's direction
                locked_obj_angle = rel_angles[best_idx]
                angle_to_locked = abs(locked_obj_angle - current_doa)
                
                if angle_to_locked < DOA_LOCK_TOLERANCE:
                    # Sound is from locked object's direction - increment confidence
                    # Only add if not too recent (debounce to prevent rapid hits)
                    # Check last detection time to enforce minimum gap
                    last_detection_time = self.sound_detections[-1][0] if self.sound_detections else 0
                    time_since_last = current_time - last_detection_time
                    
                    if time_since_last > SOUND_HIT_DEBOUNCE:  # Configurable debounce time
                        self.sound_detections.append((current_time, current_doa))
                        # Remove old detections (older than configured window)
                        self.sound_detections = [(t, a) for t, a in self.sound_detections 
                                                if current_time - t < SOUND_HISTORY_WINDOW]
                        self.sound_confidence = min(1.0, len(self.sound_detections) / self.max_sound_hits)
                        logger.info(f"Sound from locked object! Confidence: {self.sound_confidence*100:.0f}% ({len(self.sound_detections)} hits)")
                    else:
                        logger.debug(f"Sound detected but too soon ({time_since_last:.1f}s < {SOUND_HIT_DEBOUNCE}s, debounce)")
                else:
                    logger.debug(f"Sound from different direction ({angle_to_locked:.1f}° away from locked object)")
                
                logger.debug(f"Tracking locked object (IoU: {best_iou:.2f})")
                return best_idx
            else:
                # Lost track of locked object - check if closest object is the new target
                logger.info("Lost locked object - checking for new target")
                self.locked_object_id = None
                self.locked_object_bbox = None
                self.sound_detections = []
                self.sound_confidence = 0.0
                # Fall through to create new lock below
        
        # No locked object or lost track - create new lock on closest object
        # First sound detection - start tracking this object
        self.locked_object_id = closest_idx  # Simple index-based tracking
        self.locked_object_bbox = boxes_xyxy[closest_idx]
        self.sound_detections = [(current_time, current_doa)]
        self.sound_confidence = 1.0 / self.max_sound_hits
        logger.info(f"Locked onto new object at DOA {current_doa:.1f}° (confidence: {self.sound_confidence*100:.0f}%)")
        
        return closest_idx
    
    def draw_detections(self, frame: np.ndarray, detections, highlight_idx: Optional[int]):
        """Draw object detections on frame with enhanced tracking visuals.
        
        Args:
            frame: Image frame to draw on
            detections: YOLO detection results
            highlight_idx: Index of object to highlight (if any)
        """
        if len(detections.boxes.cls) == 0:
            return
        
        boxes_xyxy = detections.boxes.xyxy.cpu().numpy()
        confs = detections.boxes.conf.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy().astype(int)
        
        for i, bbox in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            cls_id = classes[i] if i < len(classes) else -1
            conf = confs[i] if i < len(confs) else None
            
            # Filter by confidence threshold
            if conf is not None and conf < YOLO_CONFIDENCE_THRESHOLD:
                continue
            
            # Determine if this is the locked/tracked object
            # Check both: is this the highlight_idx AND is there a locked object
            is_locked = (self.locked_object_id is not None and 
                        i == highlight_idx and 
                        self.sound_confidence > 0)
            
            # Choose bounding box color
            if is_locked:
                # Locked object - color intensity based on confidence
                confidence_intensity = int(255 * self.sound_confidence)
                color = (0, 255 - confidence_intensity, confidence_intensity)  # Green to Red transition
                thickness = 4
            elif highlight_idx is not None and i == highlight_idx:
                color = (0, 165, 255)  # Orange for initial detection
                thickness = 3
            else:
                color = (0, 255, 0)  # Green for other objects
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw loading bar for locked object
            if is_locked and self.sound_confidence > 0:
                bar_width = x2 - x1
                bar_height = 10
                bar_x = x1
                bar_y = y2 + 5
                
                # Background bar (gray)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                            (100, 100, 100), cv2.FILLED)
                
                # Filled portion (confidence level)
                filled_width = int(bar_width * self.sound_confidence)
                # Color gradient from yellow to red
                bar_color = (0, int(255 * (1 - self.sound_confidence)), 
                           int(255 * (0.5 + 0.5 * self.sound_confidence)))
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                            bar_color, cv2.FILLED)
                
                # Border
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                            (255, 255, 255), 1)
                
                # Percentage text
                conf_text = f"{int(self.sound_confidence * 100)}%"
                text_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                text_x = bar_x + bar_width + 5
                text_y = bar_y + bar_height - 2
                cv2.putText(frame, conf_text, (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Prepare label text (class name and confidence)
            label = ""
            if cls_id >= 0 and cls_id in self.model.names:
                label = self.model.names[cls_id]
            if conf is not None:
                conf_pct = conf * 100
                label = f"{label} {conf_pct:.1f}%" if label else f"{conf_pct:.1f}%"
            
            # Add "LOCKED" indicator
            if is_locked:
                label = f"🔒 {label}" if label else "🔒 LOCKED"
            
            if label:
                # Draw filled background for label for readability
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_w, text_h = text_size
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.rectangle(frame, (x1, label_y - text_h), (x1 + text_w, label_y), color, cv2.FILLED)
                cv2.putText(frame, label, (x1, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_status(self, frame: np.ndarray):
        """Draw status information on frame.
        
        Args:
            frame: Image frame to draw on
        """
        with self.audio_trigger_lock:
            triggered = (time.time() - self.last_trigger_time) < TRIGGER_HOLD_TIME
            current_doa = self.last_doa_angle if triggered else None
        
        # Draw DOA indicator
        if current_doa is not None:
            status_text = f"DOA: {current_doa:.1f}°"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw trigger status
        trigger_text = "TRIGGERED" if triggered else "WAITING"
        trigger_color = (0, 255, 0) if triggered else (128, 128, 128)
        cv2.putText(frame, trigger_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, trigger_color, 2)
        
        # Draw tracking status
        if self.locked_object_id is not None:
            tracking_text = f"LOCKED: {self.sound_confidence*100:.0f}% ({len(self.sound_detections)} hits)"
            tracking_color = (0, int(255 * (1 - self.sound_confidence)), int(255 * self.sound_confidence))
            cv2.putText(frame, tracking_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracking_color, 2)
    
    def run(self):
        """Run the main processing loop."""
        try:
            # Initialize components
            self.initialize_audio()
            self.initialize_yolo()
            self.initialize_camera()
            
            logger.info("Starting main processing loop. Press 'q' to quit.")
            frame_count = 0
            start_time = time.time()
            
            # Main loop: process video frames and overlay detection + audio cues
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Unable to read from camera.")
                    break
                
                # Run YOLOv10 inference on the frame
                # Automatically uses TensorRT if .engine file is provided
                # Pass device explicitly to ensure GPU usage
                results = self.model(frame, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False, device=self.device)
                detections = results[0]
                highlight_idx = None
                
                # Check if locked object is still in frame (even without new sound)
                if self.locked_object_id is not None and self.locked_object_bbox is not None:
                    if len(detections.boxes.cls) > 0:
                        boxes_xyxy = detections.boxes.xyxy.cpu().numpy()
                        best_iou = 0.0
                        best_idx = None
                        
                        for i, bbox in enumerate(boxes_xyxy):
                            iou = self.calculate_iou(self.locked_object_bbox, bbox)
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = i
                        
                        if best_iou > TRACKING_IOU_THRESHOLD:
                            # Update bbox and set as highlight
                            self.locked_object_bbox = boxes_xyxy[best_idx]
                            highlight_idx = best_idx
                            logger.debug(f"Locked object in frame (IoU: {best_iou:.2f})")
                        else:
                            # Lost the object
                            logger.info("Locked object left frame - resetting")
                            self.locked_object_id = None
                            self.locked_object_bbox = None
                            self.sound_detections = []
                            self.sound_confidence = 0.0
                    else:
                        # No objects detected
                        logger.info("No objects detected - resetting lock")
                        self.locked_object_id = None
                        self.locked_object_bbox = None
                        self.sound_detections = []
                        self.sound_confidence = 0.0
                
                # Determine if an audio trigger is active and select target object
                with self.audio_trigger_lock:
                    triggered = (time.time() - self.last_trigger_time) < TRIGGER_HOLD_TIME
                    current_doa = self.last_doa_angle if triggered else None
                
                # Call find_closest_object when we have a new sound trigger
                # This updates confidence for locked objects OR creates new locks
                if triggered and current_doa is not None:
                    result_idx = self.find_closest_object(detections, current_doa)
                    # Use the result if we don't already have a highlight from tracking
                    if highlight_idx is None and result_idx is not None:
                        highlight_idx = result_idx
                
                # Draw detections and status
                self.draw_detections(frame, detections, highlight_idx)
                self.draw_status(frame)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    logger.debug(f"Processing at {fps:.1f} FPS")
                
                # Display the resulting frame
                cv2.imshow("Audio-Visual Fusion", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested by user")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        # Stop audio thread
        self.audio_active = False
        if self.audio_thread is not None:
            logger.info("Waiting for audio thread to stop...")
            self.audio_thread.join(timeout=2.0)
            if self.audio_thread.is_alive():
                logger.warning("Audio thread did not stop within timeout")
        
        # Close audio DOA interface
        if self.audio_doa_instance is not None and hasattr(self.audio_doa_instance, '_tuning'):
            try:
                self.audio_doa_instance._tuning.close()
                logger.info("ReSpeaker DOA interface closed")
            except Exception as e:
                logger.warning(f"Error closing DOA interface: {e}")
        
        # Release camera
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        logger.info("Cleanup complete")


def main():
    """Main entry point."""
    try:
        app = AudioVisionFusion()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

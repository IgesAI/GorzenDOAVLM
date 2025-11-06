"""
Export YOLOv10 to TensorRT Engine for Jetson
===========================================

This script exports the YOLOv10 model to TensorRT format for optimal
performance on NVIDIA Jetson devices.
"""

import logging
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics not installed. Run: pip install ultralytics")
    exit(1)


def export_to_tensorrt(model_path: str = 'yolov10n.pt', 
                       output_path: str = None,
                       workspace: int = 4,
                       half: bool = True):
    """Export YOLO model to TensorRT engine.
    
    Args:
        model_path: Path to YOLO model (.pt file)
        output_path: Output path for engine (default: model_path with .engine extension)
        workspace: TensorRT workspace size in GB
        half: Use FP16 precision (recommended for Jetson)
    """
    logger.info("=" * 60)
    logger.info("YOLOv10 TensorRT Export for Jetson")
    logger.info("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Cannot export TensorRT engine.")
        logger.error("Ensure CUDA is installed and GPU is accessible.")
        return False
    
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Determine output path
    if output_path is None:
        model_file = Path(model_path)
        output_path = str(model_file.with_suffix('.engine'))
    
    logger.info(f"Input model: {model_path}")
    logger.info(f"Output engine: {output_path}")
    logger.info(f"Workspace: {workspace}GB")
    logger.info(f"Precision: {'FP16' if half else 'FP32'}")
    logger.info("")
    logger.info("This may take several minutes...")
    
    try:
        # Load model
        logger.info("Loading YOLO model...")
        model = YOLO(model_path)
        
        # Export to TensorRT
        logger.info("Exporting to TensorRT engine...")
        model.export(
            format='engine',
            device=0,          # GPU 0
            half=half,          # FP16
            workspace=workspace, # Workspace size
            simplify=True,      # Simplify model
        )
        
        # Check if engine was created
        engine_path = Path(output_path)
        if not engine_path.exists():
            # Ultralytics creates engine with different name
            engine_path = Path(model_path).with_suffix('.engine')
        
        if engine_path.exists():
            size_mb = engine_path.stat().st_size / 1e6
            logger.info("")
            logger.info("=" * 60)
            logger.info("TensorRT Engine Created Successfully!")
            logger.info("=" * 60)
            logger.info(f"Engine path: {engine_path}")
            logger.info(f"Engine size: {size_mb:.1f} MB")
            logger.info("")
            logger.info("Update audio_vision_fusion.py:")
            logger.info(f'  YOLO_MODEL_PATH = "{engine_path.name}"')
            logger.info("=" * 60)
            return True
        else:
            logger.error("Engine file not found after export!")
            logger.error("Check for errors above.")
            return False
            
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        logger.error("")
        logger.error("Troubleshooting:")
        logger.error("1. Ensure CUDA/TensorRT is installed")
        logger.error("2. Check GPU memory is available")
        logger.error("3. Try reducing workspace size")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export YOLOv10 to TensorRT')
    parser.add_argument('--model', type=str, default='yolov10n.pt',
                       help='Input YOLO model path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output engine path (default: model.engine)')
    parser.add_argument('--workspace', type=int, default=4,
                       help='TensorRT workspace size in GB (default: 4)')
    parser.add_argument('--fp32', action='store_true',
                       help='Use FP32 precision (default: FP16)')
    
    args = parser.parse_args()
    
    success = export_to_tensorrt(
        model_path=args.model,
        output_path=args.output,
        workspace=args.workspace,
        half=not args.fp32
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()


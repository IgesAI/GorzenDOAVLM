"""
YOLO Model Benchmark Script
===========================

Benchmarks different YOLO models to compare performance, GPU usage, and speed.
Helps determine which model is best for your hardware and use case.
"""

import time
import numpy as np
import torch
from pathlib import Path
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics not installed. Run: pip install ultralytics")
    exit(1)


class YOLOBenchmark:
    """Benchmark YOLO models for performance comparison."""
    
    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        """Initialize benchmark with frame dimensions.
        
        Args:
            frame_width: Width of test frames
            frame_height: Height of test frames
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("CUDA not available, using CPU (will be slow)")
    
    def create_dummy_frame(self) -> np.ndarray:
        """Create a dummy frame for testing."""
        return np.random.randint(0, 255, (self.frame_height, self.frame_width, 3), dtype=np.uint8)
    
    def benchmark_model(self, model_path: str, warmup_iterations: int = 5, 
                       test_iterations: int = 100) -> Dict[str, float]:
        """Benchmark a YOLO model.
        
        Args:
            model_path: Path to YOLO model file
            warmup_iterations: Number of warmup iterations
            test_iterations: Number of test iterations
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {model_path}")
        logger.info(f"{'='*60}")
        
        # Check if model exists
        if not Path(model_path).exists() and not model_path.startswith('http'):
            logger.warning(f"Model not found: {model_path}")
            logger.info("Attempting to download...")
        
        try:
            model = YOLO(model_path)
            logger.info(f"Model loaded successfully")
            
            # Get model info
            dummy_frame = self.create_dummy_frame()
            
            # Warmup
            logger.info(f"Warming up ({warmup_iterations} iterations)...")
            for _ in range(warmup_iterations):
                _ = model(dummy_frame, verbose=False)
            
            # Clear GPU memory stats
            if self.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            # Benchmark
            logger.info(f"Running benchmark ({test_iterations} iterations)...")
            start_time = time.time()
            
            for _ in range(test_iterations):
                _ = model(dummy_frame, verbose=False)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            fps = test_iterations / elapsed_time
            avg_time_ms = (elapsed_time / test_iterations) * 1000
            
            # Get GPU memory usage
            if self.device == 'cuda':
                peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
                current_memory_gb = torch.cuda.memory_allocated() / 1e9
            else:
                peak_memory_gb = 0
                current_memory_gb = 0
            
            # Get model size
            model_file = Path(model_path)
            model_size_mb = model_file.stat().st_size / 1e6 if model_file.exists() else 0
            
            results = {
                'model_path': model_path,
                'fps': fps,
                'avg_time_ms': avg_time_ms,
                'peak_memory_gb': peak_memory_gb,
                'current_memory_gb': current_memory_gb,
                'model_size_mb': model_size_mb,
                'device': self.device
            }
            
            logger.info(f"Results:")
            logger.info(f"  FPS: {fps:.1f}")
            logger.info(f"  Avg Time: {avg_time_ms:.2f} ms")
            logger.info(f"  Peak VRAM: {peak_memory_gb:.2f} GB")
            logger.info(f"  Model Size: {model_size_mb:.1f} MB")
            
            # Cleanup
            del model
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"Error benchmarking {model_path}: {e}")
            return None
    
    def compare_models(self, model_paths: List[str]) -> None:
        """Compare multiple models.
        
        Args:
            model_paths: List of model paths to compare
        """
        results = []
        
        for model_path in model_paths:
            result = self.benchmark_model(model_path)
            if result:
                results.append(result)
        
        # Print comparison table
        self.print_comparison_table(results)
    
    def print_comparison_table(self, results: List[Dict[str, float]]) -> None:
        """Print a comparison table of results.
        
        Args:
            results: List of benchmark results
        """
        if not results:
            logger.warning("No results to compare")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'='*80}")
        logger.info(f"{'Model':<20} {'FPS':<10} {'Time (ms)':<12} {'VRAM (GB)':<12} {'Size (MB)':<12}")
        logger.info("-" * 80)
        
        for result in results:
            model_name = Path(result['model_path']).stem
            logger.info(f"{model_name:<20} {result['fps']:<10.1f} {result['avg_time_ms']:<12.2f} "
                       f"{result['peak_memory_gb']:<12.2f} {result['model_size_mb']:<12.1f}")
        
        # Find best models
        logger.info(f"\n{'='*80}")
        logger.info("RECOMMENDATIONS")
        logger.info(f"{'='*80}")
        
        fastest = max(results, key=lambda x: x['fps'])
        most_efficient = min(results, key=lambda x: x['peak_memory_gb'])
        smallest = min(results, key=lambda x: x['model_size_mb'])
        
        logger.info(f"Fastest: {Path(fastest['model_path']).stem} ({fastest['fps']:.1f} FPS)")
        logger.info(f"Most Memory Efficient: {Path(most_efficient['model_path']).stem} "
                   f"({most_efficient['peak_memory_gb']:.2f} GB)")
        logger.info(f"Smallest Model: {Path(smallest['model_path']).stem} "
                   f"({smallest['model_size_mb']:.1f} MB)")


def main():
    """Run benchmark on common YOLO models."""
    print("YOLO Model Benchmark Tool")
    print("=" * 60)
    print()
    
    # Default models to test
    models_to_test = [
        'yolov8n.pt',      # Current model
        'yolov10n.pt',     # Newest nano
        'yolov8s.pt',      # Small (for comparison)
        'yolov5n.pt',      # YOLOv5 nano
    ]
    
    print("Testing models:")
    for model in models_to_test:
        print(f"  - {model}")
    print()
    
    benchmark = YOLOBenchmark(frame_width=1280, frame_height=720)
    
    # Test each model
    benchmark.compare_models(models_to_test)
    
    print("\n" + "=" * 60)
    print("Note: Models will be auto-downloaded if not present")
    print("=" * 60)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Run training for the Golf Card Game AI with optimized settings.
This script provides a convenient way to start training with the right parameters.
"""

import os
import sys
import argparse
import torch
import subprocess
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Run Golf Card Game AI training with optimized settings')
    parser.add_argument('--episodes', type=int, default=20000, help='Number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size of the neural network')
    parser.add_argument('--eval-interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load model from')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--profile', action='store_true', help='Enable profiling to identify bottlenecks')
    parser.add_argument('--cuda-optimize', action='store_true', help='Apply additional CUDA optimizations')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of worker processes for environment (0=auto)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create timestamped directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"training_run_{timestamp}"
    logs_dir = os.path.join("logs", run_name)
    models_dir = os.path.join("models", run_name)
    
    # Ensure directories exist
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Starting training run: {run_name}")
    print(f"Logs will be saved to: {logs_dir}")
    print(f"Models will be saved to: {models_dir}")
    
    # Detect hardware and set appropriate batch size
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        print(f"CUDA GPU detected: {gpu_name} with {gpu_memory:.2f} GB memory")
        
        # Adjust batch size based on GPU memory
        if gpu_memory >= 16:
            batch_size = 512
        elif gpu_memory >= 8:
            batch_size = 256
        else:
            batch_size = 128
            
        print(f"Automatically set batch size to {batch_size} based on GPU memory")
        
        # Apply CUDA optimizations if requested
        if args.cuda_optimize:
            print("Applying additional CUDA optimizations:")
            
            # Set environment variables for CUDA optimization
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Asynchronous kernel launches
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Memory allocator optimization
            
            # Enable TF32 for faster computation on Ampere GPUs
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                print("- Enabling TensorFloat-32 (TF32) for faster computation")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Set cuDNN benchmark mode
            print("- Enabling cuDNN benchmark mode")
            torch.backends.cudnn.benchmark = True
            
            # Disable debugging features for better performance
            print("- Disabling PyTorch debugging features")
            torch.autograd.set_detect_anomaly(False)
            torch.autograd.profiler.profile(enabled=False)
            torch.autograd.profiler.emit_nvtx(enabled=False)
            
            # Set number of threads for CPU operations
            import multiprocessing
            num_cpu = multiprocessing.cpu_count()
            torch.set_num_threads(num_cpu)
            print(f"- Setting PyTorch to use {num_cpu} CPU threads")
            
            # Set worker processes if not specified
            if args.num_workers == 0:
                args.num_workers = min(4, num_cpu)
                print(f"- Setting number of worker processes to {args.num_workers}")
    elif torch.backends.mps.is_available():
        print("Apple MPS (Metal Performance Shaders) detected")
        batch_size = 128
        print(f"Set batch size to {batch_size} for MPS")
    else:
        print("Running on CPU")
        batch_size = 64
        print(f"Set batch size to {batch_size} for CPU")
    
    # Build command to run training
    cmd = [
        sys.executable, 
    ]
    
    # Add profiling if requested
    if args.profile:
        print("Enabling profiling to identify bottlenecks")
        # Use PyTorch profiler
        cmd.extend(["-m", "torch.utils.bottleneck"])
        
    cmd.extend([
        "train.py",
        "--episodes", str(args.episodes),
        "--batch-size", str(batch_size),
        "--hidden-size", str(args.hidden_size),
        "--eval-interval", str(args.eval_interval),
        "--save-dir", models_dir,
        "--logs-dir", logs_dir,
        "--log-level", args.log_level
    ])
    
    # Add load model if specified
    if args.load_model:
        cmd.extend(["--load-model", args.load_model])
    
    # Add number of workers if specified
    if args.num_workers > 0:
        cmd.extend(["--num-workers", str(args.num_workers)])
    
    # Run the training process
    print("Starting training with command:", " ".join(cmd))
    
    # Create a file to store profiling information
    if args.profile:
        profile_file = os.path.join(logs_dir, "profile_output.txt")
        with open(profile_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f"Profiling results saved to {profile_file}")
    else:
        subprocess.run(cmd)
    
    print(f"Training complete. Results saved to {logs_dir} and {models_dir}")

if __name__ == "__main__":
    main() 
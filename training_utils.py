import os
import datetime
import logging
from typing import Tuple, Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class RunDirectories:
    """Dataclass to store paths to training run directories."""
    run_dir: str
    logs_dir: str
    models_dir: str
    charts_dir: str

def get_run_dir(algorithm_name: str = 'run') -> str:
    """
    Generate a unique run directory name with timestamp.
    
    Args:
        algorithm_name: Name of the algorithm (e.g., 'dqn', 'ppo')
        
    Returns:
        Path to the run directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('runs', f'{algorithm_name}_run_{timestamp}')
    return run_dir

def setup_logging(log_level: str, run_dir: str) -> Tuple[logging.Logger, str]:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (e.g., 'INFO', 'DEBUG')
        run_dir: Directory to store logs
        
    Returns:
        Tuple containing the logger and the path to the logs directory
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create logs directory within run directory
    logs_dir = os.path.join(run_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamped log file
    log_file = os.path.join(logs_dir, "training.log")
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Get logger
    logger = logging.getLogger()
    
    return logger, logs_dir

def setup_run_directories(run_dir: str) -> RunDirectories:
    """
    Create all necessary directories for a training run.
    
    Args:
        run_dir: Base run directory
        
    Returns:
        RunDirectories object containing paths to different directories
    """
    # Make sure the run directory exists
    os.makedirs(run_dir, exist_ok=True)
    
    # Create logs directory
    logs_dir = os.path.join(run_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create models directory
    models_dir = os.path.join(run_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create charts directory
    charts_dir = os.path.join(run_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    return RunDirectories(
        run_dir=run_dir,
        logs_dir=logs_dir,
        models_dir=models_dir,
        charts_dir=charts_dir
    )

def calculate_ema(data, alpha=0.1):
    """
    Calculate Exponential Moving Average with the given alpha.
    
    Args:
        data: Array-like data to calculate EMA for
        alpha: Smoothing factor (higher = less smoothing)
        
    Returns:
        Array with EMA values
    """
    ema = np.zeros_like(data, dtype=float)
    if len(data) > 0:
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema 
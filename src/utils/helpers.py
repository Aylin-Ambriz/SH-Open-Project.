import os
import torch
from pathlib import Path
from tqdm.auto import tqdm

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def setup_directories():
    root = get_project_root()
    dirs = [
        'data',
        'models',
        'notebooks',
        'src/preprocessing',
        'src/training',
        'src/utils'
    ]
    for dir_path in tqdm(dirs, desc = "Creating directories"):
        (root / dir_path).mkdir(parents=True, exist_ok=True)

def check_cuda_memory():
    """Check available CUDA memory"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = total_memory - allocated_memory
        
        print("\nCUDA Memory Status:")
        print(f"Total: {total_memory / 1e9:.2f} GB")
        print(f"Reserved: {reserved_memory / 1e9:.2f} GB")
        print(f"Allocated: {allocated_memory / 1e9:.2f} GB")
        print(f"Free: {free_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available")

def optimize_cuda_settings():
    """Optimize CUDA settings for training"""
    if torch.cuda.is_available():
        # Enable cudnn autotuner
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    return False
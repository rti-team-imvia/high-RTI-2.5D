import numpy as np
import sys

def check_gpu_availability():
    """
    Check if a GPU is available through PyTorch.
    
    Returns:
        tuple: (is_available, device_name, cuda_version, device)
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            device = torch.device("cuda")
            print(f"GPU available: {device_name}")
            print(f"CUDA version: {cuda_version}")
            return True, device_name, cuda_version, device
        else:
            print("CUDA is available but no GPU detected, using CPU only")
            return False, None, None, torch.device("cpu")
    except (ImportError, ModuleNotFoundError):
        print("PyTorch not installed or CUDA not available, using CPU only")
        return False, None, None, "cpu"

def gpu_enabled_bilateral_filter(depth_map, d=7, sigma_color=0.05, sigma_space=2.0):
    """
    Apply bilateral filtering with GPU acceleration if available.
    
    Args:
        depth_map: 2D numpy array with depth values
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        numpy.ndarray: Filtered depth map
    """
    import cv2
    
    try:
        import cupy as cp
        # Check if cupy is available and initialized
        if cp.is_available() and cp.cuda.runtime.getDeviceCount() > 0:
            # Convert numpy array to cupy array
            depth_cupy = cp.asarray(depth_map.astype(np.float32))
            
            # Use OpenCV's CUDA implementation if available
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_depth = cv2.cuda_GpuMat()
                gpu_depth.upload(depth_map.astype(np.float32))
                gpu_result = cv2.cuda.bilateralFilter(gpu_depth, d, sigma_color, sigma_space)
                result = gpu_result.download()
                print("Used CUDA-accelerated bilateral filter")
                return result
    except (ImportError, ModuleNotFoundError, Exception) as e:
        pass
    
    # Fall back to CPU implementation
    result = cv2.bilateralFilter(
        depth_map.astype(np.float32), 
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )
    return result
import numpy as np
import cv2
import json
import os
import open3d as o3d
from camera.intrinsics import CameraIntrinsics

def load_depth_map(file_path):
    """
    Load a depth map from an image file.
    
    Args:
        file_path: Path to the depth map image
        
    Returns:
        numpy.ndarray: Depth map as a 2D array
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Depth map file not found: {file_path}")
    
    # For 32-bit TIFF files, use a specialized loader
    if file_path.lower().endswith('.tiff') or file_path.lower().endswith('.tif'):
        try:
            # Try using tifffile for better 32-bit TIFF support
            import tifffile
            depth_map = tifffile.imread(file_path)
            print(f"Loaded 32-bit TIFF with tifffile: dtype={depth_map.dtype}, shape={depth_map.shape}")
            
            # If multi-channel, take the first channel
            if len(depth_map.shape) > 2:
                depth_map = depth_map[:, :, 0]
            
            # For 32-bit float TIFFs, assume values are already in metric units
            if depth_map.dtype == np.float32 or depth_map.dtype == np.float64:
                print(f"Detected floating-point depth map, assuming values are already in meters")
                # Convert to float32 for consistency
                depth_map = depth_map.astype(np.float32)
            return depth_map
            
        except ImportError:
            print("tifffile module not found, falling back to OpenCV (may not handle 32-bit TIFFs correctly)")
    
    # Standard handling for other formats or fallback
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
        # Read as image file with unchanged bit depth
        depth_map = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        if depth_map is None:
            raise ValueError(f"Failed to read depth map: {file_path}. File may be corrupted or in an unsupported format.")
        
        # If image has multiple channels, convert to single channel
        if len(depth_map.shape) > 2:
            # Use the first channel or convert to grayscale
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
        
        # Process based on bit depth
        print(f"Loaded depth map with dtype: {depth_map.dtype}, shape: {depth_map.shape}")
        
        # Handle based on data type
        if depth_map.dtype == np.float32 or depth_map.dtype == np.float64:
            # For floating point, assume values are already in meters
            print("Detected floating-point depth map, assuming values are already in meters")
            # Just ensure float32 for consistency
            depth_map = depth_map.astype(np.float32)
        elif depth_map.dtype == np.uint16:
            # 16-bit depth, normalize to meters assuming typical depth sensor range
            depth_map = depth_map.astype(np.float32) / 1000.0  # Convert mm to meters
            print("Converted 16-bit depth map from mm to meters")
        elif depth_map.dtype == np.uint8:
            # 8-bit depth, this is likely not a true depth map but just for visualization
            depth_map = depth_map.astype(np.float32) / 50.0  # Rough conversion
            print("Converted 8-bit depth map to approximate metric scale")
    elif file_path.endswith('.npy'):
        # Load as numpy array
        depth_map = np.load(file_path)
        print(f"Loaded .npy file with dtype: {depth_map.dtype}, shape: {depth_map.shape}")
    else:
        raise ValueError(f"Unsupported file format for depth map: {file_path}")
    
    return depth_map

def load_mask(file_path):
    """
    Load a binary mask from an image file.
    
    Args:
        file_path: Path to the mask image
        
    Returns:
        numpy.ndarray: Binary mask as a 2D boolean array
    """
    if not os.path.exists(file_path):
        return None
        
    # Read image
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert to binary mask
    mask = mask > 0
    
    return mask

def load_normal_map(file_path):
    """
    Load a normal map from an image file.
    
    Args:
        file_path: Path to the normal map image
        
    Returns:
        numpy.ndarray: Normal map as a 3-channel array
    """
    if not os.path.exists(file_path):
        return None
        
    # Read normal map (BGR format)
    normal_map = cv2.imread(file_path, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB
    normal_map = cv2.cvtColor(normal_map, cv2.COLOR_BGR2RGB)
    
    # Normalize to [-1, 1] range
    normal_map = normal_map.astype(np.float32) / 127.5 - 1.0
    
    # Ensure unit length normals
    magnitude = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
    normal_map = normal_map / (magnitude + 1e-10)
    
    return normal_map

def save_volume(volume, file_path):
    """
    Save a volumetric representation to a file.
    
    Args:
        volume: Dictionary containing volumetric data
        file_path: Path where to save the volume
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Get the mesh from the volume
    mesh = volume["mesh"]
    
    # Save as OBJ file if file_path is an OBJ file
    if file_path.lower().endswith('.obj'):
        o3d.io.write_triangle_mesh(file_path, mesh)
    elif file_path.lower().endswith('.ply'):
        o3d.io.write_triangle_mesh(file_path, mesh)
    else:
        # Default to PLY if not specified
        ply_path = file_path if file_path.lower().endswith('.ply') else f"{os.path.splitext(file_path)[0]}.ply"
        o3d.io.write_triangle_mesh(ply_path, mesh)
        
        # Also save the point cloud
        pcd_path = f"{os.path.splitext(file_path)[0]}_points.ply"
        o3d.io.write_point_cloud(pcd_path, volume["point_cloud"])

def load_intrinsics(file_path):
    """
    Load camera intrinsics from a JSON file.
    
    Args:
        file_path: Path to the intrinsics JSON file
        
    Returns:
        CameraIntrinsics: Camera intrinsics object
    """
    if not os.path.exists(file_path):
        # Return default intrinsics
        return CameraIntrinsics()
        
    with open(file_path, 'r') as f:
        intrinsics_dict = json.load(f)
        
    return CameraIntrinsics.from_dict(intrinsics_dict)

def save_intrinsics(intrinsics, file_path):
    """
    Save camera intrinsics to a JSON file.
    
    Args:
        intrinsics: CameraIntrinsics object
        file_path: Path to save the intrinsics JSON file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(intrinsics.to_dict(), f, indent=4)
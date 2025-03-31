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

def load_normal_map(file_path, convert_bgr_to_rgb=True, flip_y=True, flip_z=True):
    """
    Load a normal map from an image file and convert to proper [-1,1] range.
    Handles different normal map conventions.
    
    Args:
        file_path: Path to the normal map image
        convert_bgr_to_rgb: Whether to convert from BGR to RGB color order
        flip_y: Whether to flip the Y component
        flip_z: Whether to flip the Z component
        
    Returns:
        numpy.ndarray: Normal map as a 3-channel array in range [-1,1]
    """
    if not os.path.exists(file_path):
        return None
        
    # Read normal map (BGR format in OpenCV)
    normal_map = cv2.imread(file_path, cv2.IMREAD_COLOR)
    
    if normal_map is None:
        print(f"Warning: Failed to load normal map from {file_path}")
        return None
    
    # Normal maps typically store XYZ in RGB channels
    # But OpenCV loads as BGR, so we need to convert if we want RGB=XYZ
    if convert_bgr_to_rgb:
        print("Converting normal map from BGR to RGB")
        normal_map = cv2.cvtColor(normal_map, cv2.COLOR_BGR2RGB)
    else:
        print("Keeping normal map in BGR color order (B=X, G=Y, R=Z)")
    
    # Normalize from [0,255] to [-1,1] range
    # Normal maps typically store (0,0,1) as (128,128,255)
    normal_map = normal_map.astype(np.float32) / 127.5 - 1.0
    
    # Flip components if needed due to coordinate system differences
    if flip_y:
        print("Flipping Y component of normal map")
        normal_map[:, :, 1] = -normal_map[:, :, 1]
    
    if flip_z:
        print("Flipping Z component of normal map")
        normal_map[:, :, 2] = -normal_map[:, :, 2]
    
    # Ensure unit length normals
    magnitude = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
    # Avoid division by zero
    magnitude = np.maximum(magnitude, 1e-10)
    normal_map = normal_map / magnitude
    
    print(f"Loaded normal map with shape {normal_map.shape}")
    print(f"Normal map range: X [{normal_map[:,:,0].min():.3f}, {normal_map[:,:,0].max():.3f}], "
          f"Y [{normal_map[:,:,1].min():.3f}, {normal_map[:,:,1].max():.3f}], "
          f"Z [{normal_map[:,:,2].min():.3f}, {normal_map[:,:,2].max():.3f}]")
    
    return normal_map

def transform_normals_to_world_space(normal_map, intrinsics):
    """
    Transform normals from camera space to world space.
    
    Args:
        normal_map: Normal map in camera space (H, W, 3)
        intrinsics: Camera intrinsics object
        
    Returns:
        numpy.ndarray: Transformed normals in world space
    """
    height, width = normal_map.shape[:2]
    transformed_normals = np.copy(normal_map)
    
    # Create coordinate grid
    v, u = np.indices((height, width))
    
    # Calculate ray directions for each pixel
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy
    
    # Calculate ray direction in camera space
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)
    
    # Normalize ray directions
    ray_lengths = np.sqrt(x**2 + y**2 + 1)
    x /= ray_lengths
    y /= ray_lengths
    z /= ray_lengths
    
    # Construct local coordinate frames for each pixel
    # Use ray direction as Z axis
    z_axis = np.stack([x, y, z], axis=-1)
    
    # Create orthogonal vectors for X and Y axes
    # We can use cross product with [0,1,0] for X axis
    # and then cross product of Z and X for Y axis
    up_vector = np.array([0, 1, 0])
    x_axis = np.zeros_like(z_axis)
    y_axis = np.zeros_like(z_axis)
    
    for i in range(height):
        for j in range(width):
            z_dir = z_axis[i, j]
            
            # Handle case where ray is parallel to up vector
            if abs(np.dot(z_dir, up_vector)) > 0.99:
                # Use alternative up vector
                up_vector = np.array([1, 0, 0])
                
            x_dir = np.cross(up_vector, z_dir)
            x_dir = x_dir / np.linalg.norm(x_dir)
            y_dir = np.cross(z_dir, x_dir)
            y_dir = y_dir / np.linalg.norm(y_dir)
            
            x_axis[i, j] = x_dir
            y_axis[i, j] = y_dir
    
    # Transform normals from camera space to world space
    # The normal in camera space is defined in relation to the pixel's viewing direction
    # We need to transform it using the local tangent space
    world_normals = np.zeros_like(normal_map)
    
    for i in range(height):
        for j in range(width):
            # Get the normal in tangent space
            normal = normal_map[i, j]
            
            # Create rotation matrix from tangent space to world space
            rotation = np.column_stack([x_axis[i, j], y_axis[i, j], z_axis[i, j]])
            
            # Apply rotation
            world_normal = rotation @ normal
            
            # Ensure normal is unit length
            world_normals[i, j] = world_normal / np.linalg.norm(world_normal)
    
    return world_normals



def transform_normals_to_world_space_gpu(normal_map, intrinsics):
    """
    GPU-accelerated transformation of normals from camera space to world space.
    Fully vectorized implementation for maximum GPU performance.
    
    Args:
        normal_map: Normal map in camera space (H, W, 3)
        intrinsics: Camera intrinsics object
        
    Returns:
        numpy.ndarray: Transformed normals in world space
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU for normal transformation...")
        else:
            device = torch.device("cpu")
            print("No GPU available, using CPU for normal transformation...")
    except ImportError:
        print("PyTorch not installed, using CPU implementation...")
        return transform_normals_to_world_space(normal_map, intrinsics)
    
    height, width = normal_map.shape[:2]
    
    # Move data to GPU
    normal_tensor = torch.tensor(normal_map, device=device, dtype=torch.float32)
    
    # Create coordinate grid
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij'  # Specify indexing to avoid warning
    )
    
    # Calculate ray directions for each pixel (vectorized)
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy
    
    # Calculate ray direction in camera space
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = torch.ones_like(x, device=device)
    
    # Normalize ray directions (vectorized)
    ray_lengths = torch.sqrt(x**2 + y**2 + 1)
    x = x / ray_lengths
    y = y / ray_lengths
    z = z / ray_lengths
    
    # Stack for z_axis
    z_axis = torch.stack([x, y, z], dim=-1)
    
    # Create up vector tensor (broadcast to all pixels)
    up_vector = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    up_vector = up_vector.reshape(1, 1, 3).expand(height, width, 3)
    
    # Handle special case where ray is parallel to up vector
    # Compute dot product between z_axis and up_vector
    dot_product = (z_axis * up_vector).sum(dim=-1, keepdim=True)
    parallel_mask = torch.abs(dot_product) > 0.99
    
    # Create alternative up vector for parallel cases
    alt_up_vector = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    alt_up_vector = alt_up_vector.reshape(1, 1, 3).expand(height, width, 3)
    
    # Apply mask to select between primary and alternative up vectors
    effective_up = torch.where(parallel_mask.expand(-1, -1, 3), alt_up_vector, up_vector)
    
    # Compute x_axis using cross product (vectorized)
    x_axis = torch.linalg.cross(effective_up, z_axis, dim=-1)
    x_axis_norm = torch.norm(x_axis, dim=-1, keepdim=True)
    x_axis = x_axis / (x_axis_norm + 1e-10)  # Avoid division by zero
    
    # Compute y_axis using cross product (vectorized)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis_norm = torch.norm(y_axis, dim=-1, keepdim=True)
    y_axis = y_axis / (y_axis_norm + 1e-10)  # Avoid division by zero
    
    # Create rotation matrices for all pixels (H×W×3×3)
    rotation_matrices = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    
    # Reshape normal_tensor to allow for batch matrix multiplication
    normals_reshaped = normal_tensor.reshape(height * width, 3, 1)
    rotation_matrices_reshaped = rotation_matrices.reshape(height * width, 3, 3)
    
    # Perform batch matrix multiplication (H*W matrix multiplications at once)
    world_normals_reshaped = torch.bmm(rotation_matrices_reshaped, normals_reshaped)
    
    # Reshape and normalize
    world_normals = world_normals_reshaped.reshape(height, width, 3)
    world_normals_norm = torch.norm(world_normals, dim=-1, keepdim=True)
    world_normals = world_normals / (world_normals_norm + 1e-10)
    
    # Return as numpy array
    return world_normals.cpu().numpy()

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
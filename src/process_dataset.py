import os
import sys
import numpy as np
import time
import cv2
import open3d as o3d
import glob
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.depth_processor import DepthProcessor
from processing.volume_generator import VolumeGenerator
from utils.file_io import *
from utils.visualization import *
from utils.gpu_utils import *
from camera.intrinsics import CameraIntrinsics

def process_dataset(input_dir, dataset_name, point_cloud_dir):
    """
    Process a single dataset and save the point cloud with PBR properties
    
    Args:
        input_dir: Directory containing the dataset files
        dataset_name: Name of the dataset (for output filename)
        point_cloud_dir: Directory to save the point cloud PLY file
    
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Define input files
    depth_map_path = os.path.join(input_dir, 'depth.tiff')
    mask_path = os.path.join(input_dir, 'mask.png')
    normal_map_path = os.path.join(input_dir, 'normal.png')
    color_map_path = os.path.join(input_dir, 'baseColor.png')
    
    # Skip if depth map doesn't exist
    if not os.path.exists(depth_map_path):
        print(f"Error: Depth map not found at {depth_map_path}")
        return False
    
    # Load camera intrinsics (default values used if file not found)
    intrinsics_path = os.path.join(input_dir, 'intrinsics.json')
    camera_intrinsics = load_intrinsics(intrinsics_path)
    
    print(f"Camera intrinsics: fx={camera_intrinsics.fx}, fy={camera_intrinsics.fy}, "
          f"cx={camera_intrinsics.cx}, cy={camera_intrinsics.cy}")
    
    # Load the depth map
    print(f"Loading depth map from {depth_map_path}")
    depth_map = load_depth_map(depth_map_path)
    print(f"Depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}, min: {depth_map.min()}, max: {depth_map.max()}")
    
    # Load mask if available
    mask = None
    if os.path.exists(mask_path):
        print(f"Loading mask from {mask_path}")
        mask = load_mask(mask_path)
        print(f"Mask shape: {mask.shape}, valid pixels: {np.sum(mask)}")
    
    # Load normal map if available
    normal_map = None
    if os.path.exists(normal_map_path):
        print(f"Loading normal map from {normal_map_path}")
        normal_map = load_normal_map(normal_map_path)
        if normal_map is not None:
            print(f"Normal map shape: {normal_map.shape}")
            print("Transforming normals from camera space to world space...")
            normal_map = transform_normals_to_world_space_gpu(normal_map, camera_intrinsics)
    
    # Load color image if available
    color_map = None
    if os.path.exists(color_map_path):
        print(f"Loading color texture from {color_map_path}")
        color_map = cv2.imread(color_map_path)
        if color_map is not None:
            color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            print(f"Color map shape: {color_map.shape}")
        else:
            print(f"Failed to load color texture from {color_map_path}")
    
    # Load roughness map if available
    roughness_map = None
    roughness_path = os.path.join(input_dir, 'roughness.png')
    if os.path.exists(roughness_path):
        print(f"Loading roughness map from {roughness_path}")
        roughness_map = cv2.imread(roughness_path, cv2.IMREAD_GRAYSCALE)
        if roughness_map is not None:
            print(f"Roughness map shape: {roughness_map.shape}")
            # Normalize to [0,1] range if needed
            if roughness_map.dtype == np.uint8:
                roughness_map = roughness_map.astype(np.float32) / 255.0
        else:
            print(f"Failed to load roughness map from {roughness_path}")
            
    # Load metallic map if available
    metallic_map = None
    metallic_path = os.path.join(input_dir, 'metallic.png')
    if os.path.exists(metallic_path):
        print(f"Loading metallic map from {metallic_path}")
        metallic_map = cv2.imread(metallic_path, cv2.IMREAD_GRAYSCALE)
        if metallic_map is not None:
            print(f"Metallic map shape: {metallic_map.shape}")
            # Normalize to [0,1] range if needed
            if metallic_map.dtype == np.uint8:
                metallic_map = metallic_map.astype(np.float32) / 255.0
        else:
            print(f"Failed to load metallic map from {metallic_path}")
    
    # Process the depth map using the normal map for correction
    print("Processing depth map...")
    depth_processor = DepthProcessor()
    processed_depth = depth_processor.process_metric_depth(
        depth_map, 
        mask
    )
    
    # Generate the volumetric representation
    print("Generating volumetric representation...")
    volume_generator = VolumeGenerator(camera_intrinsics, selective_gpu=True)
    volume = volume_generator.create_volume_from_metric_depth(
        processed_depth,
        mask, 
        normal_map=normal_map,
        color_map=color_map,
        roughness_map=roughness_map,
        metallic_map=metallic_map,
        voxel_size=0.001
    )
    
    # Update material properties after point cloud filtering
    if 'material_properties' in volume and volume["point_cloud"] is not None:
        # Check and fix material property sizes to match point count
        num_points = len(volume["point_cloud"].points)
        for prop_name in list(volume['material_properties'].keys()):
            if len(volume['material_properties'][prop_name]) != num_points:
                print(f"Warning: Material property {prop_name} size doesn't match point count.")
                print(f"Property has {len(volume['material_properties'][prop_name])} values but point cloud has {num_points} points.")
                volume['material_properties'].pop(prop_name)
                print(f"Removed {prop_name} property - properties will not be included in the PLY file")
    
    # Apply normal-based coloring if needed
    if not volume["point_cloud"].has_colors():
        print("No color data available, coloring by normals instead")
        volume["point_cloud"] = color_point_cloud_by_normals(volume["point_cloud"])
    else:
        print("Using original colors from texture for point cloud")
    
    # Save the extended PLY with material properties
    ply_filename = f"point_cloud_pbr_{dataset_name}.ply"
    pcd_output_path = os.path.join(point_cloud_dir, ply_filename)
    print(f"Saving point cloud with PBR materials to {pcd_output_path}")
    
    save_extended_ply_with_plyfile(
        volume["point_cloud"],
        pcd_output_path,
        material_properties=volume.get("material_properties", {})
    )
    
    print(f"Dataset {dataset_name} processing completed successfully!")
    return True


def main():
    # Set base directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_base_dir = os.path.join(base_dir, 'data', 'input')
    point_cloud_dir = os.path.join(base_dir, 'data', 'point_cloud')
    
    # Create output directory if it doesn't exist
    os.makedirs(point_cloud_dir, exist_ok=True)
    
    # Start time
    start_time = time.time()
    
    # Process the main input folder itself
    if os.path.exists(os.path.join(input_base_dir, 'depth.tiff')):
        process_dataset(input_base_dir, "main", point_cloud_dir)
    
    # Get all subdirectories that look like datasets (containing depth.tiff)
    # First try numbered subdirectories format (1.data, 2.data, etc.)
    dataset_dirs = sorted(glob.glob(os.path.join(input_base_dir, "*.data")))
    
    # If no numbered directories found, look for any subdirectory with depth.tiff
    if not dataset_dirs:
        # Get all subdirectories
        for root, dirs, _ in os.walk(input_base_dir):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                # Check if this directory contains a depth map
                if os.path.exists(os.path.join(dir_path, 'depth.tiff')):
                    dataset_dirs.append(dir_path)
    
    # Process each dataset
    print(f"Found {len(dataset_dirs)} datasets to process")
    
    processed_count = 0
    for dataset_dir in dataset_dirs:
        # Extract the dataset name from the directory path
        dataset_name = os.path.basename(dataset_dir).split('.')[0]
        
        # Process the dataset
        if process_dataset(dataset_dir, dataset_name, point_cloud_dir):
            processed_count += 1
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Batch processing completed!")
    print(f"Processed {processed_count} out of {len(dataset_dirs)} datasets")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"All point clouds saved to: {point_cloud_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
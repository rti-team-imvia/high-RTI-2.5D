import os
import sys
import numpy as np
import time
import cv2
import open3d as o3d  # Make sure Open3D is imported
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.depth_processor import DepthProcessor
from processing.volume_generator import VolumeGenerator
from utils.file_io import *
from utils.visualization import *
from camera.intrinsics import CameraIntrinsics

def main():
    # Set input and output paths
    input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'input')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define input files
    depth_map_path = os.path.join(input_dir, 'depth.tiff')
    mask_path = os.path.join(input_dir, 'mask.png')
    normal_map_path = os.path.join(input_dir, 'normal.png')  # Add this line
    color_map_path = os.path.join(input_dir, 'baseColor.png')  # Adjust filename as needed
    
    # Load camera intrinsics (default values used if file not found)
    intrinsics_path = os.path.join(input_dir, 'intrinsics.json')
    camera_intrinsics = load_intrinsics(intrinsics_path)
    
    print(f"Camera intrinsics: fx={camera_intrinsics.fx}, fy={camera_intrinsics.fy}, "
          f"cx={camera_intrinsics.cx}, cy={camera_intrinsics.cy}")
    
    # Load the depth map
    print(f"Loading depth map from {depth_map_path}")
    depth_map = load_depth_map(depth_map_path)
    print(f"Depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}, min: {depth_map.min()}, max: {depth_map.max()}")
    
    # Check if we have metric floating point data
    is_floating_point = np.issubdtype(depth_map.dtype, np.floating)
    if is_floating_point:
        print("Detected floating point depth map - treating as metric depth in meters")
    
    # Check depth orientation
    cross_section_path = os.path.join(output_dir, 'depth_cross_section.png')
    print(f"Creating depth cross-section to {cross_section_path}")
    visualize_depth_cross_section(depth_map, save_path=cross_section_path, show=False)
    
    # Load mask if available
    mask = None
    if os.path.exists(mask_path):
        print(f"Loading mask from {mask_path}")
        mask = load_mask(mask_path)
        print(f"Mask shape: {mask.shape}, valid pixels: {np.sum(mask)}")
    
    # Visualize the input depth map
    depth_vis_path = os.path.join(output_dir, 'depth_visualization.png')
    print(f"Visualizing depth map to {depth_vis_path}")
    visualize_depth_map(depth_map, save_path=depth_vis_path, show=False)
    
    # Load normal map if available
    normal_map = None
    if os.path.exists(normal_map_path):
        print(f"Loading normal map from {normal_map_path}")
        normal_map = load_normal_map(normal_map_path)
        print(f"Normal map shape: {normal_map.shape}")
        
        # Visualize the input normal map
        normal_vis_path = os.path.join(output_dir, 'normal_visualization.png')
        print(f"Visualizing normal map to {normal_vis_path}")
        visualize_normal_map(normal_map, save_path=normal_vis_path, show=False)
    
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
    
    # Process the depth map using the normal map for correction
    print("Processing depth map...")
    depth_processor = DepthProcessor()
    processed_depth = depth_processor.process_metric_depth(
        depth_map, 
        mask,
        normal_map,  # Pass the normal map to correct depth inaccuracies
        camera_intrinsics  # Pass the camera intrinsics for correct projection
    )
    
    # Visualize the processed depth map with normal enhancement
    processed_vis_path = os.path.join(output_dir, 'processed_depth_visualization.png')
    print(f"Visualizing processed depth map to {processed_vis_path}")
    visualize_depth_map(processed_depth, normal_map=normal_map, save_path=processed_vis_path, show=False)
    
    # Generate the volumetric representation
    print("Generating volumetric representation...")
    volume_generator = VolumeGenerator(camera_intrinsics, selective_gpu=True)  # Enable selective GPU
    # Add normal_map and color_map parameters
    volume = volume_generator.create_volume_from_metric_depth(
        processed_depth,
        mask, 
        normal_map=normal_map,
        color_map=color_map,  # Add this parameter
        voxel_size=0.001  # Adjust voxel size for resolution
    )
    
    # Save the volumetric mesh
    mesh_output_path = os.path.join(output_dir, 'reconstructed_mesh.obj')
    print(f"Saving volumetric mesh to {mesh_output_path}")
    save_volume(volume, mesh_output_path)
    
    # Visualize the generated volume
    vis_output_dir = os.path.join(output_dir, 'visualizations')
    print(f"Visualizing volumetric representation to {vis_output_dir}")
    visualize_volume(volume, save_dir=vis_output_dir, show=True)
    
    # Add visualization to debug camera and point positions
    camera_debug_path = os.path.join(vis_output_dir, 'camera_and_points.png')
    print(f"Creating camera-point debug visualization to {camera_debug_path}")
    visualize_camera_and_points(volume["point_cloud"], save_path=camera_debug_path, show=False)
    
    # Save point cloud with normals
    pcd_output_path = os.path.join(output_dir, 'point_cloud.ply')
    print(f"Saving point cloud with normals to {pcd_output_path}")

    # Only apply normal-based coloring if we don't have color data
    if volume["point_cloud"].has_colors():
        print("Using original colors from texture for point cloud")
    else:
        # Fall back to normal-based coloring if no colors are available
        print("No color data available, coloring by normals instead")
        volume["point_cloud"] = color_point_cloud_by_normals(volume["point_cloud"])
        print("Added colors based on normal directions")

    # Save in PLY format which preserves normals
    o3d.io.write_point_cloud(pcd_output_path, volume["point_cloud"])
    
    # Add normal vector visualization
    normal_vis_path = os.path.join(vis_output_dir, 'point_cloud_with_normals.png')
    print(f"Creating visualization of point cloud with normal vectors to {normal_vis_path}")
    visualize_point_cloud_with_normals(
        volume["point_cloud"], 
        scale=0.02,          # Adjust this to change arrow length
        sample_ratio=0.01,   # Adjust this to show more or fewer arrows
        save_path=normal_vis_path, 
        show=True
    )
    
    print("Processing completed successfully!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
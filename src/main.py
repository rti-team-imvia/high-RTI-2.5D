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
from utils.gpu_utils import *
from camera.intrinsics import CameraIntrinsics

def main():
    # Set input and output paths
    input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'input')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output visualizations directory
    vis_output_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # Define input files
    depth_map_path = os.path.join(input_dir, 'depth.tiff')
    mask_path = os.path.join(input_dir, 'mask.png')
    normal_map_path = os.path.join(input_dir, 'normal.png')
    color_map_path = os.path.join(input_dir, 'baseColor.png')
    
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
        
        # After loading the normal map:
        if normal_map is not None:
            original_normal_map = normal_map.copy()  # Store the original normals
            print("Transforming normals from camera space to world space...")
            normal_map = transform_normals_to_world_space_gpu(normal_map, camera_intrinsics)
            
            # Visualize the transformed normal map
            transformed_normal_vis_path = os.path.join(output_dir, 'transformed_normal_visualization.png')
            print(f"Visualizing transformed normal map to {transformed_normal_vis_path}")
            visualize_normal_map(normal_map, save_path=transformed_normal_vis_path, show=False)
            
            # Visualize comparison of original and transformed normal maps
            print("Visualizing comparison of original and transformed normal maps...")
            visualize_normal_map_comparison(
                original_normal_map, 
                normal_map, 
                save_path=os.path.join(output_dir, 'normal_map_comparison.png'), 
                show=False
            )
    
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
    
    # Visualize PBR material properties if all maps are available
    if color_map is not None and roughness_map is not None and metallic_map is not None:
        pbr_vis_path = os.path.join(output_dir, 'pbr_materials_visualization.png')
        print(f"Visualizing PBR materials to {pbr_vis_path}")
        visualize_pbr_materials(
            color_map, 
            roughness_map, 
            metallic_map, 
            save_path=pbr_vis_path, 
            show=False
        )
    
    # Process the depth map using the normal map for correction
    print("Processing depth map...")
    depth_processor = DepthProcessor()
    processed_depth = depth_processor.process_metric_depth(
        depth_map, 
        mask
    )
    
    # Visualize the processed depth map with normal enhancement
    processed_vis_path = os.path.join(output_dir, 'processed_depth_visualization.png')
    print(f"Visualizing processed depth map to {processed_vis_path}")
    visualize_depth_map(
        processed_depth, 
        normal_map=normal_map, 
        save_path=processed_vis_path, 
        show=False, 
        consider_flips=True
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
    
    # First, after VolumeGenerator creates the point cloud, update material properties
    if 'material_properties' in volume and volume["point_cloud"] is not None:
        # Check and fix material property sizes to match point count
        num_points = len(volume["point_cloud"].points)
        for prop_name in list(volume['material_properties'].keys()):
            if len(volume['material_properties'][prop_name]) != num_points:
                print(f"Warning: Material property {prop_name} size doesn't match point count.")
                print(f"Regenerating {prop_name} property for filtered point cloud...")
                # Remove the mismatched property
                volume['material_properties'].pop(prop_name)
     
    # Save the volumetric mesh in different formats
    print("Visualizing volumetric representation to {}".format(vis_output_dir))
    
    # 1. Save as OBJ (with MTL for materials)
    mesh_output_path = os.path.join(output_dir, 'reconstructed_mesh.obj')
    print(f"Saving volumetric mesh to {mesh_output_path}")
    save_volume(volume, mesh_output_path, save_point_cloud=False)
    
    # 2. Save as Extended PLY format for point cloud with PBR properties
    pcd_output_path = os.path.join(output_dir, 'point_cloud_pbr.ply')
    print(f"Saving point cloud with PBR materials to {pcd_output_path}")

    # Apply normal-based coloring if needed
    if not volume["point_cloud"].has_colors():
        print("No color data available, coloring by normals instead")
        volume["point_cloud"] = color_point_cloud_by_normals(volume["point_cloud"])
    else:
        print("Using original colors from texture for point cloud")

    # Save the extended PLY with material properties
    save_extended_ply_with_plyfile(
        volume["point_cloud"],
        pcd_output_path,
        material_properties=volume.get("material_properties", {})
    )
    
    # Create camera-point debug visualization
    camera_vis_path = os.path.join(vis_output_dir, 'camera_and_points.png')
    print(f"Creating camera-point debug visualization to {camera_vis_path}")
    visualize_camera_and_points(
        volume["point_cloud"], 
        save_path=camera_vis_path,
        show=False
    )
    
    # Create normal vector visualization
    normal_vis_path = os.path.join(vis_output_dir, 'point_cloud_with_normals.png')
    print(f"Creating visualization of point cloud with normal vectors to {normal_vis_path}")
    visualize_point_cloud_with_normals(
        volume["point_cloud"], 
        scale=0.01,         
        sample_ratio=0.001,  
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
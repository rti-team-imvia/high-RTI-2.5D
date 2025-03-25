import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.depth_processor import DepthProcessor
from processing.volume_generator import VolumeGenerator
from utils.file_io import load_depth_map, load_mask, load_normal_map, save_volume, load_intrinsics
from utils.visualization import visualize_depth_map, visualize_volume, visualize_depth_cross_section, visualize_camera_and_points
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
    
    # Process the depth map - minimal processing for metric depth data
    print("Processing depth map...")
    depth_processor = DepthProcessor()
    processed_depth = depth_processor.process_metric_depth(
        depth_map, 
        mask
    )
    
    # Visualize the processed depth map
    processed_vis_path = os.path.join(output_dir, 'processed_depth_visualization.png')
    print(f"Visualizing processed depth map to {processed_vis_path}")
    visualize_depth_map(processed_depth, save_path=processed_vis_path, show=False)
    
    # Generate the volumetric representation
    print("Generating volumetric representation...")
    volume_generator = VolumeGenerator(camera_intrinsics)
    # For metric depth, no need for camera offset
    volume = volume_generator.create_volume_from_metric_depth(
        processed_depth,
        mask, 
        voxel_size=0.01  # Adjust voxel size for resolution
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
    
    print("Processing completed successfully!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
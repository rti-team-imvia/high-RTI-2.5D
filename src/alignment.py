import os
import sys
import numpy as np
import copy
import time
import open3d as o3d
import glob

def preprocess_point_cloud(pcd, voxel_size=0.01, normals_radius_factor=5):
    """
    Preprocess point cloud for alignment:
    - Downsample for faster processing
    - Estimate normals (if not already available)
    
    Args:
        pcd: open3d.geometry.PointCloud
        voxel_size: Voxel size for downsampling
        normals_radius_factor: Factor to multiply voxel_size for normal estimation radius
        
    Returns:
        open3d.geometry.PointCloud: Downsampled point cloud with normals
    """
    print(":: Preprocessing point cloud...")
    # Create a copy to avoid modifying the original
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals if they don't exist
    if not pcd_down.has_normals():
        radius_normal = voxel_size * normals_radius_factor
        print(f":: Estimating normals with search radius {radius_normal}...")
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pcd_down.orient_normals_towards_camera_location([0, 0, 0])
    
    return pcd_down

def compute_fpfh_features(pcd, voxel_size, radius_feature_factor=5):
    """
    Compute FPFH features for point cloud
    
    Args:
        pcd: open3d.geometry.PointCloud with normals
        voxel_size: Voxel size used for downsampling
        radius_feature_factor: Factor to multiply voxel_size for FPFH radius
        
    Returns:
        open3d.pipelines.registration.Feature: FPFH features
    """
    radius_feature = voxel_size * radius_feature_factor
    print(f":: Computing FPFH features with search radius {radius_feature}...")
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, 
                              voxel_size, ransac_n=4, max_iterations=64000, 
                              confidence=0.999, max_correspondence_distance_factor=1.5):
    """
    Perform global registration using FPFH and RANSAC
    
    Args:
        source_down: Downsampled source point cloud
        target_down: Downsampled target point cloud
        source_fpfh: FPFH features for source point cloud
        target_fpfh: FPFH features for target point cloud
        voxel_size: Voxel size used for downsampling
        ransac_n: Number of points to use for RANSAC
        max_iterations: Maximum number of RANSAC iterations
        confidence: RANSAC confidence parameter
        max_correspondence_distance_factor: Factor to multiply voxel_size for max correspondence distance
        
    Returns:
        RegistrationResult: Initial alignment result
    """
    distance_threshold = voxel_size * max_correspondence_distance_factor
    print(f":: RANSAC registration with distance threshold {distance_threshold:.4f}...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iterations, confidence))
    
    print(f":: RANSAC completed with fitness={result.fitness:.4f}, inlier_rmse={result.inlier_rmse:.4f}")
    return result

def refine_registration(source, target, result_ransac, voxel_size, 
                      max_iterations=100, max_correspondence_distance_factor=3.0):
    """
    Refine registration using ICP
    
    Args:
        source: Source point cloud
        target: Target point cloud
        result_ransac: Initial RANSAC alignment result
        voxel_size: Voxel size used for downsampling
        max_iterations: Maximum number of ICP iterations
        max_correspondence_distance_factor: Factor to multiply voxel_size for max correspondence distance
        
    Returns:
        RegistrationResult: Refined alignment result
    """
    distance_threshold = voxel_size * max_correspondence_distance_factor
    print(f":: Refining alignment using point-to-plane ICP, distance threshold {distance_threshold:.4f}...")
    
    # Make sure both clouds have normals for point-to-plane ICP
    if not source.has_normals():
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=30))
    if not target.has_normals():
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=30))
        
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    
    print(f":: ICP refinement completed with fitness={result.fitness:.4f}, inlier_rmse={result.inlier_rmse:.4f}")
    return result

def transfer_material_properties(source_pcd, transformed_source, material_properties):
    """
    Transfer material properties to the transformed point cloud
    
    Args:
        source_pcd: Original source point cloud 
        transformed_source: Transformed source point cloud
        material_properties: Dictionary with material properties
    
    Returns:
        dict: Material properties mapped to transformed point cloud
    """
    # If there are no material properties, return empty dict
    if not material_properties:
        return {}
    
    # Create a copy of material properties
    transformed_properties = {}
    
    # Simply copy the properties since the point count and order remain the same
    # We're just applying a transformation to the same set of points
    for prop_name, prop_values in material_properties.items():
        transformed_properties[prop_name] = prop_values.copy()
    
    return transformed_properties

def align_point_clouds_fpfh_ransac(source_pcd, target_pcd, source_material_props=None, 
                                 voxel_size=0.01, visualize=True, visualize_result_path=None):
    """
    Align source point cloud to target using FPFH+RANSAC and ICP refinement
    
    Args:
        source_pcd: Source point cloud (will be transformed)
        target_pcd: Target point cloud (reference)
        source_material_props: Material properties for source point cloud
        voxel_size: Voxel size for downsampling
        visualize: Whether to visualize the alignment result
        visualize_result_path: Path to save visualization result
    
    Returns:
        tuple: (transformed point cloud, transformation matrix, transformed material properties)
    """
    start_time = time.time()
    
    # Preprocess point clouds
    source_down = preprocess_point_cloud(source_pcd, voxel_size)
    target_down = preprocess_point_cloud(target_pcd, voxel_size)
    
    # Compute FPFH features
    source_fpfh = compute_fpfh_features(source_down, voxel_size)
    target_fpfh = compute_fpfh_features(target_down, voxel_size)
    
    # Global registration with RANSAC
    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    
    # Apply initial transformation
    source_temp = copy.deepcopy(source_pcd)
    source_temp.transform(result_ransac.transformation)
    
    # Refine with ICP
    result_icp = refine_registration(source_temp, target_pcd, result_ransac, voxel_size)
    
    # Create the final transformed source point cloud
    source_transformed = copy.deepcopy(source_pcd)
    final_transformation = result_icp.transformation
    source_transformed.transform(final_transformation)
    
    # Transfer material properties
    transformed_material_props = transfer_material_properties(
        source_pcd, source_transformed, source_material_props)
    
    # Report time
    elapsed_time = time.time() - start_time
    print(f":: Alignment completed in {elapsed_time:.2f} seconds")
    print(f":: Final transformation matrix:\n{final_transformation}")
    
    # Visualize result if requested
    if visualize:
        visualize_registration_result(source_pcd, target_pcd, source_transformed, 
                                     visualize_result_path)
    
    return source_transformed, final_transformation, transformed_material_props

def visualize_registration_result(source_original, target, source_transformed, 
                                save_path=None, show=True):
    """
    Visualize registration result
    
    Args:
        source_original: Original source point cloud
        target: Target point cloud
        source_transformed: Transformed source point cloud
        save_path: Path to save visualization
        show: Whether to show the visualization
    """
    # Create a new visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Get render option and set background color
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    render_option.point_size = 2.0
    
    # Create a copy of source_original with red color
    source_red = copy.deepcopy(source_original)
    source_red.paint_uniform_color([1, 0, 0])  # Red
    
    # Create a copy of target with green color
    target_green = copy.deepcopy(target)
    target_green.paint_uniform_color([0, 1, 0])  # Green
    
    # Create a copy of source_transformed with blue color
    source_transformed_blue = copy.deepcopy(source_transformed)
    source_transformed_blue.paint_uniform_color([0, 0, 1])  # Blue
    
    # Add all geometries
    vis.add_geometry(source_red)
    vis.add_geometry(target_green)
    vis.add_geometry(source_transformed_blue)
    
    # Update view
    vis.update_geometry(source_red)
    vis.update_geometry(target_green)
    vis.update_geometry(source_transformed_blue)
    
    # Reset view
    vis.reset_view_point(True)
    
    # Run visualization
    vis.poll_events()
    vis.update_renderer()
    
    # Save image if path provided
    if save_path:
        vis.capture_screen_image(save_path)
        print(f":: Saved alignment visualization to {save_path}")
    
    # Show if requested
    if show:
        vis.run()
    
    # Close the window
    vis.destroy_window()

def batch_align_point_clouds(reference_path, source_paths, output_dir, voxel_size=0.01):
    """
    Align multiple point clouds to a reference
    
    Args:
        reference_path: Path to reference PLY file
        source_paths: List of paths to source PLY files
        output_dir: Directory to save aligned point clouds
        voxel_size: Voxel size for downsampling
    
    Returns:
        list: Paths to aligned point clouds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference point cloud
    print(f":: Loading reference point cloud from {reference_path}")
    reference_pcd = o3d.io.read_point_cloud(reference_path)
    
    if len(reference_pcd.points) == 0:
        print(f"Error: Reference point cloud is empty")
        return []
    
    # Process each source point cloud
    aligned_paths = []
    for i, source_path in enumerate(source_paths):
        print(f"\n{'='*80}")
        print(f":: Processing source {i+1}/{len(source_paths)}: {source_path}")
        
        # Load source point cloud
        source_pcd = o3d.io.read_point_cloud(source_path)
        
        if len(source_pcd.points) == 0:
            print(f"Error: Source point cloud is empty, skipping")
            continue
        
        # Extract source filename and create output path
        source_filename = os.path.basename(source_path)
        source_name = os.path.splitext(source_filename)[0]
        aligned_filename = f"{source_name}_aligned.ply"
        aligned_path = os.path.join(output_dir, aligned_filename)
        vis_path = os.path.join(output_dir, f"{source_name}_alignment.png")
        
        # Read source material properties if they exist
        # Note: We need to extract these from the PLY file since Open3D doesn't load custom properties
        source_material_props = extract_material_properties_from_ply(source_path)
        
        # Align source to reference
        print(f":: Aligning {source_filename} to reference...")
        aligned_pcd, transformation, aligned_props = align_point_clouds_fpfh_ransac(
            source_pcd, reference_pcd, source_material_props, 
            voxel_size=voxel_size, visualize=True, visualize_result_path=vis_path)
        
        # Save aligned point cloud with material properties
        print(f":: Saving aligned point cloud to {aligned_path}")
        save_point_cloud_with_properties(aligned_pcd, aligned_path, aligned_props)
        
        aligned_paths.append(aligned_path)
        print(f":: Completed alignment for {source_filename}")
    
    return aligned_paths

def extract_material_properties_from_ply(ply_path):
    """
    Extract material properties from PLY file
    
    Args:
        ply_path: Path to PLY file
    
    Returns:
        dict: Material properties
    """
    try:
        from plyfile import PlyData
        
        # Read PLY file
        plydata = PlyData.read(ply_path)
        vertex_data = plydata['vertex']
        
        # Extract material properties
        material_props = {}
        
        # Check for roughness
        if 'roughness' in vertex_data.dtype.names:
            roughness = vertex_data['roughness']
            material_props['roughness'] = np.array(roughness)
            print(f":: Extracted roughness property with {len(roughness)} values")
        
        # Check for metallic
        if 'metallic' in vertex_data.dtype.names:
            metallic = vertex_data['metallic']
            material_props['metallic'] = np.array(metallic)
            print(f":: Extracted metallic property with {len(metallic)} values")
        
        return material_props
    
    except Exception as e:
        print(f":: Warning: Failed to extract material properties from {ply_path}: {e}")
        return {}

def save_point_cloud_with_properties(pcd, file_path, material_properties=None):
    """
    Save point cloud with material properties
    
    Args:
        pcd: open3d.geometry.PointCloud
        file_path: Output file path
        material_properties: Dictionary with material properties
    
    Returns:
        str: Path to saved file
    """
    try:
        # Check if we need to use extended PLY
        if material_properties and (
            'roughness' in material_properties or 'metallic' in material_properties):
            # Import save_extended_ply_with_plyfile from file_io module
            from utils.file_io import save_extended_ply_with_plyfile
            return save_extended_ply_with_plyfile(pcd, file_path, material_properties)
        else:
            # Use standard Open3D write function
            o3d.io.write_point_cloud(file_path, pcd, write_ascii=False)
            print(f":: Saved point cloud to {file_path} (without material properties)")
            return file_path
    
    except Exception as e:
        print(f":: Error saving point cloud: {e}")
        # Fallback to standard Open3D save
        o3d.io.write_point_cloud(file_path, pcd, write_ascii=False)
        print(f":: Saved point cloud to {file_path} (without material properties, fallback)")
        return file_path

if __name__ == "__main__":
    # Set base directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Handle command line arguments
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        print("\nPoint Cloud Alignment Tool")
        print("=" * 50)
        print("Usage:")
        print("  python alignment.py <reference_ply> <source_dir> <output_dir>")
        print("  python alignment.py --list")
        print("\nOptions:")
        print("  --list             List available point cloud files")
        print("  --help, -h         Show this help message")
        print("\nExample:")
        print("  python alignment.py data/point_cloud/reference.ply data/point_cloud data/aligned_output")
        sys.exit(0)
    
    # List available point cloud files
    if '--list' in sys.argv:
        point_cloud_dir = os.path.join(base_dir, 'data', 'point_cloud')
        print("\nAvailable point cloud files:")
        print("=" * 50)
        
        if not os.path.exists(point_cloud_dir):
            print(f"Point cloud directory not found: {point_cloud_dir}")
            sys.exit(1)
            
        files = sorted(glob.glob(os.path.join(point_cloud_dir, '*.ply')))
        
        if not files:
            print(f"No PLY files found in {point_cloud_dir}")
            sys.exit(1)
            
        for i, file_path in enumerate(files):
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"{i+1}. {file_name} ({file_size:.2f} MB)")
            
        print("\nUse one of these files as reference with:")
        print(f"python alignment.py data/point_cloud/FILE_NAME.ply data/point_cloud data/aligned_output")
        sys.exit(0)
    
    # Get parameters
    if len(sys.argv) < 4:
        print("Error: Missing required arguments")
        print("Run 'python alignment.py --help' for usage information")
        sys.exit(1)
    
    reference_path = sys.argv[1]
    source_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Fix paths if needed (make absolute if they're relative)
    if not os.path.isabs(reference_path):
        reference_path = os.path.join(base_dir, reference_path)
    
    if not os.path.isabs(source_dir):
        source_dir = os.path.join(base_dir, source_dir)
        
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, output_dir)
    
    # Verify that reference file exists
    if not os.path.exists(reference_path):
        print(f"Error: Reference file not found: {reference_path}")
        print("\nAvailable PLY files in data/point_cloud:")
        point_cloud_dir = os.path.join(base_dir, 'data', 'point_cloud')
        if os.path.exists(point_cloud_dir):
            files = sorted(glob.glob(os.path.join(point_cloud_dir, '*.ply')))
            for i, file_path in enumerate(files[:5]):  # Show first 5 files
                print(f"  - {os.path.basename(file_path)}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")
        print("\nRun 'python alignment.py --list' to see all available files")
        sys.exit(1)
    
    # Find all PLY files in source directory
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)
    
    source_paths = [os.path.join(source_dir, f) for f in os.listdir(source_dir) 
                   if f.lower().endswith('.ply') and os.path.join(source_dir, f) != reference_path]
    
    if not source_paths:
        print(f"No source PLY files found in {source_dir} (excluding reference file)")
        sys.exit(1)
    
    print(f"Reference file: {os.path.basename(reference_path)}")
    print(f"Found {len(source_paths)} source PLY files to align")
    
    # Align all point clouds to reference
    aligned_paths = batch_align_point_clouds(reference_path, source_paths, output_dir)
    
    print(f"\n{'='*80}")
    print(f"Alignment complete! Aligned {len(aligned_paths)} point clouds")
    print(f"Aligned point clouds saved to {output_dir}")
    print(f"{'='*80}")
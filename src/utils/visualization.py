import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import os
import cv2

def visualize_depth_map(depth_map, normal_map=None, save_path=None, show=True):
    """
    Visualize a depth map.
    
    Args:
        depth_map: 2D numpy array with depth values
        normal_map: Optional normal map for enhanced visualization
        save_path: Optional path to save the visualization
        show: Whether to display the visualization
    """
    plt.figure(figsize=(12, 10))
    
    # If we have a normal map, create an enhanced visualization
    if normal_map is not None:
        plt.subplot(2, 2, 1)
        plt.imshow(depth_map, cmap='viridis')
        plt.colorbar(label='Depth (m)')
        plt.title('Depth Map')
        
        # Visualize normals
        plt.subplot(2, 2, 2)
        # Convert to RGB by shifting from [-1,1] to [0,1] range
        rgb_normals = (normal_map + 1.0) / 2.0
        plt.imshow(rgb_normals)
        plt.title('Normal Map (RGB = XYZ)')
        
        # Visualize normal-enhanced depth
        plt.subplot(2, 2, 3)
        
        # Use normals to create shaded relief
        # Extract components from normal map
        nx = normal_map[:,:,0]
        ny = normal_map[:,:,1]
        nz = normal_map[:,:,2]
        
        # Create a simple shading based on dot product with a light direction
        light_dir = np.array([0.5, 0.5, 0.7])  # Light from top-right
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        shading = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
        
        # Clip to [0, 1] range
        shading = np.clip(shading, 0, 1)
        
        # Apply shading to depth map
        plt.imshow(depth_map * shading, cmap='gray')
        plt.colorbar(label='Shaded Depth')
        plt.title('Normal-Enhanced Depth')
        
        # Compute gradient of depth for comparison
        plt.subplot(2, 2, 4)
        dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        plt.imshow(gradient_magnitude, cmap='viridis')
        plt.colorbar(label='Gradient Magnitude')
        plt.title('Depth Gradient (Edges)')
    else:
        # Simple depth visualization
        plt.imshow(depth_map, cmap='viridis')
        plt.colorbar(label='Depth (m)')
        plt.title('Depth Map')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_point_cloud(point_cloud, title="Point Cloud", save_path=None, show=True):
    """
    Visualize a point cloud using Open3D.
    
    Args:
        point_cloud: open3d.geometry.PointCloud object
        title: Title for the visualization window
        save_path: Optional path to save the visualization
        show: Whether to display the visualization
    """
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise ValueError("Input must be an Open3D PointCloud")
        
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, width=1024, height=768)
    
    # Add point cloud to the visualizer
    vis.add_geometry(point_cloud)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    vis.add_geometry(coord_frame)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    # Update the view
    vis.poll_events()
    vis.update_renderer()
    
    # Save screenshot if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path)
    
    # Show if requested
    if show:
        vis.run()
    
    # Close the visualizer
    vis.destroy_window()

def visualize_mesh(mesh, title="Mesh Visualization", save_path=None, show=True):
    """
    Visualize a mesh using Open3D.
    
    Args:
        mesh: open3d.geometry.TriangleMesh object
        title: Title for the visualization window
        save_path: Optional path to save the visualization
        show: Whether to display the visualization
    """
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise ValueError("Input must be an Open3D TriangleMesh")
    
    # Ensure mesh has normals
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
        
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, width=1024, height=768)
    
    # Add mesh to the visualizer
    vis.add_geometry(mesh)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    vis.add_geometry(coord_frame)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.mesh_show_wireframe = True
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    
    # Update the view
    vis.poll_events()
    vis.update_renderer()
    
    # Save screenshot if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path)
    
    # Show if requested
    if show:
        vis.run()
    
    # Close the visualizer
    vis.destroy_window()

def visualize_volume(volume, save_dir=None, show=True):
    """
    Visualize a volumetric representation (mesh and point cloud).
    
    Args:
        volume: Dictionary with 'mesh' and 'point_cloud' keys
        save_dir: Optional directory to save visualizations
        show: Whether to display visualizations
    """
    mesh = volume["mesh"]
    point_cloud = volume["point_cloud"]
    
    # Create save paths if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        mesh_save_path = os.path.join(save_dir, "mesh_visualization.png")
        pcd_save_path = os.path.join(save_dir, "point_cloud_visualization.png")
    else:
        mesh_save_path = None
        pcd_save_path = None
    
    # Visualize mesh
    visualize_mesh(mesh, "Volumetric Mesh", mesh_save_path, show)
    
    # Visualize point cloud
    visualize_point_cloud(point_cloud, "Source Point Cloud", pcd_save_path, show)

def visualize_depth_cross_section(depth_map, save_path=None, show=True):
    """
    Visualize a cross-section of the depth map to check depth orientation.
    
    Args:
        depth_map: 2D numpy array with depth values
        save_path: Optional path to save the visualization
        show: Whether to display the visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Get the center row and column for cross-sections
    h, w = depth_map.shape
    center_row = depth_map[h//2, :]
    center_col = depth_map[:, w//2]
    
    # Plot horizontal cross-section
    plt.subplot(2, 1, 1)
    plt.plot(center_row)
    plt.title(f'Horizontal Cross-Section (Row {h//2})')
    plt.xlabel('Column')
    plt.ylabel('Depth Value')
    plt.grid(True)
    
    # Plot vertical cross-section
    plt.subplot(2, 1, 2)
    plt.plot(center_col)
    plt.title(f'Vertical Cross-Section (Column {w//2})')
    plt.xlabel('Row')
    plt.ylabel('Depth Value')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_camera_and_points(point_cloud, save_path=None, show=True):
    """
    Visualize the point cloud with camera coordinate system.
    
    Args:
        point_cloud: open3d.geometry.PointCloud object
        save_path: Optional path to save the visualization
        show: Whether to display the visualization
    """
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("Camera and Points", width=1024, height=768)
    
    # Add point cloud to the visualizer
    vis.add_geometry(point_cloud)
    
    # Add a coordinate frame representing the camera
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0]
    )
    vis.add_geometry(camera_frame)
    
    # Create a small sphere to represent camera position
    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.02, resolution=20
    )
    camera_sphere.paint_uniform_color([1, 0, 0])  # Red
    vis.add_geometry(camera_sphere)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 3.0
    
    # Set initial viewpoint - looking at the scene from behind the camera
    view_control = vis.get_view_control()
    
    # Get the bounding box of the point cloud to set a good viewpoint
    pcd_center = np.mean(np.asarray(point_cloud.points), axis=0) if len(point_cloud.points) > 0 else np.zeros(3)
    
    # Update the view
    vis.poll_events()
    vis.update_renderer()
    
    # Save screenshot if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path)
    
    # Show if requested
    if show:
        vis.run()
    
    # Close the visualizer
    vis.destroy_window()

def visualize_depth_comparison(original_depth, processed_depth, mask=None, save_path=None, show=True):
    """
    Visualize a comparison between original and processed depth maps to see outlier removal effects.
    
    Args:
        original_depth: Original depth map
        processed_depth: Processed depth map
        mask: Optional binary mask for valid regions
        save_path: Optional path to save the visualization
        show: Whether to display the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original depth map
    im1 = axes[0, 0].imshow(original_depth, cmap='viridis')
    axes[0, 0].set_title('Original Depth Map')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Processed depth map
    im2 = axes[0, 1].imshow(processed_depth, cmap='viridis')
    axes[0, 1].set_title('Processed Depth Map')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Create a difference map
    diff_map = np.zeros_like(original_depth)
    valid = (original_depth > 0) & (processed_depth > 0)
    if mask is not None:
        valid = valid & (mask > 0)
    
    if np.any(valid):
        diff_map[valid] = original_depth[valid] - processed_depth[valid]
    
    # Difference map
    im3 = axes[1, 0].imshow(diff_map, cmap='RdBu')
    axes[1, 0].set_title('Difference (Blue = Removed Outliers)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 3D visualization of a cross-section
    h, w = original_depth.shape
    mid_row = h // 2
    
    axes[1, 1].plot(original_depth[mid_row, :], 'r-', label='Original')
    axes[1, 1].plot(processed_depth[mid_row, :], 'g-', label='Processed')
    axes[1, 1].set_title(f'Cross-section at Row {mid_row}')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_normal_map(normal_map, save_path=None, show=True):
    """
    Visualize a normal map as an RGB image.
    
    Args:
        normal_map: 3D numpy array with shape [H, W, 3] containing XYZ normal vectors
        save_path: Optional path to save the visualization
        show: Whether to display the visualization
    """
    plt.figure(figsize=(10, 8))
    
    # Convert normals from [-1,1] range to [0,1] for visualization
    vis_normals = (normal_map + 1.0) / 2.0
    
    # Plot normal map
    plt.imshow(vis_normals)
    plt.title('Normal Map (RGB = XYZ)')
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def color_point_cloud_by_normals(point_cloud):
    """
    Color a point cloud based on its normal directions.
    Uses normal XYZ components mapped to RGB colors.
    
    Args:
        point_cloud: open3d.geometry.PointCloud object with normals
        
    Returns:
        open3d.geometry.PointCloud: Point cloud with colors
    """
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise ValueError("Input must be an Open3D PointCloud")
        
    if not point_cloud.has_normals():
        point_cloud.estimate_normals()
        point_cloud.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        print("Estimated normals for the point cloud")
    
    # Get the normal vectors
    normals = np.asarray(point_cloud.normals)
    
    # Map normal directions to colors
    # Convert from [-1, 1] range to [0, 1] range for RGB colors
    colors = (normals + 1.0) / 2.0
    
    # Assign colors to the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return point_cloud

def visualize_point_cloud_with_normals(point_cloud, scale=0.05, sample_ratio=0.05, title="Point Cloud with Normals", save_path=None, show=True):
    """
    Visualize a point cloud with normal vectors displayed as arrows.
    
    Args:
        point_cloud: open3d.geometry.PointCloud object with normals
        scale: Length of the normal arrows relative to the scene size
        sample_ratio: Ratio of points for which to display normals (to avoid cluttering)
        title: Title for the visualization window
        save_path: Optional path to save the visualization
        show: Whether to display the visualization
    """
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise ValueError("Input must be an Open3D PointCloud")
        
    if not point_cloud.has_normals():
        print("Point cloud has no normals. Estimating normals...")
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        point_cloud.orient_normals_towards_camera_location(np.array([0, 0, 0]))
    
    # Sample points to avoid cluttering the visualization
    total_points = len(point_cloud.points)
    num_samples = max(1, int(total_points * sample_ratio))
    indices = np.random.choice(np.arange(total_points), num_samples, replace=False)
    
    # Create a sampled point cloud for visualization
    sampled_points = np.asarray(point_cloud.points)[indices]
    sampled_normals = np.asarray(point_cloud.normals)[indices]
    
    # Create a LineSet to represent normal vectors as arrows
    line_points = []
    line_indices = []
    
    # Calculate bounding box diagonal for appropriate normal vector scale
    bbox = point_cloud.get_axis_aligned_bounding_box()
    bbox_size = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    arrow_length = bbox_size * scale
    
    # Create line segments for the normals
    for i, (point, normal) in enumerate(zip(sampled_points, sampled_normals)):
        line_points.append(point)
        line_points.append(point + normal * arrow_length)
        line_indices.append([i*2, i*2+1])
    
    # Create LineSet and set colors to represent normal directions
    normal_lines = o3d.geometry.LineSet()
    normal_lines.points = o3d.utility.Vector3dVector(line_points)
    normal_lines.lines = o3d.utility.Vector2iVector(line_indices)
    
    # Use bright red color for normal vectors
    normal_colors = [[1, 0, 0] for _ in range(len(line_indices))]
    normal_lines.colors = o3d.utility.Vector3dVector(normal_colors)
    
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, width=1024, height=768)
    
    # Add the point cloud
    vis.add_geometry(point_cloud)
    
    # Add normal lines
    vis.add_geometry(normal_lines)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=arrow_length * 3, origin=[0, 0, 0]
    )
    vis.add_geometry(coord_frame)
    
    # Configure the view
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])  # Face the camera along -Z direction
    view_control.set_up([0, 1, 0])      # Y axis is up
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    # Update the view
    vis.poll_events()
    vis.update_renderer()
    
    # Save screenshot if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path)
    
    # Show if requested
    if show:
        vis.run()
    
    # Close the visualizer
    vis.destroy_window()
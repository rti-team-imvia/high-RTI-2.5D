import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import os

def visualize_depth_map(depth_map, save_path=None, show=True, colormap='viridis'):
    """
    Visualize a depth map as a heatmap.
    
    Args:
        depth_map: 2D numpy array with depth values
        save_path: Optional path to save the visualization
        show: Whether to display the visualization
        colormap: Matplotlib colormap to use
    """
    plt.figure(figsize=(10, 8))
    
    # Plot depth map
    plt.imshow(depth_map, cmap=colormap)
    plt.colorbar(label='Depth')
    plt.title('Depth Map')
    
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
import numpy as np
import open3d as o3d
from camera.intrinsics import CameraIntrinsics

class VolumeGenerator:
    """
    Class for generating volumetric representations from depth maps.
    """
    def __init__(self, intrinsics=None):
        """
        Initialize the VolumeGenerator with camera intrinsics.
        
        Args:
            intrinsics: Camera intrinsics object or None to use default values
        """
        self.intrinsics = intrinsics or CameraIntrinsics()
        
    def depth_to_point_cloud(self, depth_map, mask=None, camera_offset=0.5):
        """
        Convert a depth map to a point cloud using camera intrinsics.
        The camera is positioned at (0,0,0) looking down the positive Z axis.
        
        Args:
            depth_map: 2D numpy array with depth values
            mask: Optional binary mask to filter valid regions
            camera_offset: Distance to offset the points from camera origin
            
        Returns:
            open3d.geometry.PointCloud: 3D point cloud
        """
        height, width = depth_map.shape
        
        # Create a grid of pixel coordinates
        v, u = np.indices((height, width)).astype(np.float32)
        
        # Get intrinsics parameters
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy
        
        # Apply mask if provided
        if mask is not None:
            valid_pixels = mask > 0
        else:
            # Filter by valid depth
            valid_pixels = depth_map > 0
            
        # Apply both mask and depth validity check
        valid_pixels = valid_pixels & (depth_map > 0)
        
        # Skip if no valid pixels
        if not np.any(valid_pixels):
            print("WARNING: No valid depth values found!")
            return o3d.geometry.PointCloud()
            
        u_valid = u[valid_pixels]
        v_valid = v[valid_pixels]
        z = depth_map[valid_pixels]
        
        # Scale the depth values to appropriate range if needed
        # Add an offset to ensure all points are in front of the camera
        z = z + camera_offset
        
        print(f"Depth range after offset: {np.min(z)} to {np.max(z)}")
            
        # Calculate 3D coordinates using pinhole camera model
        x = (u_valid - cx) * z / fx
        y = (v_valid - cy) * z / fy
        
        # Create point cloud
        points = np.stack((x, y, z), axis=1)
        
        # Debug information
        print(f"Generated point cloud with {len(points)} points")
        print(f"X range: [{np.min(x)}, {np.max(x)}]")
        print(f"Y range: [{np.min(y)}, {np.max(y)}]")
        print(f"Z range: [{np.min(z)}, {np.max(z)}]")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Apply additional statistical outlier removal to clean up the point cloud
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=30,
            std_ratio=2.0  # More aggressive outlier removal
        )
        
        # Estimate normals with more neighbors for robustness
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50)
        )
        pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        
        return pcd

    def create_volume(self, depth_map, mask=None, voxel_size=0.01, camera_offset=0.5):
        """
        Create a volumetric representation from a depth map.
        
        Args:
            depth_map: 2D numpy array with depth values
            mask: Optional binary mask to filter valid regions
            voxel_size: Size of voxels in the volumetric representation
            camera_offset: Distance to offset the points from camera origin
            
        Returns:
            dict: Dictionary containing the volumetric mesh and point cloud
        """
        # Convert depth map to point cloud
        point_cloud = self.depth_to_point_cloud(depth_map, mask, camera_offset)
        
        # Check if point cloud has enough points
        if len(point_cloud.points) < 10:
            print("ERROR: Not enough points to create a volumetric representation")
            # Return empty volume
            return {
                "mesh": o3d.geometry.TriangleMesh(),
                "point_cloud": point_cloud,
                "voxel_size": voxel_size
            }
        
        # Remove outliers again (double filtering for better results)
        cl, ind = point_cloud.remove_statistical_outlier(
            nb_neighbors=30,
            std_ratio=1.5  # Even more aggressive
        )
        point_cloud = point_cloud.select_by_index(ind)
        
        # Optionally downsample for faster and more stable reconstruction
        point_cloud = point_cloud.voxel_down_sample(voxel_size)
        
        # Create mesh using Poisson surface reconstruction
        print("Generating mesh with Poisson reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, 
            depth=9,
            scale=1.1,
            linear_fit=True
        )
        
        # Filter out low-density vertices more aggressively
        if len(densities) > 0:
            # Use a higher threshold to remove more low-density regions (typically outliers)
            vertices_to_remove = densities < np.quantile(densities, 0.1)  # Remove bottom 10%
            mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Clean up mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        
        # Compute vertex normals for rendering
        mesh.compute_vertex_normals()
        
        return {
            "mesh": mesh,
            "point_cloud": point_cloud,
            "voxel_size": voxel_size
        }
    
    def extract_mesh(self, volume):
        """
        Extract the mesh from the volumetric representation.
        
        Args:
            volume: Volumetric representation dictionary
            
        Returns:
            open3d.geometry.TriangleMesh: 3D mesh
        """
        return volume["mesh"]
    
    def create_volume_from_metric_depth(self, depth_map, mask=None, voxel_size=0.01):
        """
        Create a volumetric representation from a metric depth map.
        For metric depth maps, we use the actual values without offset or scaling.
        
        Args:
            depth_map: 2D numpy array with metric depth values
            mask: Optional binary mask to filter valid regions
            voxel_size: Size of voxels in the volumetric representation (in meters)
            
        Returns:
            dict: Dictionary containing the volumetric mesh and point cloud
        """
        height, width = depth_map.shape
        
        # Create a grid of pixel coordinates
        v, u = np.indices((height, width)).astype(np.float32)
        
        # Get intrinsics parameters
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy
        
        # Apply mask if provided
        if mask is not None:
            valid_pixels = mask > 0
        else:
            valid_pixels = np.ones_like(depth_map, dtype=bool)
            
        # Further filter by valid depth values - be more tolerant with 32-bit floats
        # which might have very small but valid values
        min_depth_threshold = 0.001  # 1mm minimum depth
        valid_pixels = valid_pixels & (depth_map > min_depth_threshold)
        
        # Skip if no valid pixels
        if not np.any(valid_pixels):
            print("WARNING: No valid depth values found!")
            return {
                "mesh": o3d.geometry.TriangleMesh(),
                "point_cloud": o3d.geometry.PointCloud(),
                "voxel_size": voxel_size
            }
            
        u_valid = u[valid_pixels]
        v_valid = v[valid_pixels]
        z = depth_map[valid_pixels]
        
        # Use the metric depth values directly - no offset needed
        print(f"Using metric depth range: {np.min(z):.6f}m to {np.max(z):.6f}m")
            
        # Calculate 3D coordinates using pinhole camera model
        x = (u_valid - cx) * z / fx
        y = (v_valid - cy) * z / fy
        
        # Create point cloud
        points = np.stack((x, y, z), axis=1)
        
        # Debug information
        print(f"Generated point cloud with {len(points)} points")
        print(f"X range: [{np.min(x):.6f}m, {np.max(x):.6f}m]")
        print(f"Y range: [{np.min(y):.6f}m, {np.max(y):.6f}m]")
        print(f"Z range: [{np.min(z):.6f}m, {np.max(z):.6f}m]")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Apply statistical outlier removal
        if len(points) > 100:  # Only if we have enough points
            print("Removing statistical outliers...")
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0
            )
            print(f"After outlier removal: {len(pcd.points)} points")
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=30)
        )
        pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        
        # Create mesh using Poisson surface reconstruction
        print("Generating mesh with Poisson reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=9,
            scale=1.1,
            linear_fit=True
        )
        
        # Filter out low-density vertices if needed
        if len(densities) > 0:
            vertices_to_remove = densities < np.quantile(densities, 0.05)
            mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Clean up mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        
        # Compute vertex normals for rendering
        mesh.compute_vertex_normals()
        
        return {
            "mesh": mesh,
            "point_cloud": pcd,
            "voxel_size": voxel_size
        }
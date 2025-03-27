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

    def create_volume_from_metric_depth(self, depth_map, mask=None, normal_map=None, voxel_size=0.01):
        """
        Create a volumetric representation from a metric depth map.
        Optionally uses normal map for improved reconstruction.
        
        Args:
            depth_map: 2D numpy array with metric depth values
            mask: Optional binary mask to filter valid regions
            normal_map: Optional 3D numpy array with normal vectors (shape [H, W, 3])
            voxel_size: Size of voxels in the volumetric representation
            
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
            
        # Further filter by valid depth values
        valid_pixels = valid_pixels & (depth_map > 0)
        
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
        
        # Use the metric depth values directly
        print(f"Using metric depth range: {np.min(z):.3f}m to {np.max(z):.3f}m")
            
        # Calculate 3D coordinates using pinhole camera model
        x = (u_valid - cx) * z / fx
        y = (v_valid - cy) * z / fy
        
        # Create point cloud
        points = np.stack((x, y, z), axis=1)
        
        # Debug information
        print(f"Generated point cloud with {len(points)} points")
        print(f"X range: [{np.min(x):.3f}m, {np.max(x):.3f}m]")
        print(f"Y range: [{np.min(y):.3f}m, {np.max(y):.3f}m]")
        print(f"Z range: [{np.min(z):.3f}m, {np.max(z):.3f}m]")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # If we have a normal map, use it instead of estimating normals
        if normal_map is not None:
            print("Using provided normal map for improved reconstruction...")
            
            # Extract normals for valid pixels
            normals = normal_map[valid_pixels]
            
            # Convert camera-space normals to world-space
            # Note: This transformation depends on your camera coordinate system
            # This assumes depth increases in +Z, right is +X, down is +Y
            # Adjust if your coordinate system is different
            world_normals = np.copy(normals)
            
            # Set the normals in the point cloud
            pcd.normals = o3d.utility.Vector3dVector(world_normals)
            print(f"Applied {len(world_normals)} normals from normal map")
        else:
            # Estimate normals if no normal map provided
            print("Estimating normals from point positions...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=30)
            )
            pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        
        # Apply statistical outlier removal
        if len(points) > 100:  # Only if we have enough points
            print("Removing statistical outliers...")
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0
            )
            print(f"After outlier removal: {len(pcd.points)} points")
        
        # Apply normal-guided smoothing to flat areas
        if normal_map is not None:
            print("Applying normal-guided smoothing to improve flat regions...")
            pcd = self._smooth_flat_regions(pcd, voxel_size)
            
        # Create mesh using Poisson surface reconstruction with adjusted parameters
        # Normal maps allow for higher quality reconstruction
        print("Generating mesh with Poisson reconstruction...")
        poisson_depth = 9  # Default depth
        if normal_map is not None:
            poisson_depth = 10  # Increase depth for more detail when normals are available
            
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=poisson_depth,
            scale=1.1,
            linear_fit=True
        )
        
        # Filter out low-density vertices with more aggressive filtering when using normals
        density_threshold = 0.05  # Default threshold
        if normal_map is not None:
            density_threshold = 0.03  # More aggressive filtering with normal information
            
        if len(densities) > 0:
            vertices_to_remove = densities < np.quantile(densities, density_threshold)
            mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Clean up mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        
        # Compute vertex normals for rendering with better normal consistency
        if normal_map is not None:
            # Use larger cone angle for smoother normal transitions
            mesh.compute_vertex_normals(normalized=True)
        else:
            mesh.compute_vertex_normals()
        
        return {
            "mesh": mesh,
            "point_cloud": pcd,
            "voxel_size": voxel_size
        }

    def _smooth_flat_regions(self, point_cloud, voxel_size):
        """
        Apply smoothing to flat regions identified by normal consistency.
        This helps improve reconstruction in areas with low depth variation.
        
        Args:
            point_cloud: Open3D point cloud with normals
            voxel_size: Size of voxels for neighborhood search
            
        Returns:
            Open3D point cloud with adjusted points in flat regions
        """
        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)
        
        if len(points) < 10:
            return point_cloud  # Not enough points for meaningful smoothing
            
        # Create a KD-tree for neighborhood searches
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
        
        # Parameters for flat region detection
        normal_threshold = 0.98  # Cosine threshold for parallel normals (0.98 ≈ 11° difference)
        smoothing_factor = 0.2   # How much to adjust points (0.2 = 20% towards local plane)
        
        # Process each point
        adjusted_points = np.copy(points)
        
        print(f"Smoothing flat regions using normal consistency...")
        
        # To avoid processing every point (which could be slow for large point clouds)
        # we'll use a sampling approach or process in batches
        process_count = min(len(points), 5000)  # Limit to 5000 points
        indices = np.random.choice(len(points), process_count, replace=False)
        
        improved_count = 0
        for idx in indices:
            # Find neighboring points
            _, neighbor_indices, _ = pcd_tree.search_radius_vector_3d(points[idx], voxel_size * 3)
            
            if len(neighbor_indices) < 5:
                continue  # Not enough neighbors
                
            # Get neighboring normals and check if this is a flat region
            neighbor_normals = normals[neighbor_indices]
            reference_normal = normals[idx]
            
            # Calculate dot products to measure normal alignment
            dot_products = np.abs(np.sum(neighbor_normals * reference_normal, axis=1))
            
            # If most normals are aligned, this is likely a flat region
            if np.mean(dot_products) > normal_threshold:
                # This is a flat region - apply smoothing
                
                # Compute local plane using PCA
                neighbor_points = points[neighbor_indices]
                centroid = np.mean(neighbor_points, axis=0)
                
                # Adjust the point towards the local plane
                plane_normal = reference_normal
                point_to_plane = np.dot(points[idx] - centroid, plane_normal) * plane_normal
                
                # Smoothly move the point towards the plane
                adjusted_points[idx] = points[idx] - (smoothing_factor * point_to_plane)
                improved_count += 1
        
        # Create a new point cloud with adjusted points
        smoothed_pcd = o3d.geometry.PointCloud()
        smoothed_pcd.points = o3d.utility.Vector3dVector(adjusted_points)
        smoothed_pcd.normals = o3d.utility.Vector3dVector(normals)
        
        print(f"Improved {improved_count} points in flat regions")
        
        return smoothed_pcd
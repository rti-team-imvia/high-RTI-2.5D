import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from camera.intrinsics import CameraIntrinsics
from utils.gpu_utils import check_gpu_availability

class VolumeGenerator:
    """
    Class for generating volumetric representations from depth maps.
    """
    def __init__(self, intrinsics=None, selective_gpu=False):
        """
        Initialize the VolumeGenerator with camera intrinsics.
        
        Args:
            intrinsics: Camera intrinsics object or None to use default values
            selective_gpu: Whether to use selective GPU acceleration
        """
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.selective_gpu, _, _, self.device = check_gpu_availability()
        
        # Only use GPU if explicitly requested and available
        self.selective_gpu = self.selective_gpu and selective_gpu
        if self.selective_gpu:
            print(f"Using selective GPU acceleration for point cloud generation")
    
    def depth_to_point_cloud(self, depth_map, mask=None, normal_map=None, color_map=None, camera_offset=0.5):
        """
        Convert a depth map to a point cloud using camera intrinsics.
        The camera is positioned at (0,0,0) looking down the positive Z axis.
        
        Args:
            depth_map: 2D numpy array with depth values
            mask: Optional binary mask to filter valid regions
            normal_map: Optional 3D numpy array with normal vectors (shape [H, W, 3])
            color_map: Optional 3D numpy array with color values (shape [H, W, 3])
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
        
        # Add normals if provided
        if normal_map is not None:
            normals = normal_map[valid_pixels]
            pcd.normals = o3d.utility.Vector3dVector(normals)
            print(f"Added {len(normals)} normals from normal map")
        else:
            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        
        # Add colors if available
        if color_map is not None and color_map.shape[:2] == depth_map.shape:
            # Get colors for valid pixels
            colors = color_map[valid_pixels]
            # Normalize colors to [0, 1] range if needed
            if colors.dtype == np.uint8:
                colors = colors.astype(np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
            print(f"Applied {len(colors)} colors from color map")
        
        # Apply additional statistical outlier removal to clean up the point cloud
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=30,
            std_ratio=2.0  # More aggressive outlier removal
        )
        
        return pcd

    def depth_to_point_cloud_gpu(self, depth_map, mask=None, normal_map=None, color_map=None):
        """
        GPU-accelerated conversion of depth map to point cloud.
        Only uses GPU for the coordinate calculation which is highly parallel.
        
        Args:
            depth_map: 2D numpy array with depth values
            mask: Optional binary mask to filter valid regions
            normal_map: Optional normal map for enhanced point cloud
            color_map: Optional color map for point cloud coloring
            
        Returns:
            open3d.geometry.PointCloud: 3D point cloud
        """
        import torch
        
        height, width = depth_map.shape
        
        # Move data to GPU for the coordinate calculation
        depth_tensor = torch.tensor(depth_map, dtype=torch.float32, device=self.device)
        
        if mask is not None:
            mask_tensor = torch.tensor(mask > 0, dtype=torch.bool, device=self.device)
        else:
            mask_tensor = torch.ones((height, width), dtype=torch.bool, device=self.device)
        
        # Apply valid depth mask
        mask_tensor = mask_tensor & (depth_tensor > 0)
        
        # Skip if no valid pixels
        if not torch.any(mask_tensor):
            print("WARNING: No valid depth values found!")
            return o3d.geometry.PointCloud()
        
        # Create coordinate grids on GPU
        v, u = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32)
        )
        
        # Extract valid coordinates and depths
        valid_indices = torch.nonzero(mask_tensor, as_tuple=True)
        z = depth_tensor[valid_indices]
        u_valid = u[valid_indices]
        v_valid = v[valid_indices]
        
        # Get camera intrinsics
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy
        
        # Calculate 3D coordinates
        x = (u_valid - cx) * z / fx
        y = (v_valid - cy) * z / fy
        
        # Create point cloud (move back to CPU for Open3D)
        points = torch.stack((x, y, z), dim=1).cpu().numpy()
        
        # Debug information
        print(f"Generated point cloud with {len(points)} points")
        print(f"X range: [{points[:,0].min():.6f}m, {points[:,0].max():.6f}m]")
        print(f"Y range: [{points[:,1].min():.6f}m, {points[:,1].max():.6f}m]")
        print(f"Z range: [{points[:,2].min():.6f}m, {points[:,2].max():.6f}m]")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add normals if provided
        if normal_map is not None:
            # Extract normals for valid pixels (use CPU indices)
            valid_y = valid_indices[0].cpu().numpy()
            valid_x = valid_indices[1].cpu().numpy()
            normals = normal_map[valid_y, valid_x]
            pcd.normals = o3d.utility.Vector3dVector(normals)
            print(f"Applied {len(normals)} normals from normal map")
        else:
            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        
        # Add colors if provided
        if color_map is not None and color_map.shape[:2] == depth_map.shape:
            # Extract colors for valid pixels
            valid_y = valid_indices[0].cpu().numpy()
            valid_x = valid_indices[1].cpu().numpy()
            colors = color_map[valid_y, valid_x]
            # Normalize colors to [0, 1] range if needed
            if colors.dtype == np.uint8:
                colors = colors.astype(np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
            print(f"Applied {len(colors)} colors from color map")
        
        return pcd
    
    def create_volume_from_metric_depth(self, depth_map, mask=None, normal_map=None, color_map=None, voxel_size=0.01):
        """
        Create a volumetric representation from a metric depth map.
        Optionally uses normal map for improved reconstruction.
        
        Args:
            depth_map: 2D numpy array with metric depth values
            mask: Optional binary mask to filter valid regions
            normal_map: Optional 3D numpy array with normal vectors (shape [H, W, 3])
            color_map: Optional color map for point cloud coloring
            voxel_size: Size of voxels in the volumetric representation
            
        Returns:
            dict: Dictionary containing the volumetric mesh and point cloud
        """
        # Generate point cloud
        if self.selective_gpu:
            # Use GPU for the parallelizable coordinates calculation
            print("Using GPU-accelerated point cloud generation...")
            pcd = self.depth_to_point_cloud_gpu(depth_map, mask, normal_map, color_map)
        else:
            # Use CPU implementation
            print("Using CPU point cloud generation...")
            pcd = self.depth_to_point_cloud(depth_map, mask, normal_map, color_map)
        
        # Check if point cloud has enough points
        if len(pcd.points) < 10:
            print("ERROR: Not enough valid points to create a volumetric representation")
            # Return empty volume
            return {
                "mesh": o3d.geometry.TriangleMesh(),
                "point_cloud": pcd,
                "voxel_size": voxel_size
            }
        
        # Apply statistical outlier removal
        print("Removing outliers from point cloud...")
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )
        
        # Report on the number of points being used
        print(f"Using full resolution point cloud with {len(pcd.points)} points")
        
        # Create mesh using Poisson surface reconstruction
        print("Generating mesh with Poisson reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=9,
            scale=1.1,
            linear_fit=True
        )
        
        # Filter out low-density vertices
        if len(densities) > 0:
            # Convert densities to numpy array before division
            densities_np = np.asarray(densities)
            
            # Now we can safely perform operations on the numpy array
            if np.max(densities_np) > 0:  # Prevent division by zero
                density_colors = plt.cm.plasma(densities_np / np.max(densities_np))[:, :3]
                
                density_mesh = o3d.geometry.TriangleMesh()
                density_mesh.vertices = mesh.vertices
                density_mesh.triangles = mesh.triangles
                density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
                
                # Remove vertices with low density
                threshold = np.quantile(densities_np, 0.1)
                print(f"Removing vertices with density < {threshold:.6f}")
                vertices_to_remove = densities_np < threshold
                mesh.remove_vertices_by_mask(vertices_to_remove)
            else:
                print("Warning: All density values are zero, skipping density filtering")
        
        # Clean up mesh
        print("Cleaning up mesh...")
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
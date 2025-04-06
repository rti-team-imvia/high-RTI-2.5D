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
    
    def depth_to_point_cloud(self, depth_map, mask=None, normal_map=None, color_map=None, 
                             roughness_map=None, metallic_map=None):
        """
        Convert a depth map to a point cloud using camera intrinsics.
        Uses Open3D's built-in functionality with additional property handling.
        
        Args:
            depth_map: 2D numpy array with depth values (meters)
            mask: Optional binary mask to filter valid regions
            normal_map: Optional 3D numpy array with normal vectors (shape [H, W, 3])
            color_map: Optional 3D numpy array with color values (shape [H, W, 3])
            roughness_map: Optional roughness map (grayscale) for PBR materials
            metallic_map: Optional metallic map (grayscale) for PBR materials
            
        Returns:
            tuple: (open3d.geometry.PointCloud, dict) containing point cloud and material properties
        """
        height, width = depth_map.shape
        
        # Apply mask if provided
        masked_depth = depth_map.copy()
        if mask is not None:
            masked_depth = masked_depth * (mask > 0)
        
        # Create Open3D camera intrinsics object
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=self.intrinsics.fx,
            fy=self.intrinsics.fy,
            cx=self.intrinsics.cx,
            cy=self.intrinsics.cy
        )
        
        # Convert depth to Open3D image (must be float32)
        o3d_depth = o3d.geometry.Image(masked_depth.astype(np.float32))
        
        # Create point cloud directly from depth image
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth,
            o3d_intrinsics,
            depth_scale=1.0,  # Assuming depth is already in meters
            depth_trunc=float('inf')  # Don't truncate any depth values
        )
        
        # Report point count
        print(f"Generated point cloud with {len(pcd.points)} points")
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points)
            print(f"X range: [{points[:,0].min():.6f}m, {points[:,0].max():.6f}m]")
            print(f"Y range: [{points[:,1].min():.6f}m, {points[:,1].max():.6f}m]")
            print(f"Z range: [{points[:,2].min():.6f}m, {points[:,2].max():.6f}m]")
        
        # Add normals if provided
        if normal_map is not None:
            # We need to reconstruct the mapping from 3D points back to 2D image
            # This is complex with the built-in method, so we'll use a workaround
            # by creating a new point cloud with the same points but custom normals
            if len(pcd.points) > 0:
                # Create valid pixels mask
                valid_mask = np.logical_and(masked_depth > 0, 
                                            ~np.isnan(masked_depth) & ~np.isinf(masked_depth))
                valid_y, valid_x = np.where(valid_mask)
                
                # Only use points that have valid indices in our arrays
                if len(valid_y) == len(pcd.points):
                    normals = normal_map[valid_y, valid_x]
                    pcd.normals = o3d.utility.Vector3dVector(normals)
                    print(f"Applied {len(normals)} normals from normal map")
                else:
                    # Fallback to estimation
                    print("Warning: Unable to map normals directly, estimating instead")
                    pcd.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                    )
            
            # Orient normals consistently
            pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        else:
            # Estimate normals if not provided
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        
        # Material properties dictionary
        material_info = {}
        
        # Similar approach for colors, roughness and metallic
        if color_map is not None and len(pcd.points) > 0:
            valid_mask = masked_depth > 0
            valid_y, valid_x = np.where(valid_mask)
            
            if len(valid_y) == len(pcd.points):
                colors = color_map[valid_y, valid_x]
                if colors.dtype == np.uint8:
                    colors = colors.astype(np.float32) / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
                print(f"Applied {len(colors)} colors from color map")
        
        # Add roughness values
        if roughness_map is not None and len(pcd.points) > 0:
            valid_mask = masked_depth > 0
            valid_y, valid_x = np.where(valid_mask)
            
            if len(valid_y) == len(pcd.points):
                roughness_values = roughness_map[valid_y, valid_x]
                material_info["roughness"] = roughness_values
                print(f"Added {len(roughness_values)} roughness values")
        
        # Add metallic values
        if metallic_map is not None and len(pcd.points) > 0:
            valid_mask = masked_depth > 0
            valid_y, valid_x = np.where(valid_mask)
            
            if len(valid_y) == len(pcd.points):
                metallic_values = metallic_map[valid_y, valid_x]
                material_info["metallic"] = metallic_values
                print(f"Added {len(metallic_values)} metallic values")
        
        return pcd, material_info
    
    def create_volume_from_metric_depth(self, depth_map, mask=None, normal_map=None, color_map=None, 
                                        roughness_map=None, metallic_map=None, voxel_size=0.01):
        """
        Create a point cloud from a metric depth map with material properties.
        Uses optimized point cloud generation and cleaning.
        
        Args:
            depth_map: 2D numpy array with metric depth values
            mask: Optional binary mask to filter valid regions
            normal_map: Optional 3D numpy array with normal vectors (shape [H, W, 3])
            color_map: Optional color map for point cloud coloring
            roughness_map: Optional roughness map (grayscale) for PBR materials
            metallic_map: Optional metallic map (grayscale) for PBR materials
            voxel_size: Size of voxels for potential downsampling
            
        Returns:
            dict: Dictionary containing the point cloud with materials
        """
        # Generate point cloud (only one implementation now)
        pcd, material_info = self.depth_to_point_cloud(
            depth_map, mask, normal_map, color_map, roughness_map, metallic_map
        )
        
        # Check if point cloud has enough points
        if len(pcd.points) < 10:
            print("ERROR: Not enough valid points to create a point cloud")
            return {
                "mesh": o3d.geometry.TriangleMesh(),  # Empty mesh
                "point_cloud": pcd,
                "material_properties": material_info,
                "voxel_size": voxel_size
            }
        
        # Clean the point cloud - chained operations for cleaner code
        print("Cleaning point cloud...")
        pcd = pcd.remove_non_finite_points()
        pcd = pcd.remove_duplicated_points()
        
        # Statistical outlier removal (keep this separate for clarity)
        print("Removing outliers from point cloud...")
        pcd, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=4.0
        )

        # Update material properties to match the filtered point cloud
        if material_info:
            print("Updating material properties after outlier removal...")
            for prop_name in list(material_info.keys()):
                if len(material_info[prop_name]) > len(pcd.points):
                    # Use inlier indices to keep the matching properties
                    try:
                        material_info[prop_name] = material_info[prop_name][inlier_indices]
                        print(f"Updated {prop_name} property: {len(material_info[prop_name])} values remain")
                    except Exception as e:
                        print(f"Failed to update {prop_name} property: {str(e)}")
                        # If we can't update, we'll remove it
                        material_info.pop(prop_name)
        
        # Ensure normals are properly oriented
        if pcd.has_normals():
            print("Ensuring normals are properly oriented...")
            pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        
        # Report on the final point cloud
        print(f"Final point cloud has {len(pcd.points)} points with:")
        if pcd.has_normals():
            print(f"- Normal vectors")
        if pcd.has_colors():
            print(f"- Color information")
        if "roughness" in material_info:
            print(f"- Roughness values")
        if "metallic" in material_info:
            print(f"- Metallic values")
        
        # Create empty mesh - we're not generating a mesh but keeping the return structure
        # for compatibility with existing code
        empty_mesh = o3d.geometry.TriangleMesh()
        
        return {
            "mesh": empty_mesh,  # Empty mesh
            "point_cloud": pcd,
            "material_properties": material_info,
            "voxel_size": voxel_size
        }
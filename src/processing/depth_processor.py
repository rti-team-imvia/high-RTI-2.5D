import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters, morphology
from utils.gpu_utils import check_gpu_availability, gpu_enabled_bilateral_filter

class DepthProcessor:
    """
    Class for processing depth maps before volumetric reconstruction.
    """
    def __init__(self):
        self.depth_map = None
        self.selective_gpu, _, _, _ = check_gpu_availability()
        if self.selective_gpu:
            print("Will use GPU only for parallel operations (convolutions, large matrix ops)")
        
    def _clean_mask(self, mask):
        """
        Clean up the mask using morphological operations.
        
        Args:
            mask: Binary mask
            
        Returns:
            numpy.ndarray: Cleaned mask
        """
        # Ensure the mask is binary
        binary_mask = mask > 0
        
        # Remove small isolated regions
        # First label all connected components
        num_labels, labels = cv2.connectedComponents(binary_mask.astype(np.uint8))
        
        # Count pixels in each component
        for label in range(1, num_labels):
            component_size = np.sum(labels == label)
            
            # Remove small components (adjust threshold as needed)
            if component_size < 100:
                binary_mask[labels == label] = 0
                
        # Close small holes in the mask
        kernel = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Optional: smooth the mask boundaries
        kernel = np.ones((3, 3), np.uint8)
        smoothed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
        
        return smoothed_mask
    
    def _remove_statistical_outliers(self, valid_mask, percentile_clip=(2, 98)):
        """
        Remove statistical outliers from the depth map.
        
        Args:
            valid_mask: Boolean mask of valid pixels
            percentile_clip: Percentiles to use for clipping outliers (min, max)
        """
        # Get valid depth values
        valid_depth = self.depth_map[valid_mask]
        
        # Find outliers using percentiles
        lower_bound = np.percentile(valid_depth, percentile_clip[0])
        upper_bound = np.percentile(valid_depth, percentile_clip[1])
        
        # Create outlier mask
        outliers = (self.depth_map < lower_bound) | (self.depth_map > upper_bound)
        outliers = outliers & valid_mask  # Only consider outliers in valid regions
        
        # Report outliers
        outlier_count = np.sum(outliers)
        if outlier_count > 0:
            print(f"Removed {outlier_count} outliers ({outlier_count/np.sum(valid_mask)*100:.2f}% of valid pixels)")
            
            # Set outliers to zero
            self.depth_map[outliers] = 0
            
    def _robust_normalize(self, valid_mask, percentile_clip=(2, 98)):
        """
        Normalize depth values robustly, ignoring outliers.
        
        Args:
            valid_mask: Boolean mask of valid pixels
            percentile_clip: Percentiles to use for normalization (min, max)
        """
        valid_depth = self.depth_map[valid_mask]
        
        # Use percentiles instead of min/max to avoid influence from extreme outliers
        min_depth = np.percentile(valid_depth, percentile_clip[0])
        max_depth = np.percentile(valid_depth, percentile_clip[1])
        
        if max_depth > min_depth:  # Avoid division by zero
            # Normalize to [0, 1] range
            normalized_depth = (valid_depth - min_depth) / (max_depth - min_depth)
            # Clip to ensure we're in [0, 1] range even for values outside percentile range
            normalized_depth = np.clip(normalized_depth, 0, 1)
            self.depth_map[valid_mask] = normalized_depth
            print(f"Normalized depth values to [0, 1] range using percentiles {percentile_clip}")
        
    def _fill_small_holes(self, max_hole_size=10):
        """
        Fill small holes in the depth map.
        
        Args:
            max_hole_size: Maximum size of holes to fill
        """
        # Create a binary mask of holes (zero values)
        holes = (self.depth_map == 0).astype(np.uint8)
        
        # Label connected components
        num_labels, labels = cv2.connectedComponents(holes)
        
        # Count pixels in each component
        for label in range(1, num_labels):
            component_size = np.sum(labels == label)
            
            if component_size <= max_hole_size:
                # This is a small hole we want to fill
                hole_pixels = labels == label
                
                # Dilate to find boundary pixels
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(hole_pixels.astype(np.uint8), kernel)
                
                # Find boundary pixels (dilated area excluding the hole itself)
                boundary = dilated.astype(bool) & (~hole_pixels)
                
                if np.any(boundary):
                    # Calculate the mean depth of boundary pixels
                    mean_boundary_depth = np.mean(self.depth_map[boundary])
                    
                    # Fill the hole with the mean depth
                    self.depth_map[hole_pixels] = mean_boundary_depth
    
    def process_metric_depth(self, depth_map, mask=None):
        """
        Process a metric depth map with minimal modifications.
        For metric depth maps, we want to preserve the actual depth values.
        
        Args:
            depth_map: 2D numpy array with metric depth values
            mask: Optional binary mask to filter valid regions
            normal_map: Optional normal map to correct depth inaccuracies
            camera_intrinsics: Optional camera intrinsics for correct projection
            
        Returns:
            numpy.ndarray: Processed depth map
        """
        self.depth_map = depth_map.copy()
        
        # Apply mask if provided to isolate valid regions
        if mask is not None:
            self.depth_map = self.depth_map * (mask > 0)
        
        # Ensure no negative or invalid depth values
        self.depth_map = np.maximum(self.depth_map, 0)
        
        # Fill small holes (optional)
        #self._fill_small_holes(max_hole_size=15)
        
        # Apply bilateral filtering to reduce noise while preserving edges
        print("Using bilateral filtering for noise reduction...")
        if self.selective_gpu:
            print("Using GPU for bilateral filtering...")
            filtered_depth = gpu_enabled_bilateral_filter(
                self.depth_map, 
                d=7,  # Diameter of each pixel neighborhood
                sigma_color=0.05,  # Filter sigma in the color space
                sigma_space=2.0  # Filter sigma in the coordinate space
            )
        else:
            filtered_depth = cv2.bilateralFilter(
                self.depth_map.astype(np.float32), 
                d=7,  # Diameter of each pixel neighborhood
                sigmaColor=0.05,  # Filter sigma in the color space
                sigmaSpace=2.0  # Filter sigma in the coordinate space
            )
        
        # Compute depth statistics for reporting
        valid_pixels = self.depth_map > 0
        if np.any(valid_pixels):
            min_depth = np.min(self.depth_map[valid_pixels])
            max_depth = np.max(self.depth_map[valid_pixels])
            mean_depth = np.mean(self.depth_map[valid_pixels])
            print(f"Depth statistics - min: {min_depth:.3f}m, max: {max_depth:.3f}m, mean: {mean_depth:.3f}m")
        
        # Return the filtered depth map
        return filtered_depth

    def correct_depth_with_normals(self, depth_map, normal_map, mask=None, 
                                   camera_intrinsics=None, smoothness_weight=0.8, iterations=3):
        """
        Correct depth map inaccuracies using normal map information.
        Particularly improves flat areas by using normal consistency.
        
        Args:
            depth_map: 2D numpy array with metric depth values
            normal_map: 3D numpy array with normal vectors [H, W, 3]
            mask: Optional binary mask for valid regions
            camera_intrinsics: Camera intrinsics object with fx, fy, cx, cy
            smoothness_weight: Weight of normal-based correction (0-1)
            iterations: Number of refinement iterations
            
        Returns:
            numpy.ndarray: Corrected depth map
        """
        if normal_map is None:
            print("No normal map provided for depth correction.")
            return depth_map
            
        print("Correcting depth map using normal information...")
        
        # Create a copy to work with
        corrected_depth = depth_map.copy()
        
        # Get dimensions
        height, width = depth_map.shape
        
        # Create valid pixel mask
        if mask is not None:
            valid_mask = mask > 0
        else:
            valid_mask = np.ones_like(depth_map, dtype=bool)
        
        # Further filter by valid depth
        valid_mask = valid_mask & (depth_map > 0)
        
        # Get camera intrinsics for depth-to-point conversion
        if camera_intrinsics is not None:
            fx = camera_intrinsics.fx
            fy = camera_intrinsics.fy
            cx = camera_intrinsics.cx
            cy = camera_intrinsics.cy
            print(f"Using provided camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        else:
            # Use reasonable defaults if not provided
            fx = 525.0
            fy = 525.0
            cx = width / 2
            cy = height / 2
            print(f"Using default camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        
        # Create pixel coordinate grids
        v, u = np.indices((height, width)).astype(np.float32)
        
        # Identify regions with consistent normals (likely flat surfaces)
        print("Identifying flat regions using normal consistency...")
        
        # Calculate normal consistency in local neighborhoods
        kernel_size = 5
        normal_consistency = np.zeros_like(depth_map)
        
        # For each valid pixel
        for y in range(kernel_size, height - kernel_size, kernel_size // 2):
            for x in range(kernel_size, width - kernel_size, kernel_size // 2):
                if not valid_mask[y, x]:
                    continue
                    
                # Get neighborhood
                y_min, y_max = max(0, y - kernel_size), min(height, y + kernel_size + 1)
                x_min, x_max = max(0, x - kernel_size), min(width, x + kernel_size + 1)
                
                # Get center normal
                center_normal = normal_map[y, x]
                
                # Get neighborhood normals
                neighborhood = normal_map[y_min:y_max, x_min:x_max]
                
                # Calculate dot products with center normal
                # Reshape center normal to [1, 1, 3] for broadcasting
                center_normal = center_normal.reshape(1, 1, 3)
                dot_products = np.sum(neighborhood * center_normal, axis=2)
                
                # Calculate consistency (higher is more consistent)
                consistency = np.mean(np.abs(dot_products))
                
                # Mark this region's consistency
                normal_consistency[y_min:y_max, x_min:x_max] = np.maximum(
                    normal_consistency[y_min:y_max, x_min:x_max],
                    consistency
                )
        
        # Normalize consistency scores
        if np.max(normal_consistency) > 0:
            normal_consistency = normal_consistency / np.max(normal_consistency)
        
        # Visualize normal consistency (for debugging)
        plt.figure(figsize=(10, 8))
        plt.imshow(normal_consistency, cmap='viridis')
        plt.colorbar(label='Normal Consistency')
        plt.title('Normal Consistency Map (Higher = More Flat)')
        plt.savefig('normal_consistency.png', bbox_inches='tight')
        plt.close()
        
        # Iterative correction
        for iteration in range(iterations):
            print(f"Depth correction iteration {iteration+1}/{iterations}")
            
            # Convert depth to 3D points
            z = corrected_depth
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            # For each pixel
            for y_idx in range(kernel_size, height - kernel_size):
                for x_idx in range(kernel_size, width - kernel_size):
                    if not valid_mask[y_idx, x_idx]:
                        continue
                        
                    # Skip pixels in non-flat regions (low consistency)
                    if normal_consistency[y_idx, x_idx] < 0.7:  # Threshold for flatness
                        continue
                    
                    # Get neighborhood
                    y_min, y_max = max(0, y_idx - kernel_size), min(height, y_idx + kernel_size + 1)
                    x_min, x_max = max(0, x_idx - kernel_size), min(width, x_idx + kernel_size + 1)
                    
                    # Create local mask for valid pixels
                    local_mask = valid_mask[y_min:y_max, x_min:x_max]
                    if not np.any(local_mask):
                        continue
                    
                    # Get the surface normal at this pixel
                    normal = normal_map[y_idx, x_idx]
                    
                    # Get 3D points in neighborhood
                    local_z = z[y_min:y_max, x_min:x_max][local_mask]
                    local_x = x[y_min:y_max, x_min:x_max][local_mask]
                    local_y = y[y_min:y_max, x_min:x_max][local_mask]
                    
                    # Stack into points
                    local_points = np.stack((local_x, local_y, local_z), axis=1)
                    
                    if len(local_points) < 3:  # Need at least 3 points for a plane
                        continue
                    
                    # Compute centroid of local points
                    centroid = np.mean(local_points, axis=0)
                    
                    # Current point
                    current_point = np.array([x[y_idx, x_idx], y[y_idx, x_idx], z[y_idx, x_idx]])
                    
                    # Project current point onto the plane defined by centroid and normal
                    # The plane equation is: normal·(p - centroid) = 0
                    # where p is any point on the plane
                    
                    # Vector from centroid to current point
                    v_to_point = current_point - centroid
                    
                    # Distance from point to plane
                    dist_to_plane = np.dot(v_to_point, normal)
                    
                    # Projected point
                    projected_point = current_point - dist_to_plane * normal
                    
                    # Update depth using a weighted combination of original and corrected depth
                    # Weight by normal consistency - more consistent = more correction
                    weight = smoothness_weight * normal_consistency[y_idx, x_idx]
                    corrected_depth[y_idx, x_idx] = (1 - weight) * corrected_depth[y_idx, x_idx] + \
                                                   weight * projected_point[2]
        
        # Apply median filter to smooth results
        corrected_depth = cv2.medianBlur(
            corrected_depth.astype(np.float32), 
            ksize=3
        )
        
        # Visualize the correction (for debugging)
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(depth_map, cmap='viridis')
        plt.title('Original Depth')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(corrected_depth, cmap='viridis')
        plt.title('Corrected Depth')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        diff = corrected_depth - depth_map
        plt.imshow(diff, cmap='RdBu')
        plt.title('Difference (Red = Added, Blue = Removed)')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('depth_correction.png', bbox_inches='tight')
        plt.close()
        
        # Report statistics on changes
        valid_diff = diff[valid_mask]
        if len(valid_diff) > 0:
            print(f"Depth correction stats: min={valid_diff.min():.6f}m, max={valid_diff.max():.6f}m, mean={valid_diff.mean():.6f}m")
            print(f"Absolute changes: mean={np.abs(valid_diff).mean():.6f}m, median={np.median(np.abs(valid_diff)):.6f}m")
        
        return corrected_depth
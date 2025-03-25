import numpy as np
import cv2
from skimage import filters, morphology

class DepthProcessor:
    """
    Class for processing depth maps before volumetric reconstruction.
    """
    def __init__(self):
        self.depth_map = None
        
    def process(self, depth_map, mask=None, invert_depth=True, normalize=True, 
                remove_outliers=True, percentile_clip=(2, 98)):
        """
        Process the depth map to prepare it for volumetric reconstruction.
        
        Args:
            depth_map: 2D numpy array with depth values
            mask: Optional binary mask to filter valid regions
            invert_depth: Whether to invert depth values (if TRUE means larger values are closer to camera)
            normalize: Whether to normalize depth values to [0, 1] range
            remove_outliers: Whether to remove statistical outliers
            percentile_clip: Percentiles to use for clipping outliers (min, max)
            
        Returns:
            numpy.ndarray: Processed depth map
        """
        self.depth_map = depth_map.copy()
        
        # Improve mask if provided
        if mask is not None:
            # Clean up the mask with morphological operations
            cleaned_mask = self._clean_mask(mask)
            valid_mask = cleaned_mask > 0
        else:
            valid_mask = np.ones_like(self.depth_map, dtype=bool)
        
        # Further filter by valid depth values
        valid_mask = valid_mask & (self.depth_map > 0)
        
        # Check if we have valid pixels
        if not np.any(valid_mask):
            print("WARNING: No valid depth values found after masking!")
            return self.depth_map
        
        # Remove statistical outliers if requested
        if remove_outliers:
            self._remove_statistical_outliers(valid_mask, percentile_clip)
            # Update valid mask
            valid_mask = valid_mask & (self.depth_map > 0)
            
        # Check if we need to invert the depth values
        if invert_depth:
            # Only invert the valid pixels
            valid_depth = self.depth_map[valid_mask]
            max_depth = np.max(valid_depth)
            self.depth_map[valid_mask] = max_depth - valid_depth + 0.001  # Add small offset to avoid zeros
            print(f"Inverted depth values: new range [{np.min(self.depth_map[valid_mask])}, {np.max(self.depth_map[valid_mask])}]")
            
        # Normalize depth values if requested
        if normalize:
            self._robust_normalize(valid_mask, percentile_clip)
            
        # Remove noise with bilateral filter (preserves edges)
        # Only filter the valid regions
        filtered_depth = cv2.bilateralFilter(
            self.depth_map.astype(np.float32), 
            d=7,  # Diameter of each pixel neighborhood
            sigmaColor=0.05,  # Filter sigma in the color space
            sigmaSpace=2.0  # Filter sigma in the coordinate space
        )
        self.depth_map = filtered_depth
        
        # Fill small holes using morphological operations
        self._fill_small_holes()
        
        # Ensure no negative depth values
        self.depth_map = np.maximum(self.depth_map, 0)
        
        # Apply mask again to ensure only valid regions are kept
        if mask is not None:
            self.depth_map = self.depth_map * (valid_mask)
        
        return self.depth_map
        
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
            if component_size < 100:  # Remove components smaller than 100 pixels
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
                # Small hole, fill it with local median
                hole_pixels = labels == label
                
                # Dilate to get surrounding pixels
                kernel = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(hole_pixels.astype(np.uint8), kernel, iterations=1)
                
                # Get boundary pixels (dilated - original)
                boundary = dilated.astype(bool) & (~hole_pixels)
                
                if np.any(boundary):
                    # Get median of boundary pixels
                    fill_value = np.median(self.depth_map[boundary])
                    
                    # Fill hole
                    self.depth_map[hole_pixels] = fill_value
    
    def process_metric_depth(self, depth_map, mask=None):
        """
        Process a metric depth map (like 32-bit TIFF) with minimal modifications.
        For metric depth maps, we want to preserve the actual depth values.
        
        Args:
            depth_map: 2D numpy array with metric depth values
            mask: Optional binary mask to filter valid regions
            
        Returns:
            numpy.ndarray: Processed depth map
        """
        self.depth_map = depth_map.copy()
        
        # Apply mask if provided to isolate valid regions
        if mask is not None:
            self.depth_map = self.depth_map * (mask > 0)
        
        # For 32-bit float depth maps, ensure no negative depths
        # but use a small threshold to avoid removing legitimate very small values
        min_depth_threshold = 0.001  # 1mm minimum depth
        self.depth_map[self.depth_map < min_depth_threshold] = 0
        
        # Report depth statistics
        valid_depth = self.depth_map > min_depth_threshold
        if np.any(valid_depth):
            min_depth = np.min(self.depth_map[valid_depth])
            max_depth = np.max(self.depth_map[valid_depth])
            mean_depth = np.mean(self.depth_map[valid_depth])
            std_depth = np.std(self.depth_map[valid_depth])
            print(f"Depth statistics - min: {min_depth:.6f}m, max: {max_depth:.6f}m, mean: {mean_depth:.6f}m, std: {std_depth:.6f}m")
        
        # Optional light bilateral filtering to reduce noise while preserving edges
        # For 32-bit float depth maps, we need to be careful with the filter parameters
        if np.any(valid_depth):
            # Use depth statistics to determine appropriate filter parameters
            depth_range = max_depth - min_depth
            sigma_color = depth_range * 0.01  # 1% of depth range
            
            filtered_depth = cv2.bilateralFilter(
                self.depth_map.astype(np.float32), 
                d=5,  # Smaller neighborhood for less aggressive filtering
                sigmaColor=sigma_color,
                sigmaSpace=2.0
            )
            
            # Only update values where we had valid depth before
            self.depth_map[valid_depth] = filtered_depth[valid_depth]
        
        return self.depth_map
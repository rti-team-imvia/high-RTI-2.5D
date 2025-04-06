import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters, morphology
from utils.gpu_utils import check_gpu_availability, gpu_enabled_bilateral_filter
import os

class DepthProcessor:
    """
    Class for processing depth maps before volumetric reconstruction.
    """
    def __init__(self):
        self.depth_map = None
        self.normal_map = None
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
    
    def _remove_depth_discontinuities(self, depth_threshold_factor=2.0, acceleration_factor=1.5):
        """
        Remove problematic pixels with large depth discontinuities, using both 
        gradient (first derivative) and acceleration (second derivative) analysis.
        
        Args:
            depth_threshold_factor: Multiplier for gradient threshold
            acceleration_factor: Multiplier for acceleration threshold
        
        Returns:
            Tuple: (cleaned_depth_map, visualization_data)
        """
        print("Performing enhanced depth discontinuity detection with acceleration analysis...")
        
        # Create valid pixel mask
        valid_mask = self.depth_map > 0
        
        # --- DEPTH GRADIENT (FIRST DERIVATIVE) ANALYSIS ---
        
        # Calculate depth gradient using Sobel operators
        grad_x = cv2.Sobel(self.depth_map, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(self.depth_map, cv2.CV_32F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate gradient statistics
        valid_gradients = gradient_magnitude[valid_mask]
        mean_gradient = np.mean(valid_gradients)
        std_gradient = np.std(valid_gradients)
        
        # Define threshold for discontinuities
        gradient_threshold = mean_gradient + depth_threshold_factor * std_gradient
        
        # --- DEPTH ACCELERATION (SECOND DERIVATIVE) ANALYSIS ---
        
        # Calculate Laplacian (sum of second derivatives in x and y)
        laplacian = cv2.Laplacian(self.depth_map, cv2.CV_32F, ksize=5)
        acceleration_magnitude = np.abs(laplacian)
        
        # Calculate acceleration statistics
        valid_acceleration = acceleration_magnitude[valid_mask]
        mean_acceleration = np.mean(valid_acceleration)
        std_acceleration = np.std(valid_acceleration)
        
        # Define threshold for acceleration discontinuities
        acceleration_threshold = mean_acceleration + acceleration_factor * std_acceleration
        
        # --- COMBINED ANALYSIS ---
        
        # Identify areas with high gradient
        gradient_problem_areas = gradient_magnitude > gradient_threshold
        gradient_problem_areas = gradient_problem_areas & valid_mask
        
        # Identify areas with high acceleration
        acceleration_problem_areas = acceleration_magnitude > acceleration_threshold
        acceleration_problem_areas = acceleration_problem_areas & valid_mask
        
        # Combine problem areas (either high gradient OR high acceleration)
        problem_areas = gradient_problem_areas | acceleration_problem_areas
        
        # Dilate to capture neighboring uncertain points (which cause stretching)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        problem_areas_dilated = cv2.dilate(problem_areas.astype(np.uint8), kernel).astype(bool)
        
        # Get count of removed pixels
        removed_count = np.sum(problem_areas_dilated & valid_mask)
        if removed_count > 0:
            percent_removed = removed_count / np.sum(valid_mask) * 100
            print(f"Removing {removed_count} problematic pixels ({percent_removed:.2f}% of valid pixels)")
            print(f"  - High gradient areas: {np.sum(gradient_problem_areas & valid_mask)} pixels")
            print(f"  - High acceleration areas: {np.sum(acceleration_problem_areas & valid_mask)} pixels")
            print(f"  - Additional from dilation: {removed_count - np.sum(problem_areas & valid_mask)} pixels")
        
        # Apply the filter
        cleaned_depth = self.depth_map.copy()
        cleaned_depth[problem_areas_dilated] = 0
        
        # Create visualization data
        vis_data = {
            'original_depth': self.depth_map.copy(),
            'cleaned_depth': cleaned_depth,
            'gradient_magnitude': gradient_magnitude,
            'acceleration_magnitude': acceleration_magnitude,
            'gradient_problem_areas': gradient_problem_areas,
            'acceleration_problem_areas': acceleration_problem_areas,
            'problem_areas': problem_areas,
            'problem_areas_dilated': problem_areas_dilated
        }
        
        # Update the depth map
        self.depth_map = cleaned_depth
        
        return cleaned_depth, vis_data
    
    def _visualize_discontinuity_removal(self, vis_data, save_path=None):
        """Create detailed visualization of removed depth discontinuities with acceleration analysis."""
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        
        # Original and cleaned depth
        axes[0, 0].imshow(vis_data['original_depth'], cmap='viridis')
        axes[0, 0].set_title('Original Depth Map')
        
        axes[0, 1].imshow(vis_data['cleaned_depth'], cmap='viridis')
        axes[0, 1].set_title('Cleaned Depth Map')
        
        # Gradient and acceleration
        axes[1, 0].imshow(vis_data['gradient_magnitude'], cmap='plasma')
        axes[1, 0].set_title('Depth Gradient (First Derivative)')
        
        axes[1, 1].imshow(vis_data['acceleration_magnitude'], cmap='plasma')
        axes[1, 1].set_title('Depth Acceleration (Second Derivative)')
        
        # Problem areas
        combined_problems = np.zeros_like(vis_data['original_depth'])
        combined_problems[vis_data['gradient_problem_areas']] = 1  # Red for gradient issues
        combined_problems[vis_data['acceleration_problem_areas']] = 2  # Yellow for acceleration issues
        
        axes[2, 0].imshow(combined_problems, cmap='viridis')
        axes[2, 0].set_title('Identified Issues\n(1=Gradient, 2=Acceleration)')
        
        # Final removed areas
        removed = vis_data['original_depth'].copy()
        removed[~vis_data['problem_areas_dilated']] = 0
        axes[2, 1].imshow(removed, cmap='hot')
        axes[2, 1].set_title('Removed Areas')
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved discontinuity visualization to {save_path}")
        plt.close()
    
    def process_metric_depth(self, depth_map, mask=None, normal_map=None, 
                             visualize_discontinuities=True, discontinuity_vis_path=None):
        """Process a metric depth map with discontinuity removal."""
        self.depth_map = depth_map.copy()
        self.normal_map = normal_map
        
        # Apply mask if provided
        if mask is not None:
            self.depth_map = self.depth_map * (mask > 0)
        
        # Ensure no negative values
        self.depth_map = np.maximum(self.depth_map, 0)
        
        # Remove depth discontinuities
        if normal_map is not None:
            print("Using enhanced depth discontinuity detection...")
            
            _, vis_data = self._remove_depth_discontinuities(
                depth_threshold_factor=0.25,
                acceleration_factor=1
            )
            
            # Visualize what was removed
            if visualize_discontinuities and vis_data is not None:
                vis_path = discontinuity_vis_path or os.path.join('data', 'output', 'discontinuity_removal.png')
                self._visualize_discontinuity_removal(vis_data, save_path=vis_path)
                print(f"Created discontinuity removal visualization at: {vis_path}")
        
        # Apply bilateral filtering to reduce noise while preserving edges
        print("Using bilateral filtering for noise reduction...")
        if self.selective_gpu:
            print("Using GPU for bilateral filtering...")
            filtered_depth = gpu_enabled_bilateral_filter(
                self.depth_map, 
                d=25,  # Diameter of each pixel neighborhood
                sigma_color=0.05,  # Filter sigma in the color space
                sigma_space=15  # Filter sigma in the coordinate space
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
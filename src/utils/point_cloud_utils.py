import numpy as np
import open3d as o3d

def calculate_normal_consistency(point_cloud, radius=0.05, max_nn=30):
    """
    Calculate normal consistency for each point in a point cloud.
    This measures how consistent the normals are in a local neighborhood,
    which helps identify flat or smooth regions.
    
    Args:
        point_cloud: open3d.geometry.PointCloud with normals
        radius: Radius for neighborhood search
        max_nn: Maximum nearest neighbors to consider
        
    Returns:
        numpy.ndarray: Normal consistency scores (0-1)
    """
    if not point_cloud.has_normals():
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
    
    # Build KD-tree for neighborhood queries
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    
    # Get points and normals as numpy arrays
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    
    # Calculate consistency for each point
    consistency_scores = np.zeros(len(points))
    
    for i in range(len(points)):
        # Find neighbors
        _, idx, _ = kdtree.search_hybrid_vector_3d(points[i], radius, max_nn)
        if len(idx) < 3:
            continue
            
        # Get neighbor normals
        neighbor_normals = normals[idx]
        
        # Calculate dot products between center normal and all neighbors
        center_normal = normals[i]
        alignments = np.abs(np.dot(neighbor_normals, center_normal))
        
        # Average alignment score (1 = perfectly aligned, 0 = perpendicular)
        consistency_scores[i] = np.mean(alignments)
    
    return consistency_scores

def filter_point_cloud(point_cloud, min_neighbors=2, radius=0.02):
    """
    Filter isolated points from a point cloud.
    
    Args:
        point_cloud: open3d.geometry.PointCloud
        min_neighbors: Minimum number of neighbors a point must have
        radius: Search radius for neighbors
        
    Returns:
        open3d.geometry.PointCloud: Filtered point cloud
    """
    # Create a copy of the point cloud
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points))
    
    if point_cloud.has_normals():
        filtered_cloud.normals = o3d.utility.Vector3dVector(np.asarray(point_cloud.normals))
    
    if point_cloud.has_colors():
        filtered_cloud.colors = o3d.utility.Vector3dVector(np.asarray(point_cloud.colors))
    
    # Use Open3D's built-in filter
    filtered_cloud, _ = filtered_cloud.remove_radius_outlier(
        nb_points=min_neighbors, 
        radius=radius
    )
    
    return filtered_cloud

def merge_point_clouds(point_clouds, voxel_size=0.01):
    """
    Merge multiple point clouds and remove duplicates.
    
    Args:
        point_clouds: List of open3d.geometry.PointCloud objects
        voxel_size: Size of voxels for downsampling
        
    Returns:
        open3d.geometry.PointCloud: Merged point cloud
    """
    # Create an empty point cloud
    merged_cloud = o3d.geometry.PointCloud()
    
    # Merge all point clouds
    for pcd in point_clouds:
        merged_cloud += pcd
    
    # Remove duplicates by voxel downsampling
    return merged_cloud.voxel_down_sample(voxel_size)
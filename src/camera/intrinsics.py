import numpy as np

class CameraIntrinsics:
    """
    Class to handle camera intrinsic parameters.
    """
    def __init__(self, fx=2252.31, fy=2252.31, cx=1352.0, cy=900.0, width=2704, height=1800):
        """
        Initialize camera intrinsics with default or provided values.
        
        Args:
            fx: Focal length in x direction (pixels)
            fy: Focal length in y direction (pixels)
            cx: Principal point x-coordinate (pixels)
            cy: Principal point y-coordinate (pixels)
            width: Image width in pixels
            height: Image height in pixels
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        
    def to_matrix(self):
        """
        Create the 3x3 camera intrinsic matrix.
        
        Returns:
            numpy.ndarray: 3x3 intrinsic matrix
        """
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        
    @classmethod
    def from_matrix(cls, matrix, width=1920, height=1080):
        """
        Create a CameraIntrinsics object from an intrinsic matrix.
        
        Args:
            matrix: 3x3 intrinsic matrix
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            CameraIntrinsics: New camera intrinsics object
        """
        fx = matrix[0, 0]
        fy = matrix[1, 1]
        cx = matrix[0, 2]
        cy = matrix[1, 2]
        return cls(fx, fy, cx, cy, width, height)
        
    @classmethod
    def from_dict(cls, params):
        """
        Create a CameraIntrinsics object from a dictionary.
        
        Args:
            params: Dictionary with keys 'fx', 'fy', 'cx', 'cy', 'width', 'height'
            
        Returns:
            CameraIntrinsics: New camera intrinsics object
        """
        fx = params.get('fx', 2252.31)
        fy = params.get('fy', 2252.31)
        cx = params.get('cx', 1352.0)
        cy = params.get('cy', 900.0)
        width = params.get('width', 2704)
        height = params.get('height', 1800)
        return cls(fx, fy, cx, cy, width, height)
        
    def to_dict(self):
        """
        Convert to a dictionary representation.
        
        Returns:
            dict: Dictionary with camera intrinsic parameters
        """
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'width': self.width,
            'height': self.height
        }
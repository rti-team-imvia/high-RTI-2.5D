# Depth-to-Volume: 3D Reconstruction from Metric Depth Maps

This project converts depth maps into 3D volumetric meshes using camera intrinsic parameters. It's specifically optimized for metric monocular depth maps, including high-precision 32-bit TIFF depth images.

## Features

- Direct processing of metric depth maps (meters)
- Support for high-precision 32-bit TIFF depth files
- Camera-calibrated 3D reconstruction
- Robust point cloud generation and mesh creation
- Multiple visualization and debugging tools
- No artificial depth inversion or scaling required

## Project Structure

```
depth-to-volume
├── src
│   ├── main.py                # Entry point of the application
│   ├── camera
│   │   ├── __init__.py        # Initializes the camera module
│   │   └── intrinsics.py      # Handles camera intrinsic parameters
│   ├── processing
│   │   ├── __init__.py        # Initializes the processing module
│   │   ├── depth_processor.py  # Processes metric depth maps
│   │   └── volume_generator.py # Converts processed depth to 3D mesh
│   └── utils
│       ├── __init__.py        # Initializes the utilities module
│       ├── file_io.py         # Handles file input/output operations
│       └── visualization.py   # Contains visualization functions
├── data
│   ├── input                  # Place input depth maps and masks here
│   └── output                 # Generated 3D meshes and visualizations
└── requirements.txt           # Project dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/depth-to-volume.git
   cd depth-to-volume
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Input Data Requirements

This tool works best with:

- **Metric depth maps**: Values represent actual distances in meters
- **Supported formats**: 
  - 32-bit TIFF files (preferred for metric depth)
  - 16-bit PNG (values typically in mm, automatically converted to meters)
  - 8-bit depth maps (less accurate)
- **Camera intrinsics**: JSON file with camera parameters (fx, fy, cx, cy)
- **Optional mask**: Binary image marking the regions of interest

## Usage

1. Place your depth map (e.g., `depth.tiff`) in the `data/input` directory
2. If available, add a mask file (e.g., `mask.png`) to the same directory
3. Create a camera intrinsics file (e.g., `intrinsics.json`) with the following format:
   ```json
   {
     "fx": 525.0,
     "fy": 525.0,
     "cx": 319.5,
     "cy": 239.5,
     "width": 640,
     "height": 480
   }
   ```
4. Run the application:
   ```bash
   python src/main.py
   ```
5. The generated 3D mesh and visualizations will be saved in the `data/output` directory

## Output Files

- `reconstructed_mesh.obj`: 3D mesh generated from the depth map
- `depth_visualization.png`: Visualization of the input depth map
- `processed_depth_visualization.png`: Visualization after depth processing
- `visualizations/`: Directory containing additional visualizations:
  - Rendered views of the 3D mesh
  - Point cloud visualization
  - Camera-point relationship debugging view

## Key Parameters

- **Voxel Size**: Controls mesh detail level (smaller = more detail but slower)
- **Depth Threshold**: Minimum valid depth value (default: 1mm)
- **Normal Estimation**: Affects surface smoothness and orientation

## Dependencies

- NumPy: Numerical processing
- OpenCV: Image processing
- Open3D: 3D geometry processing and visualization
- Matplotlib: 2D visualization
- TiffFile: Handling 32-bit TIFF files
- Scikit-image: Advanced image processing

## Troubleshooting

- **Mesh quality issues**: Adjust voxel size or try uncommenting the statistical outlier removal code
- **Empty results**: Check if your depth map has valid metric values and proper camera intrinsics
- **Distorted mesh**: Ensure depth values are actually in meters, not millimeters or other units

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

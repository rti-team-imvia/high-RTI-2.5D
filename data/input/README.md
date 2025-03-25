# This file contains information about the input data format and how to prepare the depth map for processing.

## Input Data Format

The input data for this project consists of depth maps that are used to create a 2.5D volumetric representation. The depth maps should be provided in a suitable format, such as:

- **Image Format**: PNG, JPEG, or TIFF
- **Data Type**: 16-bit grayscale images are preferred for depth representation.

## Preparing the Depth Map

To prepare the depth map for processing, follow these steps:

1. **Capture the Depth Map**: Use a camera or depth sensor to capture the scene. Ensure that the depth map is aligned with the RGB image if available.

2. **Format the Depth Map**: Convert the depth map to a 16-bit grayscale image if it is not already in this format. This can be done using image processing software or libraries.

3. **Normalization**: Normalize the depth values to ensure they are within the expected range. The depth values should represent the distance from the camera to the objects in the scene.

4. **Save the Depth Map**: Save the processed depth map in the `data/input` directory of the project.

## Example

An example of a properly formatted depth map file might be named `depth_map.png` and located in the `data/input` directory. Ensure that the file is accessible by the application when running the processing pipeline.
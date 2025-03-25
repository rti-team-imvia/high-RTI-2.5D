# Input Data Format and Preparation

## Supported Depth Map Formats

This tool accepts the following depth map formats:

- **32-bit TIFF files** (preferred): Contains floating-point metric depth values in meters
- **16-bit PNG/TIFF files**: Typically contain depth in millimeters (will be converted to meters)
- **8-bit images**: Less accurate depth representation, typically for visualization

## Camera Intrinsics File

The `intrinsics.json` file should be formatted as follows:

```json
{
  "fx": 525.0,  // Focal length in x-direction (pixels)
  "fy": 525.0,  // Focal length in y-direction (pixels)
  "cx": 319.5,  // Principal point x-coordinate (pixels)
  "cy": 239.5,  // Principal point y-coordinate (pixels)
  "width": 640, // Image width (pixels)
  "height": 480 // Image height (pixels)
}
```

## Masks (Optional)

A binary mask image (`mask.png`) can be provided to isolate the region of interest:
- White pixels (255) indicate valid regions
- Black pixels (0) indicate regions to ignore

## File Naming

Standard file names expected by the application:
- `depth.tiff` or `depth.png`: Depth map
- `mask.png`: Optional mask
- `intrinsics.json`: Camera parameters

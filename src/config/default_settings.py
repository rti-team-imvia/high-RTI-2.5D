# default_settings.py

# This file contains default configuration settings for the project.

DEFAULT_DEPTH_MAP_PATH = "data/input/depth_map.png"
DEFAULT_OUTPUT_VOLUME_PATH = "data/output/volume_representation.npy"
DEFAULT_PROCESSING_SCALE = 1.0
DEFAULT_VOLUME_RESOLUTION = (256, 256, 256)  # Width, Height, Depth
DEFAULT_BACKGROUND_COLOR = (0, 0, 0)  # RGB
DEFAULT_NORMALIZATION_METHOD = "min-max"  # Options: 'min-max', 'z-score'
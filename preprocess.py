import cv2
import numpy as np
from pathlib import Path
import os

def remove_background_to_black(image_path, output_path=None):
    """
    Convert leaf image background to black while preserving the leaf.

    Args:
        image_path: Path to input image
        output_path: Path to save output image (optional)

    Returns:
        Image with black background
    """
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Create a copy for output
    result = img.copy()

    # Convert to HSV color space (better for color segmentation)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for green colors (leaves are typically green)
    # This covers various shades of green
    lower_green1 = np.array([25, 20, 20])
    upper_green1 = np.array([95, 255, 255])

    # Create mask for green areas
    mask = cv2.inRange(hsv, lower_green1, upper_green1)

    # Also consider brown/yellow areas (some leaves have these)
    lower_brown = np.array([10, 20, 20])
    upper_brown = np.array([25, 255, 255])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Combine masks
    mask = cv2.bitwise_or(mask, mask_brown)

    # Apply morphological operations to clean up the mask
    # Close small holes in the leaf
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill any remaining holes inside the leaf
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new mask with filled contours
    filled_mask = np.zeros_like(mask)
    if contours:
        # Find the largest contour (assume it's the leaf)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled_mask, [largest_contour], -1, 255, -1)

    # Optional: Smooth the edges
    filled_mask = cv2.GaussianBlur(filled_mask, (5, 5), 0)
    _, filled_mask = cv2.threshold(filled_mask, 127, 255, cv2.THRESH_BINARY)

    # Create black background
    black_bg = np.zeros_like(img)

    # Use the mask to combine leaf with black background
    mask_3channel = cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR)
    result = np.where(mask_3channel == 255, img, black_bg)

    # Save the result if output path is provided
    if output_path:
        cv2.imwrite(str(output_path), result)
        print(f"Saved result to: {output_path}")

    return result, filled_mask



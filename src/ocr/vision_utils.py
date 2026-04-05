"""
vision_utils.py
===============
Computer Vision utilities for preprocessing "Dark Data" medical records.

This module implements the physical separation of overlapping layers 
(e.g., Red/Blue stamps overlapping with dark handwritten ink) using 
HSV color-space manipulation, ensuring the downstream VLM is never 
confused by overlapping artifacts.
"""

import cv2
import numpy as np
from PIL import Image

class ClinicalImageEnhancer:
    """
    Pre-processes medical scans to separate the 'Stamp Layer' from the 'Ink Layer'.
    """

    @staticmethod
    def remove_stamps_from_image(pil_image: Image.Image) -> Image.Image:
        """
        Uses HSV color-space masking to remove colored stamps (red/blue) 
        and leaves the black/dark-blue handwriting intact.
        
        Args:
            pil_image: The original PIL Image containing the document.
            
        Returns:
            A cleaned PIL Image with the stamps neutralized (turned white).
        """
        # Convert PIL Image (RGB) to OpenCV format (BGR) for processing
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR 
        img = open_cv_image[:, :, ::-1].copy()

        # Convert to HSV color space for accurate color targeting
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 1. Isolate the Red Stamp Layer (Red wraps around the HSV spectrum)
        lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)
        
        # 2. Isolate the Blue Stamp Layer (High saturation blue)
        lower_blue, upper_blue = np.array([100, 150, 50]), np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 3. Combine masks to target all stamps
        stamp_mask = cv2.bitwise_or(red_mask, blue_mask)
        
        # 4. Dilate the mask slightly to remove color bleeding/fuzzy edges around stamps
        kernel = np.ones((3,3), np.uint8)
        stamp_mask = cv2.dilate(stamp_mask, kernel, iterations=1)
        
        # 5. Neutralize the targeted stamp pixels (turn them white)
        img[stamp_mask > 0] = [255, 255, 255]
        
        # Convert back to RGB and then to PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
import numpy as np
import cv2


def create_mask(image_shape, roi):
    """Create a mask based on ROI

    Args:
        image_shape: Shape of the image (height, width, channels)
        roi: Region of interest [x_min, y_min, x_max, y_max] in relative coordinates

    Returns:
        mask: Binary mask as numpy array
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # Convert relative coordinates to pixel coordinates
    x_min, y_min = int(roi[0] * width), int(roi[1] * height)
    x_max, y_max = int(roi[2] * width), int(roi[3] * height)

    # Set ROI region to white (255)
    mask[y_min:y_max, x_min:x_max] = 255
    return mask


def apply_mask_to_image(image, mask):
    """Apply mask to image

    Args:
        image: The original image as numpy array
        mask: Binary mask as numpy array

    Returns:
        masked_image: The masked image
    """
    masked_image = image.copy()
    return cv2.bitwise_and(masked_image, masked_image, mask=mask)

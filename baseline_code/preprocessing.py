"""
Preprocessing functions for consistent use across assignments.
Assignment 1: Classical Image Processing Pipeline
"""

import cv2
import numpy as np

def convert_to_grayscale(image):
    """Convert RGB/BGR to grayscale"""
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()

def apply_noise_reduction(image, method='median', kernel_size=5):
    """
    Apply different filters for noise reduction
    Args:
        image: Grayscale image
        method: 'gaussian', 'median', or 'bilateral'
        kernel_size: Size of the filter kernel
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif method == 'bilateral':
        # Default bilateral parameters for seed counting
        return cv2.bilateralFilter(image, kernel_size, 75, 75)
    else:
        return image

def detect_edges(image, method='canny', low_thresh=50, high_thresh=150):
    """
    Edge detection algorithms
    Args:
        image: Grayscale or filtered image
        method: 'canny' or 'sobel'
    """
    if method == 'canny':
        return cv2.Canny(image, low_thresh, high_thresh)
    elif method == 'sobel':
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        return np.uint8(np.clip(sobel, 0, 255))
    return None

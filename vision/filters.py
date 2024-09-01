import cv2
import numpy as np


def black_segmentation(image: np.ndarray) -> np.ndarray:
    """Segment black regions in the image.

    Args:
        image (np.ndarray): The input BGR image.

    Returns:
        np.ndarray: Binary image with black regions segmented.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    return th


def white_segmentation(image: np.ndarray) -> np.ndarray:
    """Segment white regions in the image.

    Args:
        image (np.ndarray): The input BGR image.

    Returns:
        np.ndarray: Binary image with white regions segmented.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)

    return th

def yellow_mask(hsv: np.ndarray) -> np.ndarray:
    """Create a mask for yellow regions in the HSV image.

    Args:
        hsv (np.ndarray): The input HSV image.

    Returns:
        np.ndarray: Binary mask where yellow regions are white.
    """
    lower = np.array([15, 70, 120])
    upper = np.array([45, 255, 255])
    return cv2.inRange(hsv, lower, upper)


def red_mask(hsv: np.ndarray) -> np.ndarray:
    """Create a mask for red regions in the HSV image.

    Args:
        hsv (np.ndarray): The input HSV image.

    Returns:
        np.ndarray: Binary mask where red regions are white.
    """
    lower = np.array([0, 70, 50])
    upper = np.array([10, 255, 255])
    return cv2.inRange(hsv, lower, upper)


def median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply median blur to the image.

    Args:
        image (np.ndarray): The input image.
        kernel_size (int): Size of the kernel. Must be an odd number.

    Returns:
        np.ndarray: Blurred image.
    """
    blur = cv2.medianBlur(image, kernel_size)
    return blur


def gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply Gaussian blur to the image.

    Args:
        image (np.ndarray): The input image.
        kernel_size (int): Size of the kernel. Must be an odd number.

    Returns:
        np.ndarray: Blurred image.
    """
    blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blur


def eroding(image: np.ndarray) -> np.ndarray:
    """Apply erode to the image.

    Args:
        image (np.ndarray): The input binary image.

    Returns:
        np.ndarray: Eroded image.
    """
    erode_kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(image, erode_kernel, iterations=1)
    return erode


def dilating(image: np.ndarray) -> np.ndarray:
    """Apply dilate to the image.

    Args:
        image (np.ndarray): The input binary image.

    Returns:
        np.ndarray: Dilated image.
    """
    dilate_kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(image, dilate_kernel, iterations=1)
    return dilate


def filter_yellow_edges(image: np.ndarray) -> np.ndarray:
    """Filter yellow edges in the image.

    Args:
        image (np.ndarray): The input BGR image.

    Returns:
        np.ndarray: Image with yellow edges detected.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur = gaussian_blur(hsv, 25)
    mask = yellow_mask(blur)
    erode = eroding(mask)

    return cv2.Canny(erode, 200, 400)


def red_line_segmentation(image: np.ndarray) -> np.ndarray:
    """Segment red lines in the image.

    Args:
        image (np.ndarray): The input BGR image.

    Returns:
        np.ndarray: Binary image with red lines segmented.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur = median_blur(hsv, 3)
    gray = red_mask(blur)

    return gray


def line_edges(image: np.ndarray) -> np.ndarray:
    """Filter only the line borders in the image.

    Args:
        image (np.ndarray): The input BGR image.

    Returns:
        np.ndarray: Image with line edges detected.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur = median_blur(hsv, 3)
    gray = red_mask(blur)
    edges = cv2.Canny(gray, 100, 200)

    return edges



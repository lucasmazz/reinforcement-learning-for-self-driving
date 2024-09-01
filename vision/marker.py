from typing import Optional
from cv2 import aruco
import numpy as np


def find_markers(image: np.ndarray) -> tuple[list[np.ndarray], list[tuple[float, float]], Optional[np.ndarray]]:
    """Track all ArUco markers in an image and return corners, centroids, and ID info.

    Args:
        image (np.ndarray): Grayscale image containing ArUco markers.

    Returns:
        Tuple[List[np.ndarray], List[Tuple[float, float]], Optional[np.ndarray]]:
            - List of detected ArUco marker corners. Each element is an array of shape (N, 1, 2), where N is the number of corners.
            - List of centroids of the detected markers. Each element is a tuple (x, y).
            - Optional array of marker IDs. If no markers are detected, this will be None.
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    parameters = aruco.DetectorParameters()
    centroids = []

    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        image, aruco_dict, parameters=parameters)
    
    if ids is not None:
        for i in range(len(ids)):
            c = corners[i][0]
            centroids.append((c[:, 0].mean(), c[:, 1].mean()))

    return corners, centroids, ids
import math
import random
import uuid
from typing import Optional
import numpy as np
from numpy.testing._private.utils import decorate_methods
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import cv2
import matplotlib.pyplot as plt
import pickle 

from vision import line_edges, find_markers, white_segmentation
from sensors import Camera


class Tracklet:

    def __init__(self, camera, debug=False):
        """Initialize a Tracklet object.

        Args:
            camera (Camera): The camera object used to capture images.
            debug (bool, optional): If True, debug information will be shown. Defaults to False.
        """
        self.camera = camera
        self.points = []
        self.tracking = []
        self.debug = debug

    @property
    def image(self) -> np.ndarray:
        """Get the current image from the camera.

        Returns:
            np.ndarray: The current camera image, flipped and converted to RGB.
        """
        im = self.camera.image
        im = cv2.flip(im, 0)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def load(self) -> None:
        """Load the tracklet by finding the center points."""
        self.find_center_points()

    def find_markers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find markers in the current image.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The marker corners, centroids, and IDs.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_centroids, marker_ids = find_markers(gray)
        return marker_corners, marker_centroids, marker_ids

    def find_closest_pair_of_points(self, array_1, array_2):
        """Find the closest pairs of points between two arrays.

        Args:
            array_1 (np.ndarray): The first array of points.
            array_2 (np.ndarray): The second array of points.

        Returns:
            list[list[np.ndarray]]: A list of closest point pairs.
        """
        closest = []

        for i, p1 in enumerate(array_1):

            min_distance = None 
            index = 0

            for j, p2 in enumerate(array_2):

                distance = math.sqrt(
                    math.pow(p1[0]-p2[0], 2) + math.pow(p1[1]-p2[1], 2))

                if min_distance is None or distance <= min_distance:
                    min_distance = distance
                    index = j

            closest.append([p1, array_2[index]])

        return closest

    def find_center_points(self) -> None:
        """Find and sample the center points of the tracklet."""
        current_image = self.image

        if current_image is not None:
            gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

            # Find tracklet center
            lines = line_edges(current_image)

            contours, hierarchy = cv2.findContours(
                lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            internal_line_points = np.array(contours[1])
            external_line_points = np.array(contours[2])
            closest_points_pair = self.find_closest_pair_of_points(
                internal_line_points.squeeze(), external_line_points.squeeze())

            # Compute and sample the center points
            self.points = np.array(
                np.average(closest_points_pair, 1), dtype=np.int32)[::5]
            
            self.points = self.points[np.argsort(self.points[:, 1])] # Sort by Y axis
            
            if self.debug:
                for point in self.points:
                    cv2.circle(current_image,
                               (point[0], point[1]), 1, (0, 0, 255), -1)

                cv2.imshow('External System', current_image)
                cv2.waitKey(1)


class MonitorSystem:

    def __init__(self, cameras: list[Camera] = [], debug: bool = False) -> None:
        """Initialize a MonitorSystem object.

        Args:
            cameras (list[Camera], optional): A list of cameras used in the system. Defaults to an empty list.
            debug (bool, optional): If True, debug information will be shown. Defaults to False.
        """
        self.cameras = cameras
        self.tracklets = []
        self.debug = debug
        self.current_tracklet_index = 0
        self.reverse = False
        
        for i, camera in enumerate(self.cameras):
            tracklet = Tracklet(camera, debug)
            self.tracklets.append(tracklet)
            
    @property
    def current_camera(self) -> Camera:
        """Get the current camera based on the tracklet index.

        Returns:
            Camera: The current camera.
        """
        return self.cameras[self.current_tracklet_index]

    def load(self) -> None:
        """Load all tracklets in the system."""
        for tracklet in self.tracklets:
            tracklet.load()

        if self.debug:
            cv2.destroyAllWindows()
            
    def reset(self, reverse: bool = False) -> None:
        """Reset the system by randomly selecting a tracklet and setting reverse mode.

        Args:
            reverse (bool, optional): If True, the system will run in reverse mode. Defaults to False.
        """
        # Change the cameras order
        self.reverse = reverse
        self.current_tracklet_index = random.randint(0, len(self.cameras) - 1)
        
    def run(self) -> tuple[Optional[float], Optional[float]]:
        """Run the monitor system to track the car.

        Returns:
            tuple[Optional[float], Optional[float]]: The distance and angle between the car and the track.
        """
        tracklet = self.tracklets[self.current_tracklet_index]
        image = tracklet.image
        marker_center = None
        distance = None
        angle = None
        
        # Find marker corners, centroids and pose estimation vectors.
        marker_corners, marker_centroids, marker_ids = tracklet.find_markers()
        
        if marker_ids is not None:
            marker_corners = np.array(marker_corners, dtype=np.int32)
            marker_centroids = np.array(marker_centroids, dtype=np.int32)
            
            # aruco.drawDetectedMarkers(image, marker_corners)
            marker_center = np.array(marker_centroids[0], dtype=np.int32)
            distance, i = self.compute_distante(marker_center, tracklet.points)
            
            car_vector = np.array(
                (marker_corners[0][0][0] + marker_corners[0][0][1])/2 - \
                (marker_corners[0][0][3] + marker_corners[0][0][2])/2,
                dtype=np.int32
            )
            
            try:
                track_vector = np.array(
                    tracklet.points[i-2] - tracklet.points[i+2], dtype=np.int32)
            # If the car is leaving the camera 
            except IndexError:
                self.increment_tracklet_index()
                return self.run()
                            
            unit_car_vector = car_vector / np.linalg.norm(car_vector)
            unit_track_vector = track_vector / np.linalg.norm(track_vector) 
        
            # The car and the track shold point to the same "y" direction. 
            if np.sign(unit_car_vector[1]) != np.sign(unit_track_vector[1]):
                # Correct track vector if they are pointing to different "y" direction
                unit_track_vector *= -1 
            
            # Using polar cordinates to compute the angles
            track_angle = np.arctan2(unit_track_vector[1], unit_track_vector[0])
            car_angle = np.arctan2(unit_car_vector[1], unit_car_vector[0])
            angle = car_angle - track_angle
            
            # If the car is leaving the camera
            if marker_center[1] < image.shape[0]/8 and not self.reverse or \
               marker_center[1] > (image.shape[0] - image.shape[0]/8) and self.reverse:
                self.increment_tracklet_index()

        if self.debug:
            # Show centroids
            for c in tracklet.points:
                cv2.circle(image, (int(c[0]), int(c[1])),  2, (0, 0, 255), -1)

        
            # Draw car marker_center
            if marker_center is not None:
                cv2.line(image, tuple(marker_center), tuple(tracklet.points[i]), (255, 0, 0), 2)
                # cv2.line(image, tuple(tracklet.points[i+2]), tuple(tracklet.points[i-2]), (255, 0, 0), 5)
                
                # Show angles between the car and the track
                angles_img = np.zeros([500, 500, 3])
                track_vec = np.array(unit_track_vector*100 + 250, dtype=np.int32)
                car_vec = np.array(unit_car_vector*100 + 250, dtype=np.int32)
                cv2.line(angles_img, (250,250), tuple(track_vec), (0, 0, 255), 3)
                cv2.line(angles_img, (250,250), tuple(car_vec), (0, 255, 0), 3)
                cv2.imshow("Angles", angles_img)
            
            cv2.imshow('External System', image)
            cv2.waitKey(1)
            # cv2.imwrite(str(time.time()).replace('.','-')+'.png', image)
        
        return distance, angle        
    
    def normalize_angle(self, angle: float) -> float:
        """Normalize the angle to be between -π/2 and π/2.

        Args:
            angle (float): The angle to normalize.

        Returns:
            float: The normalized angle.
        """
        # Wrap the angle between 0 and 2*pi
        pi_2 = 2. * np.pi
        a = math.fmod(math.fmod(angle, pi_2) + pi_2, pi_2) 
        
        # Wrap the angle between -pi/2 and pi/2
        if a > np.pi/2:
            a -= np.pi

        return a 

    def compute_distante(self, car_point: np.ndarray, track_points: np.ndarray) -> tuple[float, int]:
        """
        Compute the minimum distance between the car's current position and the track points.

        Args:
            car_point (np.ndarray): The current position of the car as a 2D point (x, y).
            track_points (np.ndarray): The array of 2D points representing the track.

        Returns:
            tuple[float, int]: A tuple containing the minimum distance and the index of the closest point.
        """
        distance = cdist(np.array(track_points), np.array([car_point]))
        index = np.argmin(distance)
        min_distance = distance[index].squeeze()
        return min_distance, index

    def increment_tracklet_index(self) -> None:
        """
        Increment or decrement the tracklet index based on the current direction.

        This method updates the `current_tracklet_index` either by incrementing or decrementing it,
        depending on whether the system is in reverse mode or not.
        """
        if not self.reverse:
            if self.current_tracklet_index < len(self.tracklets)-1:
                self.current_tracklet_index += 1
            else:
                self.current_tracklet_index=0
        else:
            if self.current_tracklet_index > len(self.tracklets)*-1:
                self.current_tracklet_index -= 1
            else:
                self.current_tracklet_index=0
        
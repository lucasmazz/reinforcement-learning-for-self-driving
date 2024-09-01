import sys
import cv2
import numpy as np

from simulator import sim


class Camera:
    """Handles camera operations in the simulation."""

    def __init__(self, connection_id: int, name: str):
        """Connects to a camera in the simulation.

        Args:
            connection_id (int): Simulation's connection ID.
            name (str): Camera's sensor name.
        """
        self.connection_id = connection_id
        self.name = name

        # Init cam handler
        error_code, self.handler = sim.simxGetObjectHandle(
            self.connection_id, name, sim.simx_opmode_blocking)
        
        # Init sensors
        error_code, _, _ = sim.simxGetVisionSensorImage(
            self.connection_id, self.handler, 0, sim.simx_opmode_streaming)

    @property
    def image(self) -> np.ndarray:
        """Gets the current image from the camera.

        Returns:
            np.ndarray: The resized image array. Returns None if there is an error.
        """
        error_code, resolution, image = sim.simxGetVisionSensorImage(
            self.connection_id, self.handler, 0, sim.simx_opmode_buffer)

        # if failed to get image
        if error_code == 1:
            return None
        
        # Fix: remove pixels with negative value if any
        image = np.array(image, dtype=np.int32)
        image[image < 0] = 255
        
        image = np.array(image, dtype=np.uint8)
        image.resize([resolution[0], resolution[1], 3])
        
        return image
    
    @property
    def frame(self) -> np.ndarray:
        """Gets the current image frame with BGR to RGB conversion and flipping.

        Returns:
            np.ndarray: The processed image frame.
        """
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 0)
        return image 
        
    def _get_position(self, handler: int, parent: int = -1) -> tuple[int, np.ndarray]:
        """Gets the position of the object.

        Args:
            handler (int): Object handler.
            parent (int, optional): Parent object handler. Defaults to -1.

        Returns:
            tuple[int, np.ndarray]: Error code and position.
        """
        return sim.simxGetObjectPosition(
            self.connection_id, handler, parent, sim.simx_opmode_blocking)
    
    def _get_orientation(self, handler: int, parent: int = -1) -> tuple[int, np.ndarray]:
        """Gets the orientation of the object.

        Args:
            handler (int): Object handler.
            parent (int, optional): Parent object handler. Defaults to -1.

        Returns:
            tuple[int, np.ndarray]: Error code and orientation.
        """
        return sim.simxGetObjectOrientation(
            self.connection_id, handler, parent, sim.simx_opmode_blocking)
        
    @property
    def position(self) -> np.ndarray:
        """Gets the position of the camera.

        Returns:
            np.ndarray: Position of the camera. Returns None if there is an error.
        """
        error_code, pos = self._get_position(handler = self.handler)
    
        if (error_code == 0):
            return pos 
        
        return None
    
    @property
    def orientation(self) -> np.ndarray:
        """Gets the orientation of the camera.

        Returns:
            np.ndarray: Orientation of the camera. Returns None if there is an error.
        """
        error_code, pos = self._get_orientation(handler = self.handler)
    
        if (error_code == 0):
            return pos 
        
        return None
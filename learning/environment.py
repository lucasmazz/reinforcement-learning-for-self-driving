import math
from typing import Callable
import torch
import torchvision.transforms as T
import numpy as np
import random


class EnvironmentManager(object):
    
    def __init__(self, car, monitor, resize_x: int, resize_y: int, 
                 image_filter: Callable[[np.ndarray], np.ndarray],
                 distance_threshold: float = 30, angle_threshold: float = 45):
        """
        Initialize the Environment Manager.

        Args:
            car: The car object representing the vehicle in the environment.
            monitor: The monitor object responsible for tracking and running the environment.
            resize_x (int): The width to resize the frame to.
            resize_y (int): The height to resize the frame to.
            image_filter: A function or transformation to filter the image from the car's camera.
            distance_threshold (float, optional): The threshold for determining if the car is too far from the track. Defaults to 30.
            angle_threshold (float, optional): The threshold for the car's angle in degrees. Defaults to 45.
        """
        self.current_screen = None
        self.done = False
        self.car = car
        self.monitor = monitor      
        self.image_filter = image_filter
        self.distance_threshold = distance_threshold
        self.angle_threshold = math.radians(angle_threshold)
        self.resize = T.Compose([T.ToPILImage(),
                                 T.Resize(
                                    (resize_y, resize_x), 
                                    interpolation=T.InterpolationMode.BICUBIC
                                 ),
                                 T.ToTensor()])
        self.reverse = False
        
    @property
    def height(self) -> int:
        """Returns the height of the environment processed frame.

        Returns:
            int: height of the processed frame
        """
        return self.frame.shape[2]
    
    @property
    def width(self) -> int:
        """Returns the width of the environment processed frame.

        Returns:
            int: width of the processed frame
        """
        return self.frame.shape[3]
    
    @property
    def num_actions_available(self) -> int:
        """Returns the total number of available actions for the car.

        Returns:
            int: Number of available steering levels plus one.
        """
        return self.car.steering_level_range + 1
    
    @property
    def frame(self) -> torch.Tensor:
        """Get the processed frame from the environment screen.

        Returns:
            torch.Tensor: The processed frame as a tensor with shape (1, C, H, W).
        """
        image = self.car.camera.frame
        
        # Feature engineering
        image = self.image_filter(image)

        # Convert to tensor
        image = np.ascontiguousarray(image, dtype=np.float32) / 255
        image = torch.from_numpy(image)
        
        # Resize, and add a batch dimension (BCHW)
        return self.resize(image).unsqueeze(0)
    
    def reset(self) -> None:
        """
        Resets the environment and the car's position and orientation.

        This method resets the monitor and the car to their initial states, adjusting
        the camera and car positions and orientations. The car is set to reverse direction
        each time reset is called.
        """
        self.current_screen = None
        self.monitor.reset(reverse=self.reverse)
        
        camera_position = self.monitor.current_camera.position
        camera_orientation = self.monitor.current_camera.orientation
        car_initial_position = self.car.last_position
        car_initial_orientation = self.car.last_orientation
        
        position = camera_position.copy()
        position[2] = car_initial_position[2]
        
        orientation = car_initial_orientation.copy()
        orientation[1] = camera_orientation[2] + \
                         math.radians(180 + random.randint(-10, 10))
                         
        self.car.reset(position, orientation, reverse=self.reverse)
        self.reverse = not self.reverse
        
    def take_action(self, action: torch.Tensor) -> tuple[float, float, bool]:
        """
        Performs an action in the environment and checks the outcome.

        Args:
            action (torch.Tensor): The action tensor representing the car's steering level.

        Returns:
            tuple[float, float, bool]: The distance to the track, the car's angle, and whether the episode is done.
        """
        done = False
        
        # action.item is the value
        self.car.steering_level = action.item()
        
        # observes the action result
        distance, angle = self.monitor.run()
        
        if distance is None or angle is None:
            return None, None, None 
        
        if distance > self.distance_threshold or abs(angle) > self.angle_threshold:
            done = True
        
        return distance, angle, done
    
    def get_reward(self, dist: float, angle: float) -> torch.Tensor:
        """
        Calculate the reward based on the car's distance and angle.

        Args:
            dist (float): The car's distance from the track.
            angle (float): The car's angle relative to the track.

        Returns:
            torch.Tensor: The reward as a tensor.
        """
        # Limits the reward throught the distance threshold
        if abs(dist) > self.distance_threshold:
            r1 = 0
        else:
            r1 = (self.distance_threshold - abs(dist)) / self.distance_threshold  
        
        # Limits the angle throught the angle threshold
        if abs(angle) > self.angle_threshold:
            r2 = 0
        else:
            r2 = (self.angle_threshold - abs(angle)) / self.angle_threshold

        reward = torch.Tensor([(r1*0.8 + r2*0.2)**2])

        return reward
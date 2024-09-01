import sys
import math
from math import pi, sqrt, degrees, radians, tan, atan, sin, cos
from typing import Optional
from simulator import sim
from sensors import Camera


class Car:
    def __init__(self, connection_id: int, camera: Camera, car_id: str,
                 motor_left_id: str, motor_right_id: str, 
                 steering_left_id: str, steering_right_id:str,
                 d: float = 0.755, l: float = 2.5772, 
                 steering_level_range: int = 2, 
                 steering_angle_treshold: float = 45):
        """Handle the simulated car.

        Args:
            connection_id (int): Connection id.
            camera (object): Front car Camera object.
            car_id (str): Car name.
            motor_left_id (str): Left motor name.
            motor_right_id (str): Right motor name.
            steering_left_id (str): Left steering name.
            steering_right_id (str): Right steering name.
            d (float, optional): Distance between left and right wheels.
            l (float, optional): Distance between front and read wheels.
            steering_level_range (int, optional): Discrete total steering levels.
            steering_angle_treshold (int, optional): Maximum steering angle in degrees.   
        """
        self.connection_id = connection_id
        self.camera = camera
        self.d = d
        self.l = l
        self.steering_level_range = steering_level_range
        self.steering_angle_treshold = steering_angle_treshold

        self._speed_level = 0
        self._steering_angle = 0
        self._last_state = None
        
        error_code, self.car_handler = sim.simxGetObjectHandle(
            connection_id,
            car_id,
            sim.simx_opmode_blocking
        )
        
        assert(error_code == 0)
        
        # Left motor
        error_code, self.motor_left_handler = sim.simxGetObjectHandle(
            connection_id,
            motor_left_id,
            sim.simx_opmode_blocking
        )
        
        assert(error_code == 0)
        
        # Right motor
        error_code, self.motor_right_handler = sim.simxGetObjectHandle(
            connection_id,
            motor_right_id,
            sim.simx_opmode_blocking
        )

        assert(error_code == 0)

        # Left wheel
        error_code, self.steering_left_handler = sim.simxGetObjectHandle(
            connection_id,
            steering_left_id,
            sim.simx_opmode_blocking
        )

        assert(error_code == 0)

        # Right wheel
        error_code, self.steering_right_handler = sim.simxGetObjectHandle(
            connection_id,
            steering_right_id,
            sim.simx_opmode_blocking
        )

        assert(error_code == 0)
        
        # Just calling the API because it seems bugged on the first call
        self.position
        self.orientation
        
        # Start the dynamics
        sim.simxSetBooleanParameter(
            self.connection_id, 
            sim.sim_boolparam_dynamics_handling_enabled, 
            True, 
            sim.simx_opmode_oneshot
        )
        
    def _get_position(self, handler: int, parent: int = -1) -> tuple[int, list[float]]:
        """Get the position of the car in the simulation.

        Args:
            handler (int): The object handler.
            parent (int, optional): The parent object handler.

        Returns:
            tuple[int, list[float]]: Error code and the position coordinates.
        """
        return sim.simxGetObjectPosition(
            self.connection_id, handler, parent, sim.simx_opmode_blocking)

    def _set_position(self, handler: int, value: list[float], parent: int = -1) -> int:
        """Set the position of the car in the simulation.

        Args:
            handler (int): The object handler.
            value (list[float]): The position coordinates to set.
            parent (int, optional): The parent object handler.

        Returns:
            int: Error code.
        """
        return sim.simxSetObjectPosition(
            self.connection_id, handler, parent, value, sim.simx_opmode_blocking)
                
    def _get_orientation(self, handler: int, parent: int = -1) -> tuple[int, list[float]]:
        """Get the orientation of the car in the simulation.

        Args:
            handler (int): The object handler.
            parent (int, optional): The parent object handler.

        Returns:
            tuple[int, list[float]]: Error code and the orientation angles.
        """
        return sim.simxGetObjectOrientation(
            self.connection_id, handler, parent, sim.simx_opmode_blocking)
    
    def _set_orientation(self, handler: int, value: list[float], parent: int = -1) -> int:
        """Set the orientation of the car in the simulation.

        Args:
            handler (int): The object handler.
            value (list[float]): The orientation angles to set.
            parent (int, optional): The parent object handler.

        Returns:
            int: Error code.
        """
        return sim.simxSetObjectOrientation(
            self.connection_id, handler, parent, value, sim.simx_opmode_blocking)
        
    def _ackermann_steering(self, angle: float) -> tuple[float, float]:
        """Gives the steering angles of ackermann steering.

        Args:
            angle (float): Desired steering angle.

        Returns:
            tuple(float, float): Steering angles from the left and right wheel.   
        """
        steering_left = 0
        steering_right = 0

        if (angle > 0):
            steering_left = atan((2*self.l*sin(angle)) /
                                  (2*self.l*cos(angle) - self.d*sin(angle)))

            steering_right = atan((2*self.l*sin(angle)) /
                                 (2*self.l*cos(angle) + self.d*sin(angle)))
        elif (angle < 0):
            angle = angle*-1

            steering_left = -atan((2*self.l*sin(angle)) /
                                   (2*self.l*cos(angle) + self.d*sin(angle)))

            steering_right = -atan((2*self.l*sin(angle)) /
                                  (2*self.l*cos(angle) - self.d*sin(angle)))

        return (steering_left, steering_right)

    @property
    def steering_angle(self) -> float:
        """Get the current steering angle of the car.

        Returns:
            float: The current steering angle.
        """
        return self._steering_angle

    @steering_angle.setter
    def steering_angle(self, value: float) -> None:
        """Set the steering angle of the car.

        Args:
            value (float): The desired steering angle.
        """
        self._steering_angle = value

        steering_left, steering_right = self._ackermann_steering(value)

        sim.simxSetJointTargetPosition(self.connection_id, self.steering_left_handler,
                                       steering_left, sim.simx_opmode_streaming)

        sim.simxSetJointTargetPosition(self.connection_id, self.steering_right_handler,
                                       steering_right, sim.simx_opmode_streaming)

    @property
    def speed_level(self) -> int:
        """Get the current speed level of the car.

        Returns:
            int: The current speed level.
        """
        return self._speed_level

    @speed_level.setter
    def speed_level(self, value: int) -> None:
        """Set the speed level of the car.

        Args:
            value (int): The desired speed level.
        """
        self._speed_level = value

        sim.simxSetJointTargetVelocity(
            self.connection_id, self.motor_left_handler, self._speed_level, sim.simx_opmode_streaming)

        sim.simxSetJointTargetVelocity(
            self.connection_id, self.motor_right_handler, self._speed_level, sim.simx_opmode_streaming)
    
    @speed_level.setter
    def steering_level(self, level: int) -> None:
        """Set the steering level of the car.

        Args:
            level (int): The desired steering level.
        """
        a = self.steering_angle_treshold/(self.steering_level_range/2)
        self.steering_angle = radians(a*(level - self.steering_level_range/2))
        
    @property
    def position(self) -> Optional[list[float]]:
        """Get the current position of the car in the simulation.

        Returns:
            Optional[list[float]]: The current position coordinates, or None if an error occurred.
        """
        error_code, pos = self._get_position(handler=self.car_handler)
    
        if (error_code == 0):
            return pos 
        
        return None
    
    @position.setter
    def position(self, value: list[float]) -> bool:
        """Set the position of the car in the simulation.

        Args:
            value (list[float]): The desired position coordinates.

        Returns:
            bool: True if successful, False otherwise.
        """
        error_code = self._set_position(handler=self.car_handler, value=value)
        
        if (error_code == 0):
            return True
        
        return False
    
    @property
    def orientation(self) -> Optional[list[float]]:
        """Get the current orientation of the car in the simulation.

        Returns:
            Optional[list[float]]: The current orientation angles, or None if an error occurred.
        """
        error_code, pos = self._get_orientation(handler=self.car_handler)
    
        if (error_code == 0):
            return pos 
        
        return None
    
    @orientation.setter
    def orientation(self, value: list[float]) -> bool:
        """Set the orientation of the car in the simulation.

        Args:
            value (list[float]): The desired orientation angles.

        Returns:
            bool: True if successful, False otherwise.
        """
        error_code = self._set_orientation(handler=self.car_handler, value=value)
        
        if (error_code == 0):
            return True
        
        return False
    
    @property
    def last_state(self) -> Optional[list[list[float]]]:
        """Get the last saved state of the car.

        Returns:
            list: A list containing the last position and orientation of the car.
        """
        return self._last_state
    
    @property 
    def last_position(self) -> Optional[list[float]]:
        """Get the last saved position of the car.

        Returns:
            list[float]: The last saved position coordinates.
        """
        return self._last_state[0]
    
    @property
    def last_orientation(self) -> Optional[list[float]]:
        """Get the last saved orientation of the car.

        Returns:
            list[float]: The last saved orientation angles.
        """
        return self._last_state[1]
        
    def save_current_state(self) -> None:
        """Save the current position and orientation of the car.

        This method stores the current position and orientation of the car 
        in the `_last_state` attribute for future reference.
        """
        self._last_state = [self.position.copy(), self.orientation.copy()]
        
    def reset(self, position: list[float], orientation: list[float], reverse: bool = False) -> None:
        """Reset the car's position and orientation in the simulation.

        Args:
            position (list[float]): The desired position coordinates to reset to.
            orientation (list[float]): The desired orientation angles to reset to.
            reverse (bool, optional): If True, reverse the car's orientation by 180 degrees.
        """
        sim.simxSetBooleanParameter(
            self.connection_id, 
            sim.sim_boolparam_dynamics_handling_enabled, 
            False, 
            sim.simx_opmode_oneshot
        )

        self.position = position
        
        if reverse:
            orientation[1] += math.radians(180)
        
        self.orientation = orientation
        
        sim.simxSetBooleanParameter(
            self.connection_id, 
            sim.sim_boolparam_dynamics_handling_enabled, 
            True, 
            sim.simx_opmode_oneshot
        )
    
        self.speed_level = self._speed_level
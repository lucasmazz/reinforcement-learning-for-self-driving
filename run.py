# -*- coding: utf-8 -*-
import math
import sys
import time
import argparse
from math import degrees, radians
import uuid
import argparse
from collections import deque
from itertools import count

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from simulator import SimConnection
from sensors import Camera
from vehicle import Car
from vision.monitor import MonitorSystem
from vision.filters import red_line_segmentation
from learning.environment import EnvironmentManager
from learning.models.cnn import ConvNet 


def run(opt: argparse.Namespace, device: torch.device) -> None:
    """Main function to run the simulation.

    Args:
        opt (argparse.Namespace): Command-line options.
        device (torch.device): Device to run the model on (CPU/GPU).
    """
    conn = SimConnection()

    if conn.id == -1:
        sys.exit("Could not connect.")
        
    # Communication with the car's front camera
    car_camera = Camera(conn.id, name="CarCamera")

    # Communication with the simulation car
    car = Car(conn.id, 
              car_camera,
              car_id='Car',
              motor_left_id="nakedCar_motorLeft", 
              motor_right_id="nakedCar_motorRight",
              steering_left_id="nakedCar_steeringLeft", 
              steering_right_id="nakedCar_steeringRight",
              steering_level_range=opt.outputs-1) # 0 - 10 steering level range
    
    # Setup the policy net
    policy_net = ConvNet(opt.resize_y, opt.resize_x, 
                         inputs=opt.frames, outputs=opt.outputs).to(device)
    
    checkpoint = torch.load(opt.checkpoint, map_location=device)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    
    resize = T.Compose([
        T.ToPILImage(),
        T.Resize((opt.resize_y, opt.resize_x), 
                 interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    
    while True:        
        car.speed_level = opt.speed
        
        # Take steps in the environment.
        for t in count():
            im = car.camera.frame
            im = red_line_segmentation(im)

            if opt.debug:
                cv2.imshow('Frame', im)
                cv2.waitKey(1)
            
            # Convert the image to torch
            im = np.ascontiguousarray(im, dtype=np.float32) / 255
            im = torch.from_numpy(im)
        
            # Resize, and add a batch dimension (BCHW)
            im = resize(im).unsqueeze(0)
                
            with torch.no_grad():
                # Select the action with the larger expected reward
                screens = deque([im] * opt.frames, opt.frames)
                state = torch.cat(list(screens), dim=1)
                action = policy_net(state.to(device)).max(1)[1].view(1, 1).to('cpu')
                car.steering_level = action.item()
                
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--debug', dest='debug', action='store_true')
    
    parser.add_argument('--speed', type=float, default=10, 
                        help='car speed level')
    
    parser.add_argument('--outputs', type=float, default=11, 
                        help='number of network outputs')
    
    parser.add_argument('--frames', type=int, default=1, 
                        help='number of the last frames that represent the state')
    
    parser.add_argument('--resize_x', type=int, default=128, 
                        help='image downsampling')
    
    parser.add_argument('--resize_y', type=int, default=96, 
                        help='image downsampling')
    
    parser.add_argument('--checkpoint', type=str, default='', 
                        help='checkpoint filename for load training')
    
    parser.add_argument('--gpu', action='store_true', help='enable gpu for training')
    
    opt = parser.parse_args()
    
    if opt.gpu and torch.cuda.is_available():
         device = torch.device("cuda")
    else:
         device = torch.device("cpu")

    run(opt, device)
 # -*- coding: utf-8 -*-
import os
import math
import sys
import time
import random
import datetime
import argparse
from datetime import datetime
from collections import deque
from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from simulator import SimConnection
from sensors import Camera
from vehicle import Car
from vision import *

from vision.monitor import MonitorSystem

from learning.environment import EnvironmentManager
from learning.agent import Agent
from learning.memory import ReplayMemory, Experience
from learning.learning import EpsilonGreedyStrategy, QValues
from learning.models.cnn import ConvNet


def plot(writer, episode, steps, episode_durations, loss) -> None:
    """
    Logs training metrics to TensorBoard and prints training statistics.

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter object for logging.
        episode (int): The current episode number.
        steps (int): The steps achieved in the current episode.
        episode_durations (list[int]): A list containing the duration of each episode.
        loss (float): The loss value from the model's optimization process.
    
    Returns:
        None
    """
    loss_t = torch.tensor(loss, dtype=torch.float)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    episode_number = len(durations_t)
    
    # Logging to TensorBoard
    writer.add_scalar('Loss', loss_t, episode)
    writer.add_scalar('Steps', steps, episode)

    # Compute the moving average
    window_size = 100
    
    if episode_number < window_size:
        # Compute moving average over all episodes if less than 100
        moving_avg = durations_t.mean().item()
    else:
        # Compute moving average over the last 100 episodes
        moving_avg = durations_t[-window_size:].mean().item()

    # Log moving average to TensorBoard
    writer.add_scalar('Moving Average', moving_avg, episode)
    
    # Print training statistics
    print(str(datetime.now()), 
          '| Episode: ', episode_number, 
          '| Steps: ', steps,
          '| Moving Average: ', round(moving_avg, 4), 
          '| Loss: ', str(loss))
        
        
def optimize_model(opt: argparse.Namespace, policy_net: nn.Module, 
                   target_net: nn.Module, optimizer: optim, criterion: nn.Module,
                   memory: ReplayMemory, device: torch.device) -> float:
    """
    Performs one step of optimization for the model, updating the policy network's parameters
    based on experiences sampled from replay memory.

    Args:
        opt (argparse.Namespace): Parsed command-line arguments containing hyperparameters.
        policy_net (nn.Module): The policy network (Q-network) being trained.
        target_net (nn.Module): The target network used to calculate target Q-values.
        optimizer (torch.optim.Optimizer): The optimizer used to adjust the policy network's weights.
        memory (ReplayMemory): The replay memory containing past experiences.
        device (torch.device): The device (CPU or GPU) on which the computation is performed.
    
    Returns:
        float | None: The loss value from the optimization step, or None if there weren't enough experiences to sample.
    """
    
    if len(memory) < opt.batch:
        return float(np.inf)
    
    # Gets samples of experience stored in the memory replay.
    experiences = memory.sample(opt.batch)
    
    # Transposes the batch of Experiences to Experiences of batches.
    batch = Experience(*zip(*experiences))
    
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_state_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), 
        device=device, 
        dtype=torch.bool
    )
    
    states = torch.cat(batch.state).to(device)
    actions = torch.cat(batch.action).to(device)
    rewards = torch.cat(batch.reward).type(torch.FloatTensor).to(device)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to(device)
    
    q_values = QValues(policy_net, target_net, device)
   
    # Get the current, next and target Q Values for given state action pair.
    current_q_values = q_values.get_current(states, actions).to(device)
    
    next_q_values = q_values.get_next(non_final_state_mask, 
                                      non_final_next_states, 
                                      opt.batch).to(device)
    
    # Computes the expected Q values
    target_q_values = (next_q_values * opt.gamma) + rewards

    # Computes the loss
    loss = criterion(current_q_values, target_q_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
        
    optimizer.step()

    return loss.item()


def train(opt: argparse.Namespace, device: torch.device) -> None: 
    """
    Trains a reinforcement learning agent to control a simulated car using a DQN-based approach.
    The function initializes the environment, networks, and other training components, and runs
    the training loop over a specified number of episodes.

    Args:
        opt (argparse.Namespace): Parsed command-line arguments containing hyperparameters and options.
        device (torch.device): The device (CPU or GPU) on which the training will be performed.
    
    Returns:
        None
    """

    conn = SimConnection()

    if conn.id == -1:
        sys.exit("Could not connect.")
        
    # Communication with the external cameras
    external_cameras = [
        Camera(conn.id, name="ExtCamera00"),
        Camera(conn.id, name="ExtCamera01"),
        Camera(conn.id, name="ExtCamera02"),
        Camera(conn.id, name="ExtCamera03"),
        Camera(conn.id, name="ExtCamera04"),
        Camera(conn.id, name="ExtCamera05"),
        Camera(conn.id, name="ExtCamera06"),
        Camera(conn.id, name="ExtCamera07"),
    ]

    # Communication with the vehicles's front camera
    car_camera = Camera(conn.id, name="CarCamera")

    # Communication with the simulated vehicle
    car = Car(conn.id, 
              car_camera,
              car_id = 'Car',
              motor_left_id = "nakedCar_motorLeft", 
              motor_right_id = "nakedCar_motorRight",
              steering_left_id = "nakedCar_steeringLeft", 
              steering_right_id = "nakedCar_steeringRight",
              steering_level_range = opt.outputs - 1)

    # Initialize the monitor_system
    monitor_system = MonitorSystem(external_cameras, opt.debug)
    monitor_system.load()
    car.save_current_state() 
    
    # Setup the environment and the replay memory
    env = EnvironmentManager(car, monitor_system, opt.resize_x, 
                             opt.resize_y, red_line_segmentation)
    
    memory = ReplayMemory(opt.memory_size)
    
    # Setup the policy net and the target net
    policy_net = ConvNet(env.height, env.width, inputs=opt.frames,  
                         outputs=env.num_actions_available).to(device)
    
    target_net = ConvNet(env.height, env.width, inputs=opt.frames, 
                         outputs=env.num_actions_available).to(device)
    
    # Setup the optimizer
    optimizer = optim.Adam(params=policy_net.parameters(), lr=opt.lr)
    
    # Define the loss function
    criterion = nn.SmoothL1Loss()
    
    current_step = 0
    loss = float(np.inf)
    start_episode = 0
    mean_last = deque([0] * opt.n_last_episodes, opt.n_last_episodes)
    episode_durations = []
    stop_training = False
    max_score = opt.start_score
    current_step = 0
    last_time = datetime.now()
    
    # Restart training from a checkpoint
    if opt.checkpoint:
        checkpoint = torch.load(opt.checkpoint)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode =  checkpoint['episode']
        loss = checkpoint['loss']
        episode_durations = list(checkpoint['episode_durations'])
        current_step = checkpoint['current_step']
        max_score = checkpoint['max_score']
    
    # Setup strategy
    strategy = EpsilonGreedyStrategy(opt.epsilon_start, 
                                     opt.epsilon_end, 
                                     opt.epsilon_decay)
    # Setup agent
    agent = Agent(strategy, env.num_actions_available, current_step)

    # Load memory saved earlier
    if opt.memory:
        memory.load(opt.memory)
    
    # Load target net weights and biases from policy net
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    writer = SummaryWriter(log_dir='runs')
       
    # Training Loop
    for episode in range(start_episode, start_episode + opt.n_episodes):
        last_time = datetime.now()
        
        # Start the environment and initial state
        env.reset()
        time.sleep(1)
        screens = deque([env.frame] * opt.frames, opt.frames)
        state = torch.cat(list(screens), dim=1)

        # Set car speed with some augmentation
        car.speed_level = random.uniform(opt.speed - 1, opt.speed + 1) 
        
        # Step in the environment
        for t in count():
            if opt.debug:
                im = env.frame.cpu().numpy().squeeze()
                cv2.imshow('Frame', im)
                cv2.waitKey(1)
                
            # Chose and take an action on the environment
            action = agent.select_action(state.to(device), 
                                         policy_net, stop_training)
            
            dist, angle, done = env.take_action(action)
            
            current_time = datetime.now()
            delta = current_time - last_time
            
            # Timeout if the car wasn't being tracked for more than 5 seconds
            if (delta.total_seconds() > 5):
                break

            # If the vehicle tracker has failed
            if dist is None or angle is None:
                continue
    
            last_time = current_time
            
            # Observe new state
            screens.append(env.frame)
            next_state = torch.cat(list(screens), dim=1) if not done else None

            # Reward computation
            reward = env.get_reward(dist, angle)
            
            print("Action:", action.item(),
                  "| Steering (°):", '{:.4f}'.format(np.degrees(car.steering_angle)), 
                  "| Speed:", '{:.4f}'.format(car.speed_level), 
                  "| Distance (px):", '{:.4f}'.format(dist), 
                  "| Angle (°):", '{:.4f}'.format(np.degrees(angle)),
                  "| Reward:", '{:.4f}'.format((reward.item())))
                        
            if (t >= opt.max_step-1):
                done = True     
                    
            # Store the experience in the memory replay
            memory.push(state, action, next_state, reward)
            
            # Change the current state to the next state
            state = next_state
            
            # Model optimization
            if done:
                car.speed_level = 0
                episode_durations.append(t + 1)
                mean_last.append(t + 1)
                mean = 0
                
                for i in range(opt.n_last_episodes):
                    mean = mean_last[i] + mean
                
                mean = mean / opt.n_last_episodes
                
                if mean < opt.training_stop and stop_training == False:
                    loss = optimize_model(opt, policy_net, target_net, 
                                          optimizer, criterion, memory, device)
                else:
                    stop_training = True
                    
                plot(writer, episode, t + 1, episode_durations, loss)
                break
            
        # Updates the Target Network weights and biases
        if episode != start_episode and episode % opt.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Saves checkpoint and memory data
        if opt.save and episode != start_episode and (episode % opt.save_frequency == 0):

            # Saves checkpoint
            torch.save({
                'episode': episode,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'episode_durations': episode_durations,
                'current_step': agent.current_step,
                'max_score': max_score,
            }, os.path.join(opt.save, "checkpoint_" + str(episode) + ".pt"))
            
            # Saves memory replay
            memory.save(os.path.join(opt.save, "memory_replay_" + str(episode) + '.data'))
    
    # After training is complete, flush the writer
    writer.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--debug', dest='debug', action='store_true')
    
    parser.add_argument('--speed', type=float, default=10, 
                        help='car speed level')
    
    parser.add_argument('--outputs', type=float, default=11, 
                        help='number of network outputs')
    
    parser.add_argument('--batch', type=int, default=128, 
                        help='model batch size')
    
    parser.add_argument('--gamma', type=float, default=0.999, 
                        help='gamma value used to compute the expected Q values')

    parser.add_argument('--epsilon_start', type=float, default=0.9, 
                        help='greater value of epsilon used in the greedy strategy')
    
    parser.add_argument('--epsilon_end', type=float, default=0.05, 
                        help='lower value of epsilon used in the greedy strategy')
    
    parser.add_argument('--epsilon_decay', type=float, default=2500, 
                        help='decay rate of epsilon over time')
    
    parser.add_argument('--target_update', type=int, default=50, 
                        help='how many episodes will update the target network weights')
    
    parser.add_argument('--memory_size', type=int, default=100000, 
                        help='capacity of replay memory')
                        
    parser.add_argument('--start_score', type=int, default=50, 
                        help='inital score to end the environment game')
    
    parser.add_argument('--end_score', type=int, default=100000, 
                        help='ending score to end the environment game')
    
    parser.add_argument('--score_rate', type=int, default=50, 
                        help='')

    parser.add_argument('--max_step', type=int, default=1000, 
                        help='max step to reset the env')

    parser.add_argument('--training_stop', type=int, default=142, 
                        help='mean last threshold for stop training')
    
    parser.add_argument('--n_episodes', type=int, default=50000, 
                        help='total episodes to be run')
    
    parser.add_argument('--n_last_episodes', type=int, default=50,
                        help='number of episodes with steps above the treshold for stopping training')
    
    parser.add_argument('--frames', type=int, default=1, 
                        help='number of the last frames that represent the state')
    
    parser.add_argument('--resize_x', type=int, default=128, 
                        help='image downsampling')
    
    parser.add_argument('--resize_y', type=int, default=96, 
                        help='image downsampling')

    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='learning rate for training')

    parser.add_argument('--checkpoint', type=str, default='', 
                        help='checkpoint filename for load training')

    parser.add_argument('--memory', type=str, default='', 
                        help='memory filename for load training')
    
    parser.add_argument('--save', type=str, default='', 
                        help='save checkpoints and memory data')
    
    parser.add_argument('--save_frequency', type=int, default=50, 
                        help='number of episodes for save checkpoints and memory data')
        
    parser.add_argument('--gpu', action='store_true', help='enable gpu for training')
    
    opt = parser.parse_args()
    
    # Setup device (GPU x CPU).
    if opt.gpu and torch.cuda.is_available():
         device = torch.device("cuda")
    else:
         device = torch.device("cpu")
    
    train(opt, device)

import io
import os 
import random
import pickle 
import gzip
from collections import namedtuple


""" 
Experience: used to create experiences that will be stored and sampled for replay memory object.
"""
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory(object):
    
    def __init__(self, capacity: int) -> None:
        """Initializes the ReplayMemory to store Experience objects.

        Args:
            capacity (int): The maximum number of experiences to store in memory.
        """
        self.capacity = capacity
        self.data = []
        self.index = 0

    def push(self, *args) -> None:
        """Stores a new experience in the memory. If the memory is full, it overwrites the oldest experience.

        Args:
            state (torch.Tensor): The last state used in Q-learning.
            action (torch.Tensor): The last action taken.
            next_state (torch.Tensor): The next state resulting from the action.
            reward (torch.Tensor): The reward received from the environment.
        """
        experience = Experience(*args) 
        
        if len(self.data) < self.capacity:
            self.data.append(experience)
        else:
            self.data[self.index % self.capacity] = experience
        
        self.index += 1

    def sample(self, batch_size: int) -> list[Experience]:
        """Returns a random sample of experiences from the memory.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            List[Experience]: A list of randomly sampled experiences.
        """
        return random.sample(self.data, batch_size) 

    def load(self, filename: str) -> bool:
        """Loads the stored experiences from a file.

        Args:
            filename (str): The path to the file from which to load the data.

        Returns:
            bool: True if the data was successfully loaded.
        """
        with gzip.open(filename, 'rb') as file:
            self.data = pickle.load(file)

        return True

    def save(self, filename: str) -> bool:
        """Saves the current experiences to a file.

        Args:
            filename (str): The path to the file in which to save the data.

        Returns:
            bool: True if the data was successfully saved.
        """
        with gzip.open(filename, 'wb') as file:
            pickle.dump(self.data, file)

        return True
    
    def __len__(self) -> int:
        """Returns the current number of experiences stored in memory.

        Returns:
            int: The number of experiences in the memory.
        """
        return len(self.data)
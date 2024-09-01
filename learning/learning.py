import math
import torch
import torch.nn as nn


class EpsilonGreedyStrategy(object):
    """Handles the epsilon-greedy strategy for exploration in reinforcement learning."""
        
    def __init__(self, start: float, end: float, decay: float):
        """Initializes the epsilon-greedy strategy.

        Args:
            start (float): Initial higher exploration rate.
            end (float): Final lower exploration rate.
            decay (float): Rate at which exploration decreases.
        """
        self.start = start
        self.end = end
        self.decay = decay
    
    def get_exploration_rate(self, current_step: int) -> float:
        """Calculates the exploration rate for the given step.

        Args:
            current_step (int): Current step number.

        Returns:
            float: Exploration rate at the given step.
        """
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step / self.decay)
            

class QValues(object):
    """Handles and computes the Q-values using policy and target networks."""
    
    def __init__(self, policy_net: nn.Module, target_net: nn.Module, device: torch.device):
        """Initializes the QValues class.

        Args:
            policy_net (nn.Module): Policy network model.
            target_net (nn.Module): Target network model.
            device (torch.device): Device to perform calculations (CPU/GPU).
        """
        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device
        
    def get_current(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Computes the Q-values for the given states and actions using the policy network.

        Args:
            states (torch.Tensor): Tensor of sample states.
            actions (torch.Tensor): Tensor of sample actions.

        Returns:
            torch.Tensor: Q-values computed by the policy network.
        """
        return self.policy_net(states).gather(1, actions)
    
    def get_next(self, non_final_state_locations: torch.Tensor,
                 non_final_next_states: torch.Tensor,
                 batch_size: int) -> torch.Tensor:
        """Computes the expected values of the next actions based on the target network.

        Args:
            non_final_state_locations (torch.Tensor): Tensor with non-final state indices.
            non_final_next_states (torch.Tensor): Tensor with non-final next states.
            batch_size (int): Size of the batch.

        Returns:
            torch.Tensor: Expected Q-values for the next states.
        """
        values = torch.zeros(batch_size).to(self.device)
        
        values[non_final_state_locations] = \
            self.target_net(non_final_next_states).max(dim=1)[0].detach()
        
        return values
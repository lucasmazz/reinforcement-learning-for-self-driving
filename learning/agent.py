import random
import torch
import torch.nn as nn
from learning.learning import EpsilonGreedyStrategy
            
            
class Agent(object):
    """Handles action selection for the agent using an epsilon-greedy strategy."""
    
    def __init__(self, strategy: EpsilonGreedyStrategy, num_actions: int, current_step: int):
        """Initializes the agent.

        Args:
            strategy (EpsilonGreedyStrategy): Object that handles the epsilon-greedy strategy.
            num_actions (int): Number of possible actions the agent can take.
            current_step (int): The current step in the training process.
        """
        self.current_step = current_step
        self.strategy = strategy
        self.num_actions = num_actions
    
    def select_action(self, state: torch.Tensor, policy_net: nn.Module, 
                      inference: bool = False) -> torch.Tensor:
        """Selects an action based on exploration or exploitation.

        Args:
            state (torch.Tensor): The current state.
            policy_net (nn.Module): The policy network (DQN) used to learn the optimal policy.
            inference (bool): If True, the model will not choose random actions.

        Returns:
            torch.Tensor: Chosen action, with shape (1, 1) and dtype long.
        """
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        
        if random.random() > rate or inference:
            with torch.no_grad():
                # Select the action with the larger expected reward
                return policy_net(state).max(1)[1].view(1, 1).to('cpu')
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device='cpu', dtype=torch.long)

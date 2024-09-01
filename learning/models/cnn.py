import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """Simple Convolutional Neural Network.
    
    This network consists of convolutional layers followed by fully connected layers.
    It's designed to process images with a specified height and width.
    """

    def __init__(self, h: int, w: int, inputs: int = 1, outputs: int = 2):
        """Initializes the ConvNet model with specified input and output dimensions.

        Args:
            h (int): The height of the input images.
            w (int): The width of the input images.
            inputs (int, optional): The number of input channels (e.g., 1 for grayscale, 3 for RGB). Default is 1.
            outputs (int, optional): The number of output classes or features. Default is 2.
        """
        super(ConvNet, self).__init__()

        self.convolutional_layers = nn.Sequential(
            # First layer
            nn.Conv2d(inputs, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second layer
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=9744, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=outputs),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the ConvNet.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        return self.linear_layers(x)
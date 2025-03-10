import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt

class CNNBackbone(nn.Module):
    """
    Configurable CNN backbone for image-based observations.
    Parameters for channel progression, kernel sizes, and strides
    """
    def __init__(
        self, 
        obs_shape=(3, 84, 84), 
        output_dim=6, 
        channels=[32, 64, 64],  
        kernel_sizes=[8, 4, 3],  
        strides=[4, 2, 1],
        hidden_dims=[512],  
    ):
        super().__init__()
        self.in_channels = obs_shape[0]
        
        # Validation to ensure parameter lists match in length
        assert len(channels) == len(kernel_sizes) == len(strides), (
            "channels, kernel_sizes, and strides must have the same length"
        )
        
        # Build conv layers dynamically based on parameters
        layers = []
        in_channels = self.in_channels
        
        for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides):
            layers.extend([
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    stride=stride
                ),
                nn.ReLU(),
                # nn.BatchNorm2d(out_channels)
            ])
            in_channels = out_channels
            
        layers.append(nn.Flatten())
        self.features = nn.Sequential(*layers)

        if hidden_dims is not None and len(hidden_dims) != 0:
            # Compute feature output size with dummy forward pass
            with torch.no_grad():
                self._dummy_input = torch.zeros(1, *obs_shape)
                self._dummy_output = self.features(self._dummy_input)
                print(f"CNN feature output shape: {self._dummy_output.shape}")

            # Build fully connected layers
            self.hidden_layers = nn.ModuleList()
            input_dim = self._dummy_output.shape[1]
            for h_dim in hidden_dims:
                self.hidden_layers.append(nn.Linear(input_dim, h_dim))
                self.hidden_layers.append(nn.ReLU())
                input_dim = h_dim
            self.hidden_layers.append(nn.Linear(input_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if hasattr(self, "hidden_layers"):
            for layer in self.hidden_layers:
                x = layer(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip_connection(x)
        out = self.relu(out)
        return out

class ResNetCNNExtractor(nn.Module):
    def __init__(self, env, features_dim=512, depth=2):
        super(ResNetCNNExtractor, self).__init__()

        self.state_space = env.observation_space.shape
        print(self.state_space)
        self.action_space = env.action_space.n

        # if the channel dim is not permuted, the input shape is (batch_size, height, width, channels) so use -1 instead
        n_input_channels = self.state_space[0]
        self.features_dim = features_dim
        self.depth = depth

        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        self.conv_layers = nn.ModuleList()
        channels = self.conv1.out_channels
        for i in range(self.depth):
            self.conv_layers.append(self._make_layer(channels, channels * 2, stride=2))
            channels *= 2
        
        self.flatten = nn.Flatten()
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            print(self.flatten(self._forward_conv(torch.as_tensor(env.observation_space.sample()[None]).float())).shape)
            n_flatten = self.flatten(self._forward_conv(torch.as_tensor(env.observation_space.sample()[None]).float())).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, self.features_dim),
            nn.ReLU()
        )

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels)
        )

    def _forward_conv(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def forward(self, observations):
        conv_out = self._forward_conv(observations)
        return self.linear(self.flatten(conv_out))
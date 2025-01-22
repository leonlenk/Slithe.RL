import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy

import matplotlib.pyplot as plt

# https://github.com/tims457/RL_Agent_Notebooks/blob/master/Policy%20Gradient%20with%20Cartpole%20and%20PyTorch.ipynb
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

# Policy for Policy Gradients

class ResNetCNNExtractor(nn.Module):
    def __init__(self, env, features_dim=512):
        super(ResNetCNNExtractor, self).__init__()

        self.state_space = env.observation_space.shape
        print(self.state_space)
        self.action_space = env.action_space.n

        # if the channel dim is not permuted, the input shape is (batch_size, height, width, channels) so use -1 instead
        n_input_channels = self.state_space[0]
        features_dim = 512

        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        self.layer1 = self._make_layer(32, 64, stride=2)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 128, stride=2)
        self.layer4 = self._make_layer(128, 128, stride=2)
        
        self.flatten = nn.Flatten()
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            print(self.flatten(self._forward_conv(torch.as_tensor(env.observation_space.sample()[None]).float())).shape)
            n_flatten = self.flatten(self._forward_conv(torch.as_tensor(env.observation_space.sample()[None]).float())).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels)
        )

    def _forward_conv(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        return x

    def forward(self, observations):
        conv_out = self._forward_conv(observations)
        return self.linear(self.flatten(conv_out))

class Policy(nn.Module):
    def __init__(self, env):
        super(Policy, self).__init__()

        self.state_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.feature_extractor = ResNetCNNExtractor(env)
        self.policy = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space)
        )
        
        # Episode policy and reward history 
        self.policy_history = torch.Tensor()
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.gamma = 0.99

    def forward(self, x):    
        logits = self.policy(x)
        return logits 

    def select_action(self, state):
        feat = self.feature_extractor(state)
        logits = self.policy(feat)
        probs = torch.softmax(logits, dim=-1)
        # print("Logits:", logits)
        # print("Probs:", probs)

        c = Categorical(probs=probs)
        action = c.sample()
        # print("Action:", action)

        # Add log probability of our chosen action to our history    
        if self.policy_history.dim() != 0:
            self.policy_history = torch.cat([self.policy_history, c.log_prob(action)])
        else:
            self.policy_history = (c.log_prob(action))

        return action

    def update_policy(self, optimizer):
        R = 0
        rewards = []
        
        # Discount future rewards back to the present using gamma
        for r in self.reward_episode[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        print(f"{rewards.sum()=}")
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        
        # Calculate loss
        # loss = $- \sum_i log(\pi(a_i|s_i)) * R_i$
        policy_loss = (torch.sum(torch.mul(self.policy_history, rewards).mul(-1)))

        # Calculate entropy loss
        # Regularization term to encourage exploration (higher entropy when probabilities are closer)
        probs = torch.exp(self.policy_history)  # Convert log_probs back to probs
        entropy_loss = -torch.sum(probs * torch.log(probs + 1e-10))  # Adding a small value to avoid log(0)
        
        loss = policy_loss + 0.005 * entropy_loss

        print(f"{loss=}")
        
        # Update network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save and initialize episode history counters
        self.loss_history.append(loss.data.item())
        self.reward_history.append(np.sum(self.reward_episode))
        self.policy_history = torch.Tensor()
        self.reward_episode= []

    def run_episode(self, env, episodes):
        optimizer = optim.Adam(self.parameters(), lr=1e-2)

        for episode in range(episodes):
            obs, info = env.reset()
            done = False
            steps = 0
            while not done:
                steps += 1
                obs = torch.from_numpy(obs).float().unsqueeze(0)
                action = self.select_action(obs)
            
                obs, reward, terminated, truncated, info = env.step(action)
                self.reward_episode.append(reward)
                done = terminated or truncated

            
            print(f"Episode: {episode}, Steps: {steps}")
            self.plot_reward_episode()
            # Compute the loss and update the policy
            self.update_policy(optimizer)

    def plot_reward_episode(self):
        plt.plot(self.reward_episode)
        plt.title('Reward over time')
        plt.ylabel('Reward')
        plt.xlabel('Steps')
        plt.show
        

# Stable Baselines3 custom CNN policy with residual block (Ignore below)
class ResNetCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(ResNetCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        self.layer1 = self._make_layer(32, 64, stride=2)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 128, stride=2)
        self.layer4 = self._make_layer(128, 128, stride=2)
        
        self.flatten = nn.Flatten()
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.flatten(self._forward_conv(torch.as_tensor(observation_space.sample()[None]).float())).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels)
        )

    def _forward_conv(self, x):
        # make channel first dim
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, observations):
        conv_out = self._forward_conv(observations)
        return self.linear(self.flatten(conv_out))

class CustomResNetPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomResNetPolicy, self).__init__(*args, **kwargs, features_extractor_class=ResNetCNN, features_extractor_kwargs=dict(features_dim=512))

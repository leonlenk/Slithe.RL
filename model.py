import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy

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
        super(ResNetCNNPolicy, self).__init__()

        self.state_space = env.observation_space.shape
        self.action_space = env.action_space.n

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
        self.policy_history = Variable(torch.Tensor()) 
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):    
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)

    def select_action(self, state):
        feat = self.feature_extractor(state)
        probs = self.policy(feat)

        c = Categorical(probs)
        action = c.sample()

    def update_policy(self):
        R = 0
        rewards = []
        
        # Discount future rewards back to the present using gamma
        for r in self.reward_episode[::-1]:
            R = r + self.gamma * R
            rewards.insert(0,R)
            
            # Scale rewards
            rewards = torch.FloatTensor(rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
            
            # Calculate loss
            loss = (torch.sum(torch.mul(self.policy_history, Variable(rewards)).mul(-1), -1))
            
            # Update network weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #Save and intialize episode history counters
            self.loss_history.append(loss.data[0])
            self.reward_history.append(np.sum(self.reward_episode))
            self.policy_history = Variable(torch.Tensor())
            self.reward_episode = []

    def run_episode(self, env, episodes):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                state, reward, done, _ = env.step(action)
                self.reward_episode.append(reward)
            # Compute the loss and update the policy
            self.update_policy()

# Stable Baselines3 custom CNN policy with residual blocks

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

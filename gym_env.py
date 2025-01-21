import gymnasium as gym

# Define a wrapper for Gym env to add survival rewards
class RewardWrapper(gym.Wrapper):
    def __init__(self, env, survival_reward=0.01):
        super(RewardWrapper, self).__init__(env)
        self.survival_reward = survival_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Add a small positive reward for surviving
        reward += self.survival_reward
        return obs, reward, terminated, truncated, info
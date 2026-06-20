import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import gymnasium as gym
from itertools import count
from collections import deque

env = gym.make('CartPole-v1', render_mode='human')
env.reset(seed=42)
torch.manual_seed(42)

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
        self.drop = nn.Dropout(p=0.6)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = F.relu(x)
        action_scores = self.fc2(x)
        return F.softmax(action_scores, dim=-1)

policy = Policy(state_size=4, action_size=2)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode():
    R = 0
    gamma = 0.99
    policy_loss = []
    returns = deque() 
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def main():
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):
            action = select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward
            if terminated or truncated:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()

        if i_episode % 10 == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')

if __name__ == '__main__':
    main()
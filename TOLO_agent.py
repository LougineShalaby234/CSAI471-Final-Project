import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import cv2
import os
import time
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import json
from Tolo_Net import TOLO_Net
# ----------------------------
# 4. Enhanced Agent
# ----------------------------
class Tolo_gent:
    def __init__(self, state_dim, action_dim, weather_simulator):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.weather_sim = weather_simulator

        self.policy_net = TOLO_Net(action_dim).to(self.device)
        self.target_net = TOLO_Net(action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                    lr=0.01, weight_decay=1e-5)
        self.memory = deque(maxlen=5000)
        self.batch_size = 16
        self.gamma = 0.99
        self.tau = 0.005  # For soft updates
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01

        self._init_tracking()
        self.reward_components = {
            'reward_base': [],
            'reward_friction': [],
            'reward_fuel': [],
            'reward_speed': [],
            'reward_weather': [],
            'reward_acceleration': []
        }

    def _init_tracking(self):
        self.loss_history = []
        self.weight_stats = {
            'means': [],
            'stds': [],
            'grad_means': []
        }
        self.weather_history = defaultdict(list)

    def _encode_weather(self, weather_str):
        return torch.tensor([self.weather_sim.weather_conditions.index(weather_str)]).long().to(self.device)

    def select_action(self, state, weather):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0).to(self.device)
            weather_t = self._encode_weather(weather)
            q_values = self.policy_net(state_t, weather_t)
            return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        # Sample and process batch
        batch = random.sample(self.memory, self.batch_size)
        states, weathers, actions, rewards, next_states, next_weathers, dones, reward_infos = zip(*batch)

        # Convert to tensors
        states = torch.stack([torch.FloatTensor(s).permute(2,0,1) for s in states]).to(self.device)
        weathers = torch.stack([self._encode_weather(w) for w in weathers]).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack([torch.FloatTensor(ns).permute(2,0,1) for ns in next_states]).to(self.device)
        next_weathers = torch.stack([self._encode_weather(nw) for nw in next_weathers]).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        for reward_info in reward_infos:
            for comp, value in reward_info.items():
                self.reward_components[comp].append(value)
        # Q-value calculation
        current_q = self.policy_net(states, weathers).gather(1, actions.unsqueeze(1))

        # Double DQN with target network
        with torch.no_grad():
            next_actions = self.policy_net(next_states, next_weathers).argmax(1)
            next_q = self.target_net(next_states, next_weathers).gather(1, next_actions.unsqueeze(1))
            target_q = rewards + (1 - dones) * self.gamma * next_q.squeeze()

        # Loss calculation with Huber loss
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 2.0)
        self.optimizer.step()

        # Soft target update
        for target_param, policy_param in zip(self.target_net.parameters(),
                                             self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

        # Update tracking
        self.loss_history.append(loss.item())
        self._track_weights()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def _track_weights(self):
        """Track weight statistics for visualization"""
        means, stds, grad_means = [], [], []
        for name, param in self.policy_net.named_parameters():
            if 'weight' in name:
                means.append(param.data.mean().item())
                stds.append(param.data.std().item())
                if param.grad is not None:
                    grad_means.append(param.grad.data.mean().item())

        self.weight_stats['means'].append(np.mean(means))
        self.weight_stats['stds'].append(np.mean(stds))
        self.weight_stats['grad_means'].append(np.mean(grad_means))




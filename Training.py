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
from TOLO_agent import Tolo_agent
from Weather_simulator import WeatherSimulator
from CarRacingEnv import CarRacingEnv
# ----------------------------
#  Visualization & Reporting
# ----------------------------
def plot_training_progress(agent, rewards, weather_freq,episode):
    plt.figure(figsize=(20, 12))

    # Weight Distribution
    plt.subplot(2, 3, 1)
    weights = np.concatenate([p.detach().cpu().numpy().flatten()
                            for p in agent.policy_net.parameters()])
    sns.histplot(weights, kde=True, bins=50)
    plt.title('Weight Distribution')

    # Weight Statistics
    plt.subplot(2, 3, 2)
    plt.plot(agent.weight_stats['means'], label='Mean Weights')
    plt.plot(agent.weight_stats['stds'], label='Weight Stds')
    plt.plot(agent.weight_stats['grad_means'], label='Gradient Means')
    plt.title('Weight Dynamics')
    plt.legend()

    # Reward Components
    plt.subplot(2, 3, 3)
    for comp in agent.reward_components:
        plt.plot(agent.reward_components[comp], label=comp)
    plt.title('Reward Components')
    plt.legend()

    # Weather Impact
    plt.subplot(2, 3, 4)
    weathers = list(agent.weather_history.keys())
    avg_rewards = [np.mean(agent.weather_history[w]) for w in weathers]
    plt.bar(weathers, avg_rewards)
    plt.title('Performance by Weather Condition')

    # Training Loss
    plt.subplot(2, 3, 5)
    plt.plot(agent.loss_history)
    plt.title('Training Loss')

    # Total Rewards
    plt.subplot(2, 3, 6)
    plt.plot(rewards)
    plt.title('Total Episode Rewards')

    plt.tight_layout()
    plt.savefig(f'training_progress_ep{episode+1}.png')
    plt.close()

def save_final_models(agent):
    """Save final model and optimizer states"""
    torch.save({
        'model_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'loss_history': agent.loss_history
    }, 'final_model.pth')

def generate_analysis_report(agent):
    """Generate comprehensive training report"""
    report = {
        'weight_stats': agent.weight_stats,
        'loss_history': agent.loss_history,
        'weather_performance': {
            w: np.mean(r) for w, r in agent.weather_history.items()
        },
        'training_metrics': {
            'final_epsilon': agent.epsilon,
            'total_episodes': len(agent.reward_history),
            'average_reward': np.mean(agent.reward_history[-100:])
        }
    }

    # Save JSON report
    with open('training_analysis.json', 'w') as f:
        json.dump(report, f, indent=4)

    # Generate weight distribution plot
    plt.figure(figsize=(10, 6))
    weights = np.concatenate([p.detach().cpu().numpy().flatten()
                            for p in agent.policy_net.parameters()])
    sns.histplot(weights, kde=True, bins=50)
    plt.title('Final Weight Distribution')
    plt.savefig('final_weights.png')
    plt.close()
# ----------------------------
# 5. Training Loop
# ----------------------------
def advanced_training():
    env = gym.make("CarRacing-v3", render_mode=None)
    weather_sim = WeatherSimulator()
    enhanced_env = CarRacingEnv(env, weather_sim)

    agent = Tolo_agent(
        state_dim=(96, 96, 3),
        action_dim=5,
        weather_simulator=weather_sim
    )

    # Training parameters
    episodes = 500
    reward_tracker = []
    weather_frequency = defaultdict(int)

    for episode in range(episodes):
        state, _ = enhanced_env.reset()
        total_reward = 0
        done = False
        weather_frequency[enhanced_env.weather.current_weather] += 1

        while not done:
            # Get current weather
            weather = enhanced_env.weather.current_weather

            # Select and execute action
            action = agent.select_action(state, weather)
            next_state, reward, done, _, info = enhanced_env.step(action)
            reward_info = info.get('reward_info', {})  # Get reward_info from info

            # Store experience with weather info
            agent.memory.append((
                state,
                weather,
                action,
                reward,
                next_state,
                enhanced_env.weather.current_weather,
                done,
                reward_info # Pass reward_info to memory
            ))


            # Update networks
            loss = agent.update()

            # Track metrics
            total_reward += reward
            state = next_state

        # Post-episode updates
        reward_tracker.append(total_reward)
        agent.weather_history[weather].append(total_reward)

        # Visualization and reporting
        if (episode+1) % 100 == 0 or episode==0:
            plot_training_progress(agent, reward_tracker, weather_frequency,episode)
        if (episode+1) % 10 == 0 or episode==0:
            print(f"Episode: {episode+1}, Total Reward: {total_reward}")

    save_final_models(agent)
    generate_analysis_report(agent)
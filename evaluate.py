"""
Evaluation script for trained TDM models
"""
import os
import yaml
import numpy as np
import torch
import argparse

import gymnasium as gym
from tdm import TDM
from env_wrapper import TDMEnvWrapper
from mpc_planner import TaskSpecificPlanner


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path, config, device='cpu'):
    """Load trained TDM model"""
    # Create environment to get dimensions
    env_name = config['env']['name']
    task_type = config['task']['locomotion_task_type']
    
    env = gym.make(env_name)
    env = TDMEnvWrapper(env, task_type, config['task'])
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = (env.action_space.low, env.action_space.high)
    goal_dim = env.goal_dim
    
    # Create TDM
    tdm = TDM(state_dim, action_dim, goal_dim, action_range, config, device)
    
    # Load weights
    tdm.load(model_path)
    
    return tdm, env


def evaluate_model(tdm, env, planner, num_episodes=10, render=False, config=None):
    """Evaluate TDM model"""
    episode_rewards = []
    episode_distances = []
    episode_lengths = []
    success_count = 0
    
    print(f"\nEvaluating model for {num_episodes} episodes...")
    print(f"{'='*60}")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        goal = env.get_goal()
        episode_reward = 0
        episode_length = 0
        
        if render:
            env.render()
        
        for step in range(config['env']['max_episode_steps']):
            # Select action
            tau = config['tdm']['tau_max']
            action = planner.select_action(obs, goal, tau)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_distances.append(info.get('distance_to_goal', 0))
        episode_lengths.append(episode_length)
        
        if info.get('goal_reached', False):
            success_count += 1
        
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Distance to goal: {info.get('distance_to_goal', 0):.4f}")
        print(f"  Length: {episode_length}")
        print(f"  Success: {info.get('goal_reached', False)}")
        print()
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_distance = np.mean(episode_distances)
    std_distance = np.std(episode_distances)
    mean_length = np.mean(episode_lengths)
    success_rate = success_count / num_episodes
    
    print(f"{'='*60}")
    print(f"Evaluation Results:")
    print(f"  Mean Distance: {mean_distance:.4f} ± {std_distance:.4f} (lower is better)")
    print(f"  Success Rate: {success_rate:.2%} ({success_count}/{num_episodes})")
    print(f"  Mean Length: {mean_length:.1f}")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f} (for reference)")
    print(f"{'='*60}\n")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'mean_length': mean_length,
        'success_rate': success_rate
    }


def visualize_trajectory(tdm, env, planner, num_episodes=3, config=None):
    """Visualize TDM trajectories"""
    print(f"\nVisualizing {num_episodes} trajectories...")
    print(f"{'='*60}")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        goal = env.get_goal()
        episode_reward = 0
        episode_length = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"Goal: {goal}")
        print(f"{'='*60}")
        
        env.render()
        
        for step in range(config['env']['max_episode_steps']):
            # Select action
            tau = config['tdm']['tau_max']
            action = planner.select_action(obs, goal, tau)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            env.render()
            
            print(f"Step {step + 1}: Reward={reward:.2f}, Distance={info.get('distance_to_goal', 0):.4f}")
            
            if terminated or truncated:
                print(f"Episode finished at step {step + 1}")
                break
        
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Final Distance: {info.get('distance_to_goal', 0):.4f}")
        print(f"  Success: {info.get('goal_reached', False)}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained TDM model')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model (.pt file)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize trajectories (only a few episodes)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    tdm, env = load_model(args.model, config, device)
    
    # Create planner
    planner = TaskSpecificPlanner(tdm, config, config['env']['name'], 
                                  config['task']['locomotion_task_type'])
    
    # Evaluate or visualize
    if args.visualize:
        visualize_trajectory(tdm, env, planner, num_episodes=3, config=config)
    else:
        evaluate_model(tdm, env, planner, num_episodes=args.episodes, 
                      render=args.render, config=config)
    
    # Close environment
    env.close()


if __name__ == '__main__':
    main()


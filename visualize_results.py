"""
Visualization script for TDM results
"""
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    plot_training_curves,
    visualize_tdm_prediction,
    compare_horizons,
    compute_goal_achievement_rate
)


def visualize_training(log_dir, save_dir=None):
    """Visualize training curves"""
    print(f"\nVisualizing training curves from {log_dir}")
    
    if save_dir:
        save_path = Path(save_dir) / 'training_curves.png'
        plot_training_curves(log_dir, save_path=str(save_path))
    else:
        plot_training_curves(log_dir)


def visualize_predictions(model_path, config_path='config.yaml', save_dir=None):
    """Visualize TDM predictions"""
    print(f"\nVisualizing TDM predictions from {model_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    import gymnasium as gym
    from tdm import TDM
    from env_wrapper import TDMEnvWrapper
    
    env = gym.make(config['env']['name'])
    env = TDMEnvWrapper(env, config['task']['locomotion_task_type'], config['task'])
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = (env.action_space.low, env.action_space.high)
    goal_dim = env.goal_dim
    
    tdm = TDM(state_dim, action_dim, goal_dim, action_range, config, device)
    tdm.load(model_path)
    
    # Get state and goal
    obs, info = env.reset()
    goal = env.get_goal()
    
    # Visualize
    tau_range = range(0, config['tdm']['tau_max'] + 1, 5)
    visualize_tdm_prediction(tdm, obs, goal, tau_range, device)
    
    env.close()


def compare_horizons_visualization(model_path, config_path='config.yaml', 
                                   horizons=None, num_episodes=10):
    """Compare performance for different horizons"""
    print(f"\nComparing horizons for model {model_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    import gymnasium as gym
    from tdm import TDM
    from env_wrapper import TDMEnvWrapper
    from mpc_planner import TaskSpecificPlanner
    
    env = gym.make(config['env']['name'])
    env = TDMEnvWrapper(env, config['task']['locomotion_task_type'], config['task'])
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = (env.action_space.low, env.action_space.high)
    goal_dim = env.goal_dim
    
    tdm = TDM(state_dim, action_dim, goal_dim, action_range, config, device)
    tdm.load(model_path)
    
    planner = TaskSpecificPlanner(tdm, config, config['env']['name'], 
                                  config['task']['locomotion_task_type'])
    
    # Compare horizons
    if horizons is None:
        horizons = [5, 10, 15, 20, 25]
    
    results = compare_horizons(tdm, env, planner, horizons, num_episodes, config)
    
    env.close()
    
    return results


def visualize_goal_achievement(model_path, config_path='config.yaml', num_episodes=100):
    """Visualize goal achievement rate"""
    print(f"\nVisualizing goal achievement for model {model_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    import gymnasium as gym
    from tdm import TDM
    from env_wrapper import TDMEnvWrapper
    from mpc_planner import TaskSpecificPlanner
    
    env = gym.make(config['env']['name'])
    env = TDMEnvWrapper(env, config['task']['locomotion_task_type'], config['task'])
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = (env.action_space.low, env.action_space.high)
    goal_dim = env.goal_dim
    
    tdm = TDM(state_dim, action_dim, goal_dim, action_range, config, device)
    tdm.load(model_path)
    
    planner = TaskSpecificPlanner(tdm, config, config['env']['name'], 
                                  config['task']['locomotion_task_type'])
    
    # Compute goal achievement rate
    results = compute_goal_achievement_rate(tdm, env, planner, num_episodes, config)
    
    env.close()
    
    return results


def create_summary_plot(log_dir, model_path, config_path='config.yaml', save_path=None):
    """Create summary plot with multiple visualizations"""
    print(f"\nCreating summary plot")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Training curves
    ax1 = plt.subplot(2, 3, 1)
    try:
        from torch.utils.tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        if 'train/episode_reward' in ea.Tags()['scalars']:
            reward_events = ea.Scalars('train/episode_reward')
            episodes = [e.step for e in reward_events]
            rewards = [e.value for e in reward_events]
            ax1.plot(episodes, rewards)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Episode Reward')
            ax1.set_title('Training Episode Reward')
            ax1.grid(True)
    except Exception as e:
        ax1.text(0.5, 0.5, f'No data\n{str(e)}', 
                ha='center', va='center', transform=ax1.transAxes)
    
    # 2. Evaluation performance
    ax2 = plt.subplot(2, 3, 2)
    try:
        if 'eval/mean_reward' in ea.Tags()['scalars']:
            eval_events = ea.Scalars('eval/mean_reward')
            steps = [e.step for e in eval_events]
            rewards = [e.value for e in eval_events]
            ax2.plot(steps, rewards)
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Mean Reward')
            ax2.set_title('Evaluation Performance')
            ax2.grid(True)
    except Exception as e:
        ax2.text(0.5, 0.5, f'No data\n{str(e)}', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Success rate
    ax3 = plt.subplot(2, 3, 3)
    try:
        if 'eval/success_rate' in ea.Tags()['scalars']:
            success_events = ea.Scalars('eval/success_rate')
            steps = [e.step for e in success_events]
            success_rates = [e.value for e in success_events]
            ax3.plot(steps, success_rates)
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('Success Rate')
            ax3.set_title('Evaluation Success Rate')
            ax3.set_ylim([0, 1])
            ax3.grid(True)
    except Exception as e:
        ax3.text(0.5, 0.5, f'No data\n{str(e)}', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Critic loss
    ax4 = plt.subplot(2, 3, 4)
    try:
        if 'train/critic_loss' in ea.Tags()['scalars']:
            loss_events = ea.Scalars('train/critic_loss')
            steps = [e.step for e in loss_events]
            losses = [e.value for e in loss_events]
            ax4.plot(steps, losses)
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('Critic Loss')
            ax4.set_title('Critic Loss')
            ax4.grid(True)
    except Exception as e:
        ax4.text(0.5, 0.5, f'No data\n{str(e)}', 
                ha='center', va='center', transform=ax4.transAxes)
    
    # 5. Noise decay
    ax5 = plt.subplot(2, 3, 5)
    try:
        if 'train/noise_std' in ea.Tags()['scalars']:
            noise_events = ea.Scalars('train/noise_std')
            steps = [e.step for e in noise_events]
            noise_stds = [e.value for e in noise_events]
            ax5.plot(steps, noise_stds)
            ax5.set_xlabel('Steps')
            ax5.set_ylabel('Noise Std')
            ax5.set_title('Exploration Noise Decay')
            ax5.grid(True)
    except Exception as e:
        ax5.text(0.5, 0.5, f'No data\n{str(e)}', 
                ha='center', va='center', transform=ax5.transAxes)
    
    # 6. Model info
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    info_text = f"""
Model Information

Environment: {config['env']['name']}
Task Type: {config['task']['locomotion_task_type']}
Tau Max: {config['tdm']['tau_max']}
Vectorized: {config['tdm']['vectorized_supervision']}

Training:
- Batch Size: {config['training']['batch_size']}
- Updates/Step: {config['training']['updates_per_step']}
- Learning Rate (Critic): {config['training']['learning_rate_critic']}
- Learning Rate (Actor): {config['training']['learning_rate_actor']}
"""
    ax6.text(0.1, 0.5, info_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Summary plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize TDM results')
    parser.add_argument('--log_dir', type=str, help='Directory containing TensorBoard logs')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'training', 'predictions', 'horizons', 'goals', 'summary'],
                       help='Visualization mode')
    parser.add_argument('--save_dir', type=str, help='Directory to save plots')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        if args.log_dir:
            visualize_training(args.log_dir, args.save_dir)
        if args.model:
            visualize_predictions(args.model, args.config, args.save_dir)
            compare_horizons_visualization(args.model, args.config)
            visualize_goal_achievement(args.model, args.config)
    elif args.mode == 'training':
        if args.log_dir:
            visualize_training(args.log_dir, args.save_dir)
    elif args.mode == 'predictions':
        if args.model:
            visualize_predictions(args.model, args.config, args.save_dir)
    elif args.mode == 'horizons':
        if args.model:
            compare_horizons_visualization(args.model, args.config)
    elif args.mode == 'goals':
        if args.model:
            visualize_goal_achievement(args.model, args.config)
    elif args.mode == 'summary':
        if args.log_dir and args.model:
            save_path = Path(args.save_dir) / 'summary.png' if args.save_dir else None
            create_summary_plot(args.log_dir, args.model, args.config, save_path)


if __name__ == '__main__':
    main()





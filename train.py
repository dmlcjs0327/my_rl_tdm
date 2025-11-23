"""
Training script for TDM
"""
import os
import yaml
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

import gymnasium as gym
from tdm import TDM
from env_wrapper import TDMEnvWrapper
from mpc_planner import TaskSpecificPlanner


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_env(config):
    """Create and wrap environment"""
    env_name = config['env']['name']
    task_type = config['task']['locomotion_task_type']
    
    # Create base environment
    env = gym.make(env_name)
    
    # Wrap with TDM wrapper
    env = TDMEnvWrapper(env, task_type, config['task'])
    
    return env


def evaluate(env, tdm, planner, num_episodes=10, config={}):
    """Evaluate TDM policy"""
    episode_rewards = []
    episode_distances = []
    episode_lengths = []
    success_rate = 0
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        goal = env.get_goal()
        
        for step in range(config['env']['max_episode_steps']):
            # Select action
            tau = config['tdm']['tau_max']
            action = planner.select_action(obs, goal, tau)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_distances.append(info.get('distance_to_goal', 0))
        episode_lengths.append(episode_length)
        
        if info.get('goal_reached', False):
            success_rate += 1
    
    success_rate /= num_episodes
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_distance': np.mean(episode_distances),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_rate
    }


def train(config):
    """Main training loop"""
    
    # Set random seed
    set_seed(config['seed'])
    
    # Create environment
    env = create_env(config)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = (env.action_space.low, env.action_space.high)  # type: ignore
    
    # Get goal dimension from wrapper
    goal_dim = env.goal_dim
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create TDM agent
    tdm = TDM(state_dim, action_dim, goal_dim, action_range, config, device)
    
    # Create planner
    planner = TaskSpecificPlanner(tdm, config, config['env']['name'], 
                                  config['task']['locomotion_task_type'])
    
    # Create logging directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config['logging']['log_dir'], 
                          f"{config['env']['name']}_{timestamp}")
    # Ensure directory exists before creating TensorBoard writer
    os.makedirs(log_dir, exist_ok=True)
    # Use absolute path for TensorBoard
    log_dir = os.path.abspath(log_dir)
    
    # TensorBoard writer
    if config['logging']['tensorboard']:
        writer = SummaryWriter(log_dir)
    
    # Save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Training loop
    total_steps = 0
    episode = 0
    episode_reward = 0
    episode_length = 0
    
    # Best model tracking (for preventing performance collapse)
    best_eval_distance = float('inf')
    best_model_path = None
    patience = config['training'].get('patience', None)  # Early stopping patience
    patience_counter = 0
    
    obs, info = env.reset()
    goal = env.get_goal()
    
    print(f"\n{'='*60}")
    print(f"Training TDM on {config['env']['name']}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")
    
    while total_steps < config['training']['total_timesteps']:
        # Select action
        tau = config['tdm']['tau_max']
        action = planner.select_action(obs, goal, tau)  # type: ignore
        
        # Add exploration noise
        noise = np.random.normal(0, tdm.noise_std, size=action_dim)
        action = np.clip(action + noise, action_range[0], action_range[1])
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Store transition (done = terminated or truncated)
        done = terminated or truncated
        tdm.replay_buffer.add(obs, action, next_obs, reward, done, goal)
        
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        total_steps += 1
        
        # Train TDM
        if total_steps >= config['training']['batch_size']:
            for _ in range(config['training']['updates_per_step']):
                train_info = tdm.train_step()
                
                if train_info is not None:
                    # Log training metrics
                    if config['logging']['tensorboard'] and total_steps % config['logging']['log_frequency'] == 0:
                        writer.add_scalar('train/critic_loss', train_info['critic_loss'], total_steps)
                        writer.add_scalar('train/actor_loss', train_info['actor_loss'], total_steps)
        
        # Decay noise (with minimum threshold)
        if total_steps % 1000 == 0:
            tdm.noise_std = max(tdm.noise_std * config['training']['noise_decay'], 
                               config['training'].get('min_noise_std', 0.01))
        
        # Episode finished
        done = terminated or truncated
        if done or episode_length >= config['env']['max_episode_steps']:
            # Log episode metrics
            if config['logging']['tensorboard']:
                writer.add_scalar('train/episode_reward', episode_reward, episode)
                writer.add_scalar('train/episode_length', episode_length, episode)
                writer.add_scalar('train/noise_std', tdm.noise_std, total_steps)
            
            # Reset environment
            obs, info = env.reset()
            goal = env.get_goal()
            episode_reward = 0
            episode_length = 0
            episode += 1
            
            # Evaluate periodically
            if episode % (config['training']['eval_frequency'] // config['env']['max_episode_steps']) == 0:
                eval_results = evaluate(env, tdm, planner, 
                                       config['training']['eval_episodes'], 
                                       config)
                
                print(f"Episode {episode}, Total Steps: {total_steps}")
                print(f"  Mean Distance: {eval_results['mean_distance']:.4f} (lower is better)")
                print(f"  Success Rate: {eval_results['success_rate']:.2%}")
                print(f"  Mean Episode Length: {eval_results['mean_length']:.1f}")
                print(f"  Noise Std: {tdm.noise_std:.4f}")
                
                # Track best model (prevent performance collapse)
                current_distance = eval_results['mean_distance']
                if current_distance < best_eval_distance:
                    best_eval_distance = current_distance
                    patience_counter = 0
                    # Save best model
                    best_model_path = os.path.join(log_dir, 'model_best.pt')
                    tdm.save(best_model_path)
                    print(f"  ✓ New best model saved! (Distance: {best_eval_distance:.4f})")
                else:
                    patience_counter += 1
                    print(f"  Best distance: {best_eval_distance:.4f} (patience: {patience_counter}/{patience if patience else 'N/A'})")
                
                # Early stopping if performance doesn't improve
                if patience is not None and patience_counter >= patience:
                    print(f"\nEarly stopping: No improvement for {patience} evaluations")
                    print(f"Loading best model from {best_model_path}")
                    tdm.load(best_model_path)
                    break
                
                print()
                
                if config['logging']['tensorboard']:
                    # Mean reward removed (redundant with mean_distance)
                    writer.add_scalar('eval/mean_distance', eval_results['mean_distance'], total_steps)
                    writer.add_scalar('eval/success_rate', eval_results['success_rate'], total_steps)
                    writer.add_scalar('eval/mean_length', eval_results['mean_length'], total_steps)
                    writer.add_scalar('eval/best_distance', best_eval_distance, total_steps)
                    # Log learning rates if decay is enabled
                    if config['training'].get('use_lr_decay', False):
                        writer.add_scalar('train/lr_actor', 
                                         tdm.actor_optimizer.param_groups[0]['lr'], total_steps)
                        writer.add_scalar('train/lr_critic', 
                                         tdm.critic_optimizer.param_groups[0]['lr'], total_steps)
        
        # Save model periodically
        if total_steps % config['logging']['save_frequency'] == 0:
            model_path = os.path.join(log_dir, f'model_{total_steps}.pt')
            tdm.save(model_path)
            print(f"Model saved to {model_path}")
    
    # Final save
    final_model_path = os.path.join(log_dir, 'model_final.pt')
    tdm.save(final_model_path)
    print(f"\nTraining completed!")
    print(f"Final model saved to {final_model_path}")
    
    # Close environment
    env.close()
    
    if config['logging']['tensorboard']:
        writer.close()


if __name__ == '__main__':
    # Load configuration
    config = load_config('config.yaml')
    
    # Train
    train(config)


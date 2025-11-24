"""
Training script for TDM with Curriculum Learning and Warm-up
"""
import os
import yaml
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

import gymnasium as gym
from tdm import TDM
from env_wrapper import TDMEnvWrapper, GoalSampler
from mpc_planner import TaskSpecificPlanner
from curriculum_learning import CurriculumLearning, WarmUpPeriod
from policy_collapse_detector import PolicyCollapseDetector


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
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


def train(config, experiment_id=None, log_dir_base=None):
    """
    Main training loop with Curriculum Learning and Warm-up
    
    Args:
        config: Configuration dictionary
        experiment_id: Experiment ID for logging
        log_dir_base: Base directory for logging
    """
    
    # Set random seed
    set_seed(config['seed'])
    
    # Create environment
    env = create_env(config)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = (env.action_space.low, env.action_space.high)
    goal_dim = env.goal_dim
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create TDM agent
    tdm = TDM(state_dim, action_dim, goal_dim, action_range, config, device)
    
    # Create planner
    planner = TaskSpecificPlanner(tdm, config, config['env']['name'], 
                                  config['task']['locomotion_task_type'])
    
    # Goal sampler
    goal_sampler = GoalSampler(config['task']['locomotion_task_type'], 
                               config['env']['name'])
    
    # Curriculum Learning
    use_curriculum = config['training'].get('use_curriculum', False)
    curriculum = None
    if use_curriculum:
        curriculum_config = config['training'].get('curriculum', {})
        curriculum = CurriculumLearning(
            goal_sampler,
            initial_difficulty=curriculum_config.get('initial_difficulty', 0.1),
            final_difficulty=curriculum_config.get('final_difficulty', 1.0),
            curriculum_type=curriculum_config.get('type', 'distance'),
            schedule=curriculum_config.get('schedule', 'linear')
        )
    
    # Warm-up Period
    use_warmup = config['training'].get('use_warmup', False)
    warmup = None
    if use_warmup:
        warmup_config = config['training'].get('warmup', {})
        warmup = WarmUpPeriod(
            warmup_steps=warmup_config.get('steps', 10000),
            initial_noise_std=warmup_config.get('initial_noise_std', 0.5),
            final_noise_std=warmup_config.get('final_noise_std', 0.2),
            initial_lr_multiplier=warmup_config.get('initial_lr_multiplier', 0.1),
            final_lr_multiplier=warmup_config.get('final_lr_multiplier', 1.0)
        )
    
    # Create logging directory
    if log_dir_base is None:
        log_dir_base = config['logging']['log_dir']
    
    if experiment_id:
        timestamp = experiment_id
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    log_dir = os.path.join(log_dir_base, 
                          f"{config['env']['name']}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.abspath(log_dir)
    
    # TensorBoard writer
    if config['logging']['tensorboard']:
        writer = SummaryWriter(log_dir)
    
    # Save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    # Training loop
    total_steps = 0
    episode = 0
    episode_reward = 0
    episode_length = 0
    
    # Best model tracking
    best_eval_distance = float('inf')
    best_model_path = None
    patience = config['training'].get('patience', None)
    patience_counter = 0
    
    # Policy Collapse Detector
    use_collapse_detection = config['training'].get('detect_policy_collapse', False)
    collapse_detector = None
    if use_collapse_detection:
        collapse_config = config['training'].get('collapse_detection', {})
        collapse_detector = PolicyCollapseDetector(
            window_size=collapse_config.get('window_size', 5),
            collapse_threshold=collapse_config.get('collapse_threshold', 0.3),
            min_evaluations=collapse_config.get('min_evaluations', 3),
            stability_threshold=collapse_config.get('stability_threshold', 0.5)
        )
        print("Policy Collapse Detection: Enabled")
    
    obs, info = env.reset()
    initial_state = obs.copy()
    
    # Curriculum learning: 초기 목표 샘플링
    goal = env.get_goal()  # 기본 목표 가져오기
    if curriculum is not None:
        curriculum_goal = curriculum.sample_goal(initial_state)
        env.current_goal = curriculum_goal
        goal = curriculum_goal
    
    print(f"\n{'='*60}")
    print(f"Training TDM on {config['env']['name']}")
    print(f"Log directory: {log_dir}")
    if curriculum:
        print(f"Curriculum Learning: {curriculum.curriculum_type} ({curriculum.schedule})")
    if warmup:
        print(f"Warm-up Period: {warmup.warmup_steps} steps")
    print(f"{'='*60}\n")
    
    while total_steps < config['training']['total_timesteps']:
        # Warm-up 업데이트
        if warmup:
            warmup.update_step(total_steps)
            # 노이즈 조정
            tdm.noise_std = warmup.get_noise_std(config['training']['noise_std'])
            # 학습률 조정 (옵션)
            if warmup.is_warmup():
                lr_multiplier = warmup.get_lr_multiplier()
                for param_group in tdm.actor_optimizer.param_groups:
                    param_group['lr'] = config['training']['learning_rate_actor'] * lr_multiplier
                for param_group in tdm.critic_optimizer.param_groups:
                    param_group['lr'] = config['training']['learning_rate_critic'] * lr_multiplier
        
        # Select action
        tau = config['tdm']['tau_max']
        action = planner.select_action(obs, goal, tau)
        
        # Add exploration noise
        noise = np.random.normal(0, tdm.noise_std, size=action_dim)
        action = np.clip(action + noise, action_range[0], action_range[1])
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Store transition
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
        
        # Decay noise (warm-up이 아닐 때만)
        if not warmup or not warmup.is_warmup():
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
                if curriculum:
                    writer.add_scalar('train/curriculum_difficulty', 
                                    curriculum.get_difficulty(), total_steps)
                if warmup:
                    writer.add_scalar('train/warmup_progress', 
                                    min(1.0, total_steps / warmup.warmup_steps), total_steps)
            
            # Reset environment
            obs, info = env.reset()
            initial_state = obs.copy()
            
            # Curriculum learning: 새로운 목표 샘플링
            goal = env.get_goal()  # 기본 목표 가져오기
            if curriculum is not None:
                # 진행도 업데이트
                progress = total_steps / config['training']['total_timesteps']
                curriculum.update_progress(progress)
                # 난이도에 맞는 목표 샘플링
                curriculum_goal = curriculum.sample_goal(initial_state)
                env.current_goal = curriculum_goal
                goal = curriculum_goal
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
                if curriculum:
                    print(f"  Curriculum Difficulty: {curriculum.get_difficulty():.3f}")
                print()
                
                # Track best model
                current_distance = eval_results['mean_distance']
                if current_distance < best_eval_distance:
                    best_eval_distance = current_distance
                    patience_counter = 0
                    best_model_path = os.path.join(log_dir, 'model_best.pt')
                    tdm.save(best_model_path)
                    print(f"  ✓ New best model saved! (Distance: {best_eval_distance:.4f})")
                else:
                    patience_counter += 1
                    print(f"  Best distance: {best_eval_distance:.4f} (patience: {patience_counter}/{patience if patience else 'N/A'})")
                
                # Policy Collapse Detection
                if collapse_detector:
                    collapse_info = collapse_detector.update(
                        eval_results['mean_distance'],
                        eval_results['success_rate']
                    )
                    
                    if collapse_info['is_collapsed']:
                        print(f"\n⚠️  POLICY COLLAPSE DETECTED!")
                        print(f"   Reason: {collapse_info['reason']}")
                        print(f"   Loading best model from {best_model_path}")
                        if best_model_path and os.path.exists(best_model_path):
                            tdm.load(best_model_path)
                        print(f"   Stopping training early due to policy collapse")
                        break
                    elif collapse_info['reason']:
                        print(f"  ⚠️  Warning: {collapse_info['reason']}")
                
                # Early stopping (patience-based)
                if patience is not None and patience_counter >= patience:
                    print(f"\nEarly stopping: No improvement for {patience} evaluations")
                    print(f"Loading best model from {best_model_path}")
                    if best_model_path and os.path.exists(best_model_path):
                        tdm.load(best_model_path)
                    break
                
                if config['logging']['tensorboard']:
                    writer.add_scalar('eval/mean_distance', eval_results['mean_distance'], total_steps)
                    writer.add_scalar('eval/success_rate', eval_results['success_rate'], total_steps)
                    writer.add_scalar('eval/mean_length', eval_results['mean_length'], total_steps)
                    writer.add_scalar('eval/best_distance', best_eval_distance, total_steps)
        
        # Save model periodically
        if total_steps % config['logging']['save_frequency'] == 0:
            model_path = os.path.join(log_dir, f'model_{total_steps}.pt')
            tdm.save(model_path)
    
    # Final save
    final_model_path = os.path.join(log_dir, 'model_final.pt')
    tdm.save(final_model_path)
    
    # Best model이 있으면 최종 모델로 복사
    if best_model_path and os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, final_model_path)
        print(f"\nBest model copied to final model")
    
    print(f"\nTraining completed!")
    print(f"Final model saved to {final_model_path}")
    print(f"Best distance: {best_eval_distance:.4f}")
    
    # Close environment
    env.close()
    
    if config['logging']['tensorboard']:
        writer.close()
    
    # Return final evaluation results
    return {
        'best_distance': best_eval_distance,
        'log_dir': log_dir,
        'model_path': final_model_path
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train TDM with Curriculum Learning')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--experiment-id', type=str, default=None,
                       help='Experiment ID for logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Train
    train(config, experiment_id=args.experiment_id)


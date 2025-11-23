"""
Example usage of TDM
"""
import yaml
import torch
import gymnasium as gym
from tdm import TDM
from env_wrapper import TDMEnvWrapper
from mpc_planner import TaskSpecificPlanner
from utils import (
    plot_training_curves,
    visualize_tdm_prediction,
    compare_horizons,
    analyze_goal_relabeling_impact
)


def example_train():
    """Example: Train TDM on Reacher task"""
    print("="*60)
    print("Example 1: Training TDM")
    print("="*60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Change environment
    config['env']['name'] = 'Reacher-v5'
    config['env']['max_episode_steps'] = 100
    config['training']['total_timesteps'] = 100000  # Shorter for demo
    
    # Save modified config
    with open('config_reacher.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("\nTo train the model, run:")
    print("python train.py")
    print("\nOr with custom config:")
    print("python train.py --config config_reacher.yaml")


def example_evaluate():
    """Example: Evaluate trained TDM"""
    print("\n" + "="*60)
    print("Example 2: Evaluating TDM")
    print("="*60)
    
    print("\nTo evaluate a trained model, run:")
    print("python evaluate.py --model ./logs/Reacher-v5_20240101_120000/model_final.pt")
    print("\nWith rendering:")
    print("python evaluate.py --model ./logs/Reacher-v5_20240101_120000/model_final.pt --render")
    print("\nVisualize trajectories:")
    print("python evaluate.py --model ./logs/Reacher-v5_20240101_120000/model_final.pt --visualize")


def example_analyze():
    """Example: Analyze TDM behavior"""
    print("\n" + "="*60)
    print("Example 3: Analyzing TDM")
    print("="*60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create environment
    env = gym.make('Reacher-v5')
    env = TDMEnvWrapper(env, 'end_effector', config['task'])
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = (env.action_space.low, env.action_space.high)
    goal_dim = env.goal_dim
    
    # Create TDM
    tdm = TDM(state_dim, action_dim, goal_dim, action_range, config, device)
    
    # Load weights (if available)
    try:
        tdm.load('./logs/Reacher-v5_20240101_120000/model_final.pt')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model not found. Please train a model first.")
        return
    
    # Create planner
    planner = TaskSpecificPlanner(tdm, config, 'Reacher-v5', 'end_effector')
    
    print("\n1. Plot training curves:")
    print("   plot_training_curves('./logs/Reacher-v5_20240101_120000')")
    
    print("\n2. Visualize TDM predictions:")
    print("   obs, _ = env.reset()")
    print("   goal = env.get_goal()")
    print("   visualize_tdm_prediction(tdm, obs, goal, range(0, 25), device)")
    
    print("\n3. Compare different horizons:")
    print("   compare_horizons(tdm, env, planner, [5, 10, 15, 20, 25], num_episodes=10, config=config)")
    
    print("\n4. Analyze goal relabeling:")
    print("   analyze_goal_relabeling_impact(tdm, env, num_samples=1000)")


def example_custom_task():
    """Example: Custom task configuration"""
    print("\n" + "="*60)
    print("Example 4: Custom Task Configuration")
    print("="*60)
    
    # Create custom config for Ant position+velocity task
    config = {
        'env': {
            'name': 'Ant-v5',
            'max_episode_steps': 50
        },
        'task': {
            'locomotion_task_type': 'position_velocity',
            'goal_sampling_strategy': 'future',
            'position_weight': 0.1,
            'velocity_weight': 0.9
        },
        'tdm': {
            'tau_max': 25,
            'distance_metric': 'L1',
            'vectorized_supervision': True,
            'reward_scale': 1.0
        },
        'network': {
            'actor': {
                'hidden_sizes': [300, 300],
                'activation': 'relu',
                'output_activation': 'tanh'
            },
            'critic': {
                'hidden_sizes': [300, 300],
                'activation': 'relu'
            }
        },
        'training': {
            'total_timesteps': 1000000,
            'learning_rate_actor': 0.0001,
            'learning_rate_critic': 0.001,
            'batch_size': 128,
            'buffer_size': 1000000,
            'updates_per_step': 10,
            'polyak': 0.999,
            'noise_type': 'normal',
            'noise_std': 0.2,
            'noise_decay': 0.9995,
            'noise_decay_steps': 100000,
            'eval_frequency': 5000,
            'eval_episodes': 10
        },
        'mpc': {
            'method': 'direct',
            'horizon': 15,
            'num_samples': 10000
        },
        'logging': {
            'log_dir': './logs',
            'tensorboard': True,
            'save_frequency': 10000,
            'log_frequency': 100
        },
        'seed': 42
    }
    
    # Save custom config
    with open('config_ant_custom.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("\nCustom configuration created: config_ant_custom.yaml")
    print("\nTo train with this config:")
    print("python train.py --config config_ant_custom.yaml")


def example_hyperparameter_tuning():
    """Example: Hyperparameter tuning"""
    print("\n" + "="*60)
    print("Example 5: Hyperparameter Tuning")
    print("="*60)
    
    print("\nKey hyperparameters to tune:")
    print("1. tau_max: Planning horizon (try 10, 15, 20, 25)")
    print("2. updates_per_step: Update frequency (try 1, 5, 10, 20)")
    print("3. reward_scale: Reward scaling (try 0.01, 1, 100)")
    print("4. learning_rate_critic: Critic learning rate (try 0.0001, 0.001)")
    print("5. batch_size: Batch size (try 64, 128, 256)")
    
    print("\nExample grid search script:")
    print("""
import itertools
import subprocess

# Define hyperparameter ranges
tau_max_values = [15, 20, 25]
updates_per_step_values = [5, 10, 20]
reward_scale_values = [0.1, 1.0, 10.0]

best_config = None
best_performance = -float('inf')

for tau_max, updates, scale in itertools.product(
    tau_max_values, updates_per_step_values, reward_scale_values
):
    # Modify config
    config['tdm']['tau_max'] = tau_max
    config['training']['updates_per_step'] = updates
    config['tdm']['reward_scale'] = scale
    
    # Save config
    with open('config_tune.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Train
    subprocess.run(['python', 'train.py', '--config', 'config_tune.yaml'])
    
    # Evaluate and get performance
    # (Implement evaluation logic)
    
    # Track best configuration
    # ...

print(f"Best configuration: {best_config}")
""")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("TDM Usage Examples")
    print("="*60)
    
    example_train()
    example_evaluate()
    example_analyze()
    example_custom_task()
    example_hyperparameter_tuning()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nFor more information, see README.md")


if __name__ == '__main__':
    main()



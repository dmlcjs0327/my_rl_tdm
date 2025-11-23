"""
Utility functions for TDM
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_training_curves(log_dir, save_path=None):
    """
    Plot training curves from TensorBoard logs
    
    Args:
        log_dir: directory containing TensorBoard logs
        save_path: path to save figure (optional)
    """
    from torch.utils.tensorboard.writer import SummaryWriter
    
    # Read events from TensorBoard
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    # Get scalar tags
    scalar_tags = ea.Tags()['scalars']
    
    # Plot metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode reward
    if 'train/episode_reward' in scalar_tags:
        reward_events = ea.Scalars('train/episode_reward')
        episodes = [e.step for e in reward_events]
        rewards = [e.value for e in reward_events]
        axes[0, 0].plot(episodes, rewards)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('Training Episode Reward')
        axes[0, 0].grid(True)
    
    # Critic loss
    if 'train/critic_loss' in scalar_tags:
        loss_events = ea.Scalars('train/critic_loss')
        steps = [e.step for e in loss_events]
        losses = [e.value for e in loss_events]
        axes[0, 1].plot(steps, losses)
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Critic Loss')
        axes[0, 1].set_title('Critic Loss')
        axes[0, 1].grid(True)
    
    # Evaluation reward
    if 'eval/mean_reward' in scalar_tags:
        eval_events = ea.Scalars('eval/mean_reward')
        steps = [e.step for e in eval_events]
        rewards = [e.value for e in eval_events]
        axes[1, 0].plot(steps, rewards)
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Mean Reward')
        axes[1, 0].set_title('Evaluation Performance')
        axes[1, 0].grid(True)
    
    # Success rate
    if 'eval/success_rate' in scalar_tags:
        success_events = ea.Scalars('eval/success_rate')
        steps = [e.step for e in success_events]
        success_rates = [e.value for e in success_events]
        axes[1, 1].plot(steps, success_rates)
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Evaluation Success Rate')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_tdm_prediction(tdm, state, goal, tau_range, device='cpu'):
    """
    Visualize TDM predictions for different horizons
    
    Args:
        tdm: trained TDM model
        state: current state
        goal: goal state
        tau_range: range of horizons to visualize
        device: torch device
    """
    state_tensor = torch.FloatTensor(state).to(device)
    goal_tensor = torch.FloatTensor(goal).to(device)
    
    predictions = []
    q_values = []
    
    for tau in tau_range:
        tau_tensor = torch.FloatTensor([tau]).to(device)
        
        # Get action from actor
        with torch.no_grad():
            action = tdm.actor(state_tensor.unsqueeze(0))
            predicted_state = tdm.critic(state_tensor.unsqueeze(0), 
                                        action, 
                                        goal_tensor.unsqueeze(0), 
                                        tau_tensor.unsqueeze(0))
            q_value = tdm.critic.compute_q_value(state_tensor.unsqueeze(0),
                                                 action,
                                                 goal_tensor.unsqueeze(0),
                                                 tau_tensor.unsqueeze(0))
        
        predictions.append(predicted_state.cpu().numpy()[0])
        q_values.append(q_value.cpu().numpy()[0])
    
    predictions = np.array(predictions)
    q_values = np.array(q_values)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Predictions
    for i in range(min(3, predictions.shape[1])):
        axes[0].plot(tau_range, predictions[:, i], label=f'Dim {i}')
    axes[0].axhline(y=goal[0], color='r', linestyle='--', label='Goal')
    axes[0].set_xlabel('Horizon (tau)')
    axes[0].set_ylabel('Predicted State')
    axes[0].set_title('TDM Predictions vs Horizon')
    axes[0].legend()
    axes[0].grid(True)
    
    # Q-values
    axes[1].plot(tau_range, q_values)
    axes[1].set_xlabel('Horizon (tau)')
    axes[1].set_ylabel('Q-value')
    axes[1].set_title('Q-value vs Horizon')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def compute_goal_achievement_rate(env, tdm, planner, num_episodes=100, config=None):
    """
    Compute goal achievement rate for different goal distances
    
    Args:
        env: environment
        tdm: trained TDM model
        planner: planner
        num_episodes: number of episodes to test
        config: configuration dictionary
    
    Returns:
        Dictionary with goal distances and success rates
    """
    from env_wrapper import GoalSampler
    
    goal_sampler = GoalSampler(config['task']['locomotion_task_type'], 
                               config['env']['name'])
    
    # Test different goal distances
    goal_distances = [0.5, 1.0, 2.0, 3.0, 4.0]
    success_rates = []
    
    for goal_dist in goal_distances:
        success_count = 0
        
        for _ in range(num_episodes):
            obs, info = env.reset()
            
            # Sample goal at specific distance
            goal = goal_sampler.sample()
            
            episode_length = 0
            for step in range(config['env']['max_episode_steps']):
                tau = config['tdm']['tau_max']
                action = planner.select_action(obs, goal, tau)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            if info.get('goal_reached', False):
                success_count += 1
        
        success_rate = success_count / num_episodes
        success_rates.append(success_rate)
        
        print(f"Goal Distance: {goal_dist:.1f}, Success Rate: {success_rate:.2%}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(goal_distances, success_rates, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Goal Distance')
    plt.ylabel('Success Rate')
    plt.title('Goal Achievement Rate vs Distance')
    plt.grid(True)
    plt.ylim([0, 1])
    plt.show()
    
    return dict(zip(goal_distances, success_rates))


def compare_horizons(tdm, env, planner, horizons, num_episodes=10, config=None):
    """
    Compare performance for different planning horizons
    
    Args:
        tdm: trained TDM model
        env: environment
        planner: planner
        horizons: list of horizons to test
        num_episodes: number of episodes per horizon
        config: configuration dictionary
    
    Returns:
        Dictionary with results for each horizon
    """
    results = {}
    
    for horizon in horizons:
        episode_rewards = []
        success_count = 0
        
        for _ in range(num_episodes):
            obs, info = env.reset()
            goal = env.get_goal()
            episode_reward = 0
            
            for step in range(config['env']['max_episode_steps']):
                action = planner.select_action(obs, goal, horizon)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            if info.get('goal_reached', False):
                success_count += 1
        
        results[horizon] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': success_count / num_episodes
        }
        
        print(f"Horizon {horizon}: "
              f"Mean Reward = {results[horizon]['mean_reward']:.2f} ± {results[horizon]['std_reward']:.2f}, "
              f"Success Rate = {results[horizon]['success_rate']:.2%}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    horizons_list = list(results.keys())
    mean_rewards = [results[h]['mean_reward'] for h in horizons_list]
    std_rewards = [results[h]['std_reward'] for h in horizons_list]
    success_rates = [results[h]['success_rate'] for h in horizons_list]
    
    axes[0].errorbar(horizons_list, mean_rewards, yerr=std_rewards, 
                    marker='o', capsize=5, capthick=2)
    axes[0].set_xlabel('Planning Horizon')
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Performance vs Planning Horizon')
    axes[0].grid(True)
    
    axes[1].plot(horizons_list, success_rates, marker='o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Planning Horizon')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_title('Success Rate vs Planning Horizon')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results


def analyze_goal_relabeling_impact(tdm, env, num_samples=1000):
    """
    Analyze the impact of goal relabeling on learning
    
    Args:
        tdm: trained TDM model
        env: environment
        num_samples: number of samples to analyze
    """
    from env_wrapper import GoalSampler
    
    # Sample transitions
    states = []
    actions = []
    next_states = []
    
    for _ in range(num_samples):
            obs, info = env.reset()
            goal = env.get_goal()
            action = env.action_space.sample()
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            
            if done:
                obs, info = env.reset()
    
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    
    # Compute Q-values for different goals
    goal_sampler = GoalSampler(env.task_type, env.unwrapped.spec.id)
    
    tau = 10
    q_values = []
    
    for i in range(min(100, num_samples)):
        state = states[i]
        action = actions[i]
        next_state = next_states[i]
        
        # Sample random goal
        goal = goal_sampler.sample()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(tdm.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(tdm.device)
        goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(tdm.device)
        tau_tensor = torch.FloatTensor([tau]).unsqueeze(0).to(tdm.device)
        
        with torch.no_grad():
            q_value = tdm.critic.compute_q_value(
                state_tensor, action_tensor, goal_tensor, tau_tensor
            )
        
        q_values.append(q_value.cpu().numpy()[0])
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(q_values, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Q-value')
    plt.ylabel('Frequency')
    plt.title('Q-value Distribution with Goal Relabeling')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Mean Q-value: {np.mean(q_values):.2f}")
    print(f"Std Q-value: {np.std(q_values):.2f}")


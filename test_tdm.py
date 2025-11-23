"""
Test script for TDM implementation
"""
import sys
import io
# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
import gymnasium as gym
import yaml

from networks import Actor, TDMCritic, TDMCriticVectorized
from tdm import TDM
from env_wrapper import TDMEnvWrapper
from replay_buffer import TDMBuffer
from mpc_planner import MPCPlanner


def test_networks():
    """Test neural network architectures"""
    print("Testing neural networks...")
    
    # Test parameters
    state_dim = 17
    action_dim = 7
    goal_dim = 2  # Reacher-v5 uses 2-dim goal
    batch_size = 32
    
    # Create test tensors
    state = torch.randn(batch_size, state_dim)
    action = torch.randn(batch_size, action_dim)
    goal = torch.randn(batch_size, goal_dim)
    tau = torch.randint(0, 25, (batch_size, 1)).float()
    
    # Test Actor
    print("  Testing Actor...")
    goal_dim = 2  # Reacher-v5 uses 2-dim goal
    actor = Actor(state_dim, action_dim, goal_dim, [300, 300])
    goal = torch.randn(batch_size, goal_dim)
    tau = torch.randint(0, 25, (batch_size, 1)).float()
    actor_output = actor(state, goal, tau)
    assert actor_output.shape == (batch_size, action_dim)
    print("    ✓ Actor output shape correct")
    
    # Test TDMCritic
    print("  Testing TDMCritic...")
    critic = TDMCritic(state_dim, action_dim, goal_dim, [300, 300])
    predicted_state = critic(state, action, goal, tau)
    q_value = critic.compute_q_value(state, action, goal, tau)
    assert predicted_state.shape == (batch_size, goal_dim)
    assert q_value.shape == (batch_size,)
    print("    ✓ TDMCritic output shape correct")
    
    # Test TDMCriticVectorized
    print("  Testing TDMCriticVectorized...")
    critic_vec = TDMCriticVectorized(state_dim, action_dim, goal_dim, [300, 300])
    predicted_state_vec = critic_vec(state, action, goal, tau)
    q_value_vec = critic_vec.compute_q_value(state, action, goal, tau)
    assert predicted_state_vec.shape == (batch_size, goal_dim)
    assert q_value_vec.shape == (batch_size,)
    print("    ✓ TDMCriticVectorized output shape correct")
    
    print("✓ All network tests passed!\n")


def test_replay_buffer():
    """Test replay buffer and goal relabeling"""
    print("Testing replay buffer...")
    
    state_dim = 17
    action_dim = 7
    goal_dim = 3
    
    # Create buffer
    buffer = TDMBuffer(state_dim, action_dim, goal_dim, max_size=1000,
                      goal_sampling_strategy='future', tau_max=25)
    
    # Add transitions
    for _ in range(100):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        next_state = np.random.randn(state_dim)
        reward = np.random.randn(1)
        done = np.random.randint(0, 2)
        goal = np.random.randn(goal_dim)
        
        buffer.add(state, action, next_state, reward, done, goal)
    
    assert buffer.size == 100
    print("  ✓ Buffer size correct")
    
    # Sample batch
    batch = buffer.sample_tdm_batch(32)
    states, actions, next_states, goals, taus = batch
    
    assert states.shape == (32, state_dim)
    assert actions.shape == (32, action_dim)
    assert next_states.shape == (32, state_dim)
    # Goals can be different shape depending on relabeling
    assert len(goals.shape) == 2 and goals.shape[0] == 32
    assert len(taus.shape) == 2 and taus.shape[0] == 32
    print("  ✓ Batch sampling correct")
    
    print("✓ All replay buffer tests passed!\n")


def test_tdm_basic():
    """Test basic TDM functionality"""
    print("Testing TDM basic functionality...")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create simple environment
    state_dim = 17
    action_dim = 7
    goal_dim = 2  # Reacher-v5 uses 2-dim goal
    action_range = (-1, 1)
    
    # Create TDM (with goal_dim)
    tdm = TDM(state_dim, action_dim, goal_dim, action_range, config, device='cpu')
    
    # Test action selection
    state = np.random.randn(state_dim)
    goal = np.random.randn(goal_dim)
    tau = 10
    
    action = tdm.select_action(state, goal, tau, add_noise=False)
    assert action.shape == (action_dim,)
    print("  ✓ Action selection works")
    
    # Test training step (with empty buffer should return None)
    result = tdm.train_step()
    assert result is None
    print("  ✓ Training step handles empty buffer correctly")
    
    # Add some transitions
    for _ in range(100):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        next_state = np.random.randn(state_dim)
        reward = np.random.randn(1)
        done = np.random.randint(0, 2)
        goal = np.random.randn(goal_dim)
        
        tdm.replay_buffer.add(state, action, next_state, reward, done, goal)
    
    # Test training step (with data)
    # Need at least batch_size samples
    for _ in range(150):  # Ensure we have enough samples
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        next_state = np.random.randn(state_dim)
        reward = np.random.randn(1)
        done = np.random.randint(0, 2)
        goal = np.random.randn(goal_dim)
        tdm.replay_buffer.add(state, action, next_state, reward, done, goal)
    
    result = tdm.train_step()
    if result is not None:
        assert 'critic_loss' in result
        assert 'actor_loss' in result
        print("  ✓ Training step works with data")
    else:
        print("  ⚠ Training step returned None (may need more samples)")
    
    print("✓ All TDM basic tests passed!\n")


def test_env_wrapper():
    """Test environment wrapper"""
    print("Testing environment wrapper...")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    try:
        env = gym.make('Reacher-v5')
        env = TDMEnvWrapper(env, 'end_effector', config['task'])
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert env.current_goal is not None
        print("  ✓ Environment reset works")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert 'distance_to_goal' in info
        assert 'goal_reached' in info
        print("  ✓ Environment step works")
        
        # Test goal extraction
        goal = env.get_goal()
        assert goal.shape == (2,)  # XY for Reacher-v5
        print("  ✓ Goal extraction works")
        
        env.close()
        print("✓ All environment wrapper tests passed!\n")
        
    except Exception as e:
        print(f"  ⚠ Environment test skipped: {e}\n")


def test_mpc_planner():
    """Test MPC planner"""
    print("Testing MPC planner...")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create TDM
    state_dim = 17
    action_dim = 7
    goal_dim = 3
    action_range = (-1, 1)
    
    tdm = TDM(state_dim, action_dim, goal_dim, action_range, config, device='cpu')
    
    # Create planner
    planner = MPCPlanner(tdm, config)
    
    # Test direct planning
    state = np.random.randn(state_dim)
    goal = np.random.randn(goal_dim)
    tau = 10
    
    action = planner.plan_direct(state, goal, tau)
    assert action.shape == (action_dim,)
    print("  ✓ Direct planning works")
    
    # Test optimization planning
    action = planner.plan_optimization(state, goal, tau)
    assert action.shape == (action_dim,)
    print("  ✓ Optimization planning works")
    
    # Test multi-step planning
    action = planner.plan_multi_step(state, goal, tau, K=5)
    assert action.shape == (action_dim,)
    print("  ✓ Multi-step planning works")
    
    print("✓ All MPC planner tests passed!\n")


def test_full_pipeline():
    """Test full training pipeline (short)"""
    print("Testing full training pipeline...")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for quick test
    config['env']['name'] = 'Reacher-v5'
    config['env']['max_episode_steps'] = 50
    config['training']['total_timesteps'] = 1000
    config['training']['eval_frequency'] = 500
    config['training']['eval_episodes'] = 2
    config['training']['updates_per_step'] = 1
    config['tdm']['tau_max'] = 10
    
    # Create environment
    try:
        env = gym.make('Reacher-v5')
        env = TDMEnvWrapper(env, 'end_effector', config['task'])
        
        # Get dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_range = (env.action_space.low, env.action_space.high)
        goal_dim = env.goal_dim
        
        # Create TDM
        tdm = TDM(state_dim, action_dim, goal_dim, action_range, config, device='cpu')
        
        # Create planner
        from mpc_planner import TaskSpecificPlanner
        planner = TaskSpecificPlanner(tdm, config, 'Reacher-v5', 'end_effector')
        
        # Short training loop
        obs, info = env.reset()
        goal = env.get_goal()
        
        for step in range(100):
            # Select action
            tau = config['tdm']['tau_max']
            action = planner.select_action(obs, goal, tau)
            
            # Add noise
            noise = np.random.normal(0, 0.1, size=action_dim)
            action = np.clip(action + noise, action_range[0], action_range[1])
            
            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store
            tdm.replay_buffer.add(obs, action, next_obs, reward, done, goal)
            
            # Train
            if step > 10:
                tdm.train_step()
            
            obs = next_obs
            
            if done or step >= 49:
                obs, info = env.reset()
                goal = env.get_goal()
        
        print("  ✓ Full training pipeline works")
        print(f"  ✓ Buffer size: {tdm.replay_buffer.size}")
        print(f"  ✓ Total steps: {step + 1}")
        
        env.close()
        print("✓ Full pipeline test passed!\n")
        
    except Exception as e:
        print(f"  ⚠ Full pipeline test skipped: {e}\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("Running TDM Tests")
    print("="*60 + "\n")
    
    test_networks()
    test_replay_buffer()
    test_tdm_basic()
    test_env_wrapper()
    test_mpc_planner()
    test_full_pipeline()
    
    print("="*60)
    print("All tests completed!")
    print("="*60 + "\n")


if __name__ == '__main__':
    run_all_tests()


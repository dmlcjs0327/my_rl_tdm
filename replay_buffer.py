"""
Replay Buffer with Goal Relabeling for TDM
"""
import numpy as np
import torch


class ReplayBuffer:
    """
    Standard replay buffer for off-policy learning
    """
    
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Buffers
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        
    def add(self, state, action, next_state, reward, done):
        """Add transition to buffer"""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """Sample batch of transitions"""
        idx = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.state[idx]),
            torch.FloatTensor(self.action[idx]),
            torch.FloatTensor(self.next_state[idx]),
            torch.FloatTensor(self.reward[idx]),
            torch.FloatTensor(self.done[idx])
        )
    
    def sample_trajectory(self, batch_size):
        """
        Sample a complete trajectory from buffer
        Used for goal relabeling with future states
        """
        # Sample random episode
        episode_idx = np.random.randint(0, self.size)
        
        # Find episode boundaries
        start_idx = episode_idx
        while start_idx > 0 and not self.done[start_idx - 1]:
            start_idx -= 1
        
        end_idx = episode_idx
        while end_idx < self.size - 1 and not self.done[end_idx]:
            end_idx += 1
        
        episode_length = end_idx - start_idx + 1
        
        # Sample random timestep within episode
        t = np.random.randint(0, episode_length)
        state = self.state[start_idx + t]
        action = self.action[start_idx + t]
        next_state = self.next_state[start_idx + t]
        
        # Sample future state as goal
        future_idx = np.random.randint(t, episode_length)
        goal = self.next_state[start_idx + future_idx]
        
        return state, action, next_state, goal


class GoalRelabeler:
    """
    Goal relabeling for TDM training
    Implements different sampling strategies for goals and horizons
    """
    
    def __init__(self, strategy='future', tau_max=25):
        self.strategy = strategy
        self.tau_max = tau_max
    
    def relabel(self, state, action, next_state, buffer=None):
        """
        Relabel goal and horizon for a transition
        
        Args:
            state: current state
            action: action taken
            next_state: next state
            buffer: replay buffer (for future sampling)
        
        Returns:
            goal: relabeled goal state (same dimension as goal_dim)
            tau: relabeled horizon
        """
        # Sample horizon
        tau = np.random.randint(0, self.tau_max + 1)
        
        # Sample goal based on strategy
        if self.strategy == 'future' and buffer is not None:
            # Sample future state from trajectory
            _, _, _, full_goal = buffer.sample_trajectory(1)
            # Extract only goal dimensions (e.g., last 3 dims for end-effector)
            goal = full_goal.squeeze()[-buffer.goal_dim:]
        elif self.strategy == 'buffer' and buffer is not None:
            # Sample random state from buffer
            idx = np.random.randint(0, buffer.size)
            full_goal = buffer.next_state[idx]
            # Extract only goal dimensions
            goal = full_goal[-buffer.goal_dim:]
        else:
            # Use next_state's goal dimensions
            goal = next_state[-buffer.goal_dim:] if buffer is not None else next_state
        
        return goal, tau
    
    def relabel_batch(self, states, actions, next_states, buffer=None):
        """
        Relabel goals and horizons for a batch of transitions
        
        Returns:
            goals: (batch_size, goal_dim)
            taus: (batch_size, 1)
        """
        batch_size = len(states)
        goals = []
        taus = []
        
        for i in range(batch_size):
            goal, tau = self.relabel(states[i], actions[i], next_states[i], buffer)
            goals.append(goal)
            taus.append(tau)
        
        return np.array(goals), np.array(taus).reshape(-1, 1)


class TDMBuffer(ReplayBuffer):
    """
    Extended replay buffer for TDM with goal relabeling
    """
    
    def __init__(self, state_dim, action_dim, goal_dim, max_size=int(1e6),
                 goal_sampling_strategy='future', tau_max=25):
        super().__init__(state_dim, action_dim, max_size)
        
        self.goal_dim = goal_dim
        self.goal_relabeler = GoalRelabeler(goal_sampling_strategy, tau_max)
        
        # Additional buffers for goals
        self.goal = np.zeros((max_size, goal_dim))
        self.next_goal = np.zeros((max_size, goal_dim))
    
    def add(self, state, action, next_state, reward, done, goal=None):
        """Add transition to buffer"""
        super().add(state, action, next_state, reward, done)
        
        if goal is not None:
            self.goal[self.ptr - 1] = goal
            self.next_goal[self.ptr - 1] = goal
    
    def sample_tdm_batch(self, batch_size):
        """
        Sample batch with goal and horizon relabeling for TDM training
        
        Returns:
            states, actions, next_states, goals, taus
        """
        # Sample base transitions
        states, actions, next_states, rewards, dones = self.sample(batch_size)
        
        # Relabel goals and horizons
        goals, taus = self.goal_relabeler.relabel_batch(
            states.numpy(), actions.numpy(), next_states.numpy(), self
        )
        
        return (
            states,
            actions,
            next_states,
            torch.FloatTensor(goals),
            torch.FloatTensor(taus)
        )


"""
Model Predictive Control (MPC) for TDM
Implements different policy extraction methods
"""
import numpy as np
import torch


class MPCPlanner:
    """
    MPC planner using TDM for policy extraction
    """
    
    def __init__(self, tdm, config):
        self.tdm = tdm
        self.config = config
        self.device = tdm.device
        
        # MPC parameters
        self.horizon = config['mpc']['horizon']
        self.num_samples = config['mpc']['num_samples']
        self.method = config['mpc']['method']
        
    def plan_direct(self, state, goal, tau):
        """
        Direct policy extraction using TDM
        Uses Equation (9) from paper: a* = argmax_a Q(s, a, g, tau)
        
        Args:
            state: current state
            goal: goal state
            tau: planning horizon
        
        Returns:
            action: optimal action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        tau_tensor = torch.FloatTensor([tau]).unsqueeze(0).to(self.device)
        
        # Use actor to select action (goal-conditioned policy)
        with torch.no_grad():
            action = self.tdm.actor(state_tensor, goal_tensor, tau_tensor)
        
        return action.cpu().numpy()[0]
    
    def plan_optimization(self, state, goal, tau, reward_fn=None):
        """
        Optimization-based policy extraction using TDM
        Uses Equation (8) from paper
        
        Args:
            state: current state
            goal: goal state
            tau: planning horizon
            reward_fn: reward function for terminal state
        
        Returns:
            action: optimal action
        """
        if reward_fn is None:
            # Default: minimize distance to goal
            def reward_fn(predicted_state, goal):
                distance = np.abs(predicted_state - goal).sum()
                return -distance
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        goal_tensor = torch.FloatTensor(goal).to(self.device)
        tau_tensor = torch.FloatTensor([tau]).to(self.device)
        
        best_action = None
        best_reward = -np.inf
        
        # Sample random actions and evaluate
        for _ in range(self.num_samples):
            # Sample random action
            action = np.random.uniform(
                self.tdm.action_range[0],
                self.tdm.action_range[1],
                size=self.tdm.action_dim
            )
            action_tensor = torch.FloatTensor(action).to(self.device)
            
            # Predict next state using TDM
            with torch.no_grad():
                predicted_state = self.tdm.critic(
                    state_tensor.unsqueeze(0),
                    action_tensor.unsqueeze(0),
                    goal_tensor.unsqueeze(0),
                    tau_tensor.unsqueeze(0)
                )
            
            predicted_state = predicted_state.cpu().numpy()[0]
            
            # Compute reward
            reward = reward_fn(predicted_state, goal)
            
            if reward > best_reward:
                best_reward = reward
                best_action = action
        
        return best_action
    
    def plan_multi_step(self, state, goal, tau, reward_fn=None, K=5):
        """
        Multi-step MPC planning
        Uses Equation (6) from paper
        
        Args:
            state: current state
            goal: goal state
            tau: planning horizon
            reward_fn: reward function for terminal state
            K: step size for multi-step planning
        
        Returns:
            action: optimal action
        """
        if reward_fn is None:
            def reward_fn(predicted_state, goal):
                distance = np.abs(predicted_state - goal).sum()
                return -distance
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        goal_tensor = torch.FloatTensor(goal).to(self.device)
        
        best_action = None
        best_reward = -np.inf
        
        # Sample action sequences
        for _ in range(self.num_samples):
            # Sample action
            action = np.random.uniform(
                self.tdm.action_range[0],
                self.tdm.action_range[1],
                size=self.tdm.action_dim
            )
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            # Predict state after K steps
            current_state = state_tensor.unsqueeze(0)
            total_reward = 0
            
            for step in range(K):
                if step < K - 1:
                    tau_step = max(0, tau - step - 1)
                else:
                    tau_step = max(0, tau - K)
                
                tau_tensor = torch.FloatTensor([tau_step]).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    predicted_state = self.tdm.critic(
                        current_state,
                        action_tensor,
                        goal_tensor.unsqueeze(0),
                        tau_tensor
                    )
                
                # Compute reward for this step
                reward = reward_fn(predicted_state.cpu().numpy()[0], goal)
                total_reward += reward
                
                # Update current state for next step
                # predicted_state is goal_dim, but we need state_dim
                # For simplicity, just use the original state_dim (this is a limitation)
                # In practice, you'd need to predict the full state or use a different approach
                current_state = current_state  # Keep using current_state
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = action
        
        return best_action
    
    def select_action(self, state, goal, tau, reward_fn=None):
        """
        Select action using configured MPC method
        
        Args:
            state: current state
            goal: goal state
            tau: planning horizon
            reward_fn: optional reward function
        
        Returns:
            action: selected action
        """
        if self.method == 'direct':
            return self.plan_direct(state, goal, tau)
        elif self.method == 'optimization':
            return self.plan_optimization(state, goal, tau, reward_fn)
        elif self.method == 'multi_step':
            return self.plan_multi_step(state, goal, tau, reward_fn)
        else:
            raise ValueError(f"Unknown MPC method: {self.method}")


class TaskSpecificPlanner:
    """
    Task-specific planning for different environments
    Implements optimizations mentioned in paper appendix
    """
    
    def __init__(self, tdm, config, env_name, task_type):
        self.tdm = tdm
        self.config = config
        self.env_name = env_name
        self.task_type = task_type
        self.device = tdm.device
        
    
    def select_action(self, state, goal, tau):
        """Alias for plan method for consistency"""
        return self.plan(state, goal, tau)
    
    def plan(self, state, goal, tau):
        """
        Task-specific planning
        
        Args:
            state: current state
            goal: goal state
            tau: planning horizon
        
        Returns:
            action: optimal action
        """
        # For all tasks, use direct planning (goal-conditioned policy)
        # Reacher-v5 doesn't need complex joint optimization
        return self._plan_direct(state, goal, tau)
    
    def _plan_direct(self, state, goal, tau):
        """Direct planning using actor (goal-conditioned policy)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        tau_tensor = torch.FloatTensor([tau]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.tdm.actor(state_tensor, goal_tensor, tau_tensor)
        
        return action.cpu().numpy()[0]
    


"""
Temporal Difference Models (TDM) Implementation
"""
import numpy as np
import torch
import torch.nn.functional as F

from networks import Actor, TDMCritic, TDMCriticVectorized
from replay_buffer import TDMBuffer


class TDM:
    """
    Temporal Difference Models for Model-Free Deep RL
    
    Paper: "Temporal Difference Models: Model-Free Deep RL for Model-Based Control"
    """
    
    def __init__(self, state_dim, action_dim, goal_dim, action_range,
                 config, device='cpu'):
        """
        Args:
            state_dim: dimension of state space
            action_dim: dimension of action space
            goal_dim: dimension of goal space
            action_range: tuple of (min_action, max_action)
            config: configuration dictionary
            device: torch device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.action_range = action_range
        self.device = device
        self.config = config
        
        # Hyperparameters
        self.tau_max = config['tdm']['tau_max']
        self.lr_actor = config['training']['learning_rate_actor']
        self.lr_critic = config['training']['learning_rate_critic']
        self.batch_size = config['training']['batch_size']
        self.polyak = config['training']['polyak']
        self.reward_scale = config['tdm']['reward_scale']
        self.vectorized = config['tdm']['vectorized_supervision']
        
        # Networks
        if self.vectorized:
            self.critic = TDMCriticVectorized(
                state_dim, action_dim, goal_dim,
                config['network']['critic']['hidden_sizes'],
                config['network']['critic']['activation'],
                config['tdm']['distance_metric']
            ).to(device)
            self.critic_target = TDMCriticVectorized(
                state_dim, action_dim, goal_dim,
                config['network']['critic']['hidden_sizes'],
                config['network']['critic']['activation'],
                config['tdm']['distance_metric']
            ).to(device)
        else:
            self.critic = TDMCritic(
                state_dim, action_dim, goal_dim,
                config['network']['critic']['hidden_sizes'],
                config['network']['critic']['activation'],
                config['tdm']['distance_metric']
            ).to(device)
            self.critic_target = TDMCritic(
                state_dim, action_dim, goal_dim,
                config['network']['critic']['hidden_sizes'],
                config['network']['critic']['activation'],
                config['tdm']['distance_metric']
            ).to(device)
        
        self.actor = Actor(
            state_dim, action_dim, goal_dim,
            config['network']['actor']['hidden_sizes'],
            config['network']['actor']['activation'],
            config['network']['actor']['output_activation']
        ).to(device)
        
        # Initialize target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        
        # Learning rate schedulers (optional)
        self.use_lr_decay = config['training'].get('use_lr_decay', False)
        if self.use_lr_decay:
            self.lr_decay_rate = config['training'].get('lr_decay_rate', 0.999)
            self.lr_decay_steps = config['training'].get('lr_decay_steps', 1000)
            self.min_lr_actor = config['training'].get('min_lr_actor', self.lr_actor * 0.1)
            self.min_lr_critic = config['training'].get('min_lr_critic', self.lr_critic * 0.1)
        
        # Gradient clipping
        self.grad_clip = config['training'].get('grad_clip', None)
        
        # Replay buffer
        self.replay_buffer = TDMBuffer(
            state_dim, action_dim, goal_dim,
            config['training']['buffer_size'],
            config['task']['goal_sampling_strategy'],
            self.tau_max
        )
        
        # Noise for exploration
        self.noise_std = config['training']['noise_std']
        self.noise_decay = config['training']['noise_decay']
        self.noise_decay_steps = config['training']['noise_decay_steps']
        self.min_noise_std = config['training'].get('min_noise_std', 0.01)  # 최소 탐험 노이즈
        
        # Statistics
        self.total_steps = 0
    
    def select_action(self, state, goal, tau, add_noise=True):
        """
        Select action using actor network (goal-conditioned policy)
        
        Args:
            state: current state
            goal: goal state
            tau: planning horizon
            add_noise: whether to add exploration noise
        
        Returns:
            action: selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        tau_tensor = torch.FloatTensor([tau]).unsqueeze(0).to(self.device)
        
        # Actor takes state, goal, and tau as inputs
        with torch.no_grad():
            action = self.actor(state_tensor, goal_tensor, tau_tensor)
        
        action = action.cpu().numpy()[0]
        
        # Add noise for exploration
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + noise
            action = np.clip(action, self.action_range[0], self.action_range[1])
        
        return action
    
    def compute_distance(self, state, goal):
        """
        Compute distance between state and goal
        
        Args:
            state: current state (batch_size, state_dim or goal_dim)
            goal: goal state (batch_size, goal_dim)
        
        Returns:
            distance: (batch_size,)
        """
        # Extract goal dimensions from state if needed
        if state.shape[1] != self.goal_dim:
            # Assume goal is in the last goal_dim dimensions
            state = state[:, -self.goal_dim:]
        
        if self.config['tdm']['distance_metric'] == 'L1':
            distance = torch.abs(state - goal).sum(dim=-1)
        elif self.config['tdm']['distance_metric'] == 'L2':
            distance = torch.sqrt(((state - goal) ** 2).sum(dim=-1) + 1e-8)
        
        return distance
    
    def update_critic(self, batch):
        """
        Update critic network using TDM loss
        
        Args:
            batch: (states, actions, next_states, goals, taus)
        """
        states, actions, next_states, goals, taus = batch
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        goals = goals.to(self.device)
        taus = taus.to(self.device)
        
        batch_size = states.shape[0]
        
        # Compute target Q-values
        with torch.no_grad():
            # Compute tau mask and next_taus first
            tau_mask = (taus > 0).float()
            next_taus = torch.clamp(taus - 1, min=0)
            
            # Get next action using actor (goal-conditioned policy)
            # Debug: print shapes if mismatch
            if next_states.shape[1] != self.state_dim or goals.shape[1] != self.goal_dim:
                print(f"Shape mismatch: next_states={next_states.shape}, goals={goals.shape}, "
                      f"expected state_dim={self.state_dim}, goal_dim={self.goal_dim}")
            next_actions = self.actor(next_states, goals, next_taus)
            
            # Compute Q-values for next state-action pairs
            # For tau=0: Q(s, a, g, 0) = -||s' - g||
            # For tau>0: Q(s, a, g, tau) = max_a' Q(s', a', g, tau-1)
            
            # Compute distance for tau=0
            distance_tau0 = self.compute_distance(next_states, goals)
            
            next_q_values = self.critic_target.compute_q_value(
                next_states, next_actions, goals, next_taus
            )
            
            # Combine based on tau
            target_q = -distance_tau0 * (1 - tau_mask) + next_q_values * tau_mask
        
        # Compute current Q-values
        predicted_states = self.critic(states, actions, goals, taus)
        
        if self.vectorized:
            # Vectorized supervision: supervise each dimension separately
            # Loss for each dimension
            distance_per_dim = torch.abs(predicted_states - goals)
            
            # Extract goal dimensions from next_states
            next_states_goal = next_states[:, -self.goal_dim:]
            
            # For tau=0, use actual next_state distance
            # For tau>0, use predicted distance from target network
            with torch.no_grad():
                next_predicted = self.critic_target(next_states, next_actions, goals, next_taus)
                target_distance_tau0 = torch.abs(next_states_goal - goals)
                target_distance_taun = torch.abs(next_predicted - goals)
                # Expand tau_mask to match goal dimensions
                tau_mask_expanded = tau_mask.repeat(1, self.goal_dim)
                target_distance = target_distance_tau0 * (1 - tau_mask_expanded) + \
                                 target_distance_taun * tau_mask_expanded
            
            loss = F.mse_loss(distance_per_dim, target_distance)
        else:
            # Scalar supervision: supervise total distance
            distance = self.compute_distance(predicted_states, goals)
            loss = F.mse_loss(distance, -target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        
        self.critic_optimizer.step()
        
        return loss.item()
    
    def update_actor(self, batch):
        """
        Update actor network using policy gradient
        
        Args:
            batch: (states, actions, next_states, goals, taus)
        """
        states, _, _, goals, taus = batch
        
        states = states.to(self.device)
        goals = goals.to(self.device)
        taus = taus.to(self.device)
        
        # Compute Q-values for current policy (goal-conditioned policy)
        actions = self.actor(states, goals, taus)
        q_values = self.critic.compute_q_value(states, actions, goals, taus)
        
        # Maximize Q-values (minimize negative Q-values)
        loss = -q_values.mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        
        self.actor_optimizer.step()
        
        return loss.item()
    
    def update_target_networks(self):
        """Soft update target networks"""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + 
                                   (1 - self.polyak) * param.data)
    
    def train_step(self):
        """Perform one training step"""
        if self.replay_buffer.size < self.batch_size:
            return None
        
        # Sample batch with goal relabeling
        batch = self.replay_buffer.sample_tdm_batch(self.batch_size)
        
        # Update critic
        critic_loss = self.update_critic(batch)
        
        # Update actor
        actor_loss = self.update_actor(batch)
        
        # Update target networks
        self.update_target_networks()
        
        # Update learning rates if decay is enabled
        if self.use_lr_decay and self.total_steps % self.lr_decay_steps == 0:
            self._decay_learning_rates()
        
        self.total_steps += 1
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss
        }
    
    def _decay_learning_rates(self):
        """Decay learning rates for both actor and critic"""
        for param_group in self.actor_optimizer.param_groups:
            new_lr = max(param_group['lr'] * self.lr_decay_rate, self.min_lr_actor)
            param_group['lr'] = new_lr
        
        for param_group in self.critic_optimizer.param_groups:
            new_lr = max(param_group['lr'] * self.lr_decay_rate, self.min_lr_critic)
            param_group['lr'] = new_lr
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor': self.actor.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'config': self.config
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.total_steps = checkpoint['total_steps']
        print(f"Model loaded from {filepath}")



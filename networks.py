"""
Neural network architectures for TDM (Temporal Difference Models)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor network for TDM (Goal-conditioned policy)"""
    
    def __init__(self, state_dim, action_dim, goal_dim, hidden_sizes=[300, 300], 
                 activation='relu', output_activation='tanh'):
        super(Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        # Input: state + goal + tau (as integer)
        input_dim = state_dim + goal_dim + 1
        
        # Build layers
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            input_dim = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        if output_activation == 'tanh':
            layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state, goal, tau):
        """
        Args:
            state: (batch_size, state_dim)
            goal: (batch_size, goal_dim)
            tau: (batch_size, 1) - planning horizon
        Returns:
            action: (batch_size, action_dim)
        """
        # Concatenate inputs
        x = torch.cat([state, goal, tau], dim=-1)
        return self.network(x)


class TDMCritic(nn.Module):
    """
    TDM Critic network
    Q(s, a, sg, tau) = -||f(s, a, sg, tau) - sg||
    """
    
    def __init__(self, state_dim, action_dim, goal_dim, hidden_sizes=[300, 300],
                 activation='relu', distance_metric='L1'):
        super(TDMCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.distance_metric = distance_metric
        
        # Input: state + action + goal + tau (as integer)
        input_dim = state_dim + action_dim + goal_dim + 1
        
        # Build layers
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            input_dim = hidden_size
        
        # Output layer predicts state (has same dimension as goal)
        layers.append(nn.Linear(input_dim, goal_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state, action, goal, tau):
        """
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            goal: (batch_size, goal_dim)
            tau: (batch_size, 1) - planning horizon
        Returns:
            predicted_state: (batch_size, goal_dim)
        """
        # Concatenate inputs
        x = torch.cat([state, action, goal, tau], dim=-1)
        predicted_state = self.network(x)
        return predicted_state
    
    def compute_q_value(self, state, action, goal, tau):
        """
        Compute Q(s, a, sg, tau) = -||f(s, a, sg, tau) - sg||
        """
        predicted_state = self.forward(state, action, goal, tau)
        
        if self.distance_metric == 'L1':
            distance = torch.abs(predicted_state - goal).sum(dim=-1)
        elif self.distance_metric == 'L2':
            distance = torch.sqrt(((predicted_state - goal) ** 2).sum(dim=-1) + 1e-8)
        
        return -distance


class TDMCriticVectorized(nn.Module):
    """
    TDM Critic with vectorized supervision
    Each dimension is supervised separately for better learning
    """
    
    def __init__(self, state_dim, action_dim, goal_dim, hidden_sizes=[300, 300],
                 activation='relu', distance_metric='L1'):
        super(TDMCriticVectorized, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.distance_metric = distance_metric
        
        # Input: state + action + goal + tau (as integer)
        input_dim = state_dim + action_dim + goal_dim + 1
        
        # Build layers
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            input_dim = hidden_size
        
        # Output layer predicts state (has same dimension as goal)
        layers.append(nn.Linear(input_dim, goal_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state, action, goal, tau):
        """
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            goal: (batch_size, goal_dim)
            tau: (batch_size, 1) - planning horizon
        Returns:
            predicted_state: (batch_size, goal_dim)
        """
        # Concatenate inputs
        x = torch.cat([state, action, goal, tau], dim=-1)
        predicted_state = self.network(x)
        return predicted_state
    
    def compute_q_value(self, state, action, goal, tau):
        """
        Compute Q(s, a, sg, tau) = -||f(s, a, sg, tau) - sg||
        """
        predicted_state = self.forward(state, action, goal, tau)
        
        if self.distance_metric == 'L1':
            distance = torch.abs(predicted_state - goal).sum(dim=-1)
        elif self.distance_metric == 'L2':
            distance = torch.sqrt(((predicted_state - goal) ** 2).sum(dim=-1) + 1e-8)
        
        return -distance


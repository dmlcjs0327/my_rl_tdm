"""
Environment wrapper for TDM
Handles goal extraction and task-specific configurations
"""
import numpy as np
import gymnasium as gym


class GoalExtractor:
    """Extract goal from environment state based on task type"""
    
    def __init__(self, task_type, env_name):
        self.task_type = task_type
        self.env_name = env_name
        
        # Define goal extraction functions for different tasks
        if 'Reacher' in env_name or 'Pusher' in env_name:
            # Reacher-v5: observation is 10-dim, use last 2 dims (end-effector relative position)
            # Pusher: use hand and puck XY
            self.goal_dim = 2 if 'Reacher' in env_name else 4
            self.extract_fn = self._extract_end_effector
        elif 'Cheetah' in env_name:
            self.goal_dim = 1
            self.extract_fn = self._extract_velocity
        elif 'Ant' in env_name:
            if task_type == 'position':
                self.goal_dim = 2
                self.extract_fn = self._extract_position
            elif task_type == 'position_velocity':
                self.goal_dim = 4
                self.extract_fn = self._extract_position_velocity
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        else:
            raise ValueError(f"Unknown environment: {env_name}")
    
    def _extract_end_effector(self, state, info):
        """Extract end effector position"""
        if 'Reacher' in self.env_name:
            # For Reacher-v5, extract last 2 dims (end-effector relative position)
            return state[-2:]
        elif 'Pusher' in self.env_name:
            # For Pusher, extract XY of hand and puck
            return state[8:12]  # hand_x, hand_y, puck_x, puck_y
    
    def _extract_velocity(self, state, info):
        """Extract velocity"""
        # For Cheetah, velocity is at index 9
        return np.array([state[9]])
    
    def _extract_position(self, state, info):
        """Extract XY position"""
        # For Ant, position is at indices 2, 3 (x, y)
        return state[2:4]
    
    def _extract_position_velocity(self, state, info):
        """Extract XY position and velocity"""
        # For Ant, position is at indices 2, 3 (x, y)
        # Velocity is at indices 8, 9 (vx, vy)
        return np.concatenate([state[2:4], state[8:10]])
    
    def extract(self, state, info=None):
        """Extract goal from state"""
        return self.extract_fn(state, info)
    
    def __call__(self, state, info=None):
        return self.extract(state, info)


class TDMEnvWrapper(gym.Wrapper):
    """
    Wrapper for Gymnasium environments to work with TDM
    """
    
    def __init__(self, env, task_type='end_effector', config=None):
        super().__init__(env)
        self.config = config or {}
        
        # Goal extractor
        self.goal_extractor = GoalExtractor(task_type, env.unwrapped.spec.id)
        self.goal_dim = self.goal_extractor.goal_dim
        
        # Task-specific settings
        self.task_type = task_type
        self.position_weight = config.get('position_weight', 0.1) if config else 0.1
        self.velocity_weight = config.get('velocity_weight', 0.9) if config else 0.9
        
        # Current goal
        self.current_goal = None
        self.initial_state = None
        
    def reset(self, **kwargs):
        """Reset environment and sample new goal"""
        obs, info = self.env.reset(**kwargs)
        
        # Extract goal from initial state
        self.current_goal = self.goal_extractor(obs, info)
        self.initial_state = obs
        
        return obs, info
    
    def compute_reward(self, state, goal):
        """
        Compute reward based on distance to goal
        
        Args:
            state: current state
            goal: target goal
        
        Returns:
            reward: scalar reward
        """
        current_goal = self.goal_extractor(state)
        
        if self.task_type == 'position_velocity':
            # Weighted distance for position and velocity
            position_dist = np.abs(current_goal[:2] - goal[:2]).sum()
            velocity_dist = np.abs(current_goal[2:] - goal[2:]).sum()
            distance = self.position_weight * position_dist + \
                      self.velocity_weight * velocity_dist
        else:
            # L1 distance
            distance = np.abs(current_goal - goal).sum()
        
        # Negative distance as reward
        return -distance
    
    def step(self, action):
        """Step environment"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Compute TDM reward based on distance to goal
        tdm_reward = self.compute_reward(obs, self.current_goal)
        
        # Check if goal is reached
        current_goal = self.goal_extractor(obs)
        distance = np.abs(current_goal - self.current_goal).sum()
        
        # For tasks with position+velocity, check both
        if self.task_type == 'position_velocity':
            position_reached = np.abs(current_goal[:2] - self.current_goal[:2]).sum() < 0.5
            velocity_reached = np.abs(current_goal[2:] - self.current_goal[2:]).sum() < 0.1
            goal_reached = position_reached and velocity_reached
        else:
            goal_reached = distance < 0.1
        
        # Update terminated if goal is reached
        terminated = terminated or goal_reached
        
        info['distance_to_goal'] = distance
        info['goal_reached'] = goal_reached
        
        return obs, tdm_reward, terminated, truncated, info
    
    def get_goal(self):
        """Get current goal"""
        return self.current_goal


class GoalSampler:
    """Sample goals for different task types"""
    
    def __init__(self, task_type, env_name):
        self.task_type = task_type
        self.env_name = env_name
        
        # Define goal sampling ranges
        if 'Reacher' in env_name:
            # Reachable positions in 2D plane (Reacher-v5)
            self.goal_range = {
                'x': (-0.3, 0.3),
                'y': (-0.3, 0.3)
            }
        elif 'Pusher' in env_name:
            # Table positions for hand and puck
            self.goal_range = {
                'hand_x': (-0.3, 0.3),
                'hand_y': (-0.3, 0.3),
                'puck_x': (-0.3, 0.3),
                'puck_y': (-0.3, 0.3)
            }
        elif 'Cheetah' in env_name:
            # Velocity range
            self.goal_range = {'velocity': (-6.0, 6.0)}
        elif 'Ant' in env_name:
            if task_type == 'position':
                # Position in 2D plane
                self.goal_range = {
                    'x': (-3.0, 3.0),
                    'y': (-3.0, 3.0)
                }
            elif task_type == 'position_velocity':
                # Position and velocity
                self.goal_range = {
                    'x': (-0.5, 0.5),
                    'y': (-0.5, 0.5),
                    'vx': (-0.25, 0.25),
                    'vy': (-0.25, 0.25)
                }
    
    def sample(self):
        """Sample a random goal"""
        if 'Reacher' in self.env_name:
            return np.array([
                np.random.uniform(self.goal_range['x'][0], self.goal_range['x'][1]),
                np.random.uniform(self.goal_range['y'][0], self.goal_range['y'][1])
            ])
        elif 'Pusher' in self.env_name:
            return np.array([
                np.random.uniform(self.goal_range['hand_x'][0], self.goal_range['hand_x'][1]),
                np.random.uniform(self.goal_range['hand_y'][0], self.goal_range['hand_y'][1]),
                np.random.uniform(self.goal_range['puck_x'][0], self.goal_range['puck_x'][1]),
                np.random.uniform(self.goal_range['puck_y'][0], self.goal_range['puck_y'][1])
            ])
        elif 'Cheetah' in self.env_name:
            return np.array([
                np.random.uniform(self.goal_range['velocity'][0], self.goal_range['velocity'][1])
            ])
        elif 'Ant' in self.env_name:
            if self.task_type == 'position':
                return np.array([
                    np.random.uniform(self.goal_range['x'][0], self.goal_range['x'][1]),
                    np.random.uniform(self.goal_range['y'][0], self.goal_range['y'][1])
                ])
            elif self.task_type == 'position_velocity':
                return np.array([
                    np.random.uniform(self.goal_range['x'][0], self.goal_range['x'][1]),
                    np.random.uniform(self.goal_range['y'][0], self.goal_range['y'][1]),
                    np.random.uniform(self.goal_range['vx'][0], self.goal_range['vx'][1]),
                    np.random.uniform(self.goal_range['vy'][0], self.goal_range['vy'][1])
                ])


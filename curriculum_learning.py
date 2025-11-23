"""
Curriculum Learning 구현
쉬운 목표부터 점진적으로 어려운 목표로 학습
"""
import numpy as np
from typing import Tuple


class CurriculumLearning:
    """
    Curriculum Learning for TDM
    점진적으로 어려운 목표를 제시하여 학습 효율 향상
    """
    
    def __init__(self, 
                 goal_sampler,
                 initial_difficulty: float = 0.1,
                 final_difficulty: float = 1.0,
                 curriculum_type: str = 'distance',
                 schedule: str = 'linear'):
        """
        Args:
            goal_sampler: GoalSampler 인스턴스
            initial_difficulty: 초기 난이도 (0.0 ~ 1.0)
            final_difficulty: 최종 난이도 (0.0 ~ 1.0)
            curriculum_type: 'distance' (거리 기반) 또는 'complexity' (복잡도 기반)
            schedule: 'linear', 'exponential', 'step'
        """
        self.goal_sampler = goal_sampler
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.curriculum_type = curriculum_type
        self.schedule = schedule
        
        # 난이도 스케줄링
        self.current_difficulty = initial_difficulty
        self.progress = 0.0  # 0.0 ~ 1.0
    
    def update_progress(self, progress: float):
        """
        학습 진행도 업데이트 (0.0 ~ 1.0)
        
        Args:
            progress: 현재 학습 진행도
        """
        self.progress = max(0.0, min(1.0, progress))
        
        # 난이도 업데이트
        if self.schedule == 'linear':
            self.current_difficulty = self.initial_difficulty + \
                (self.final_difficulty - self.initial_difficulty) * self.progress
        elif self.schedule == 'exponential':
            # 지수적 증가
            alpha = np.exp(5 * (self.progress - 1))
            self.current_difficulty = self.final_difficulty - \
                (self.final_difficulty - self.initial_difficulty) * alpha
        elif self.schedule == 'step':
            # 단계적 증가
            num_steps = 5
            step = int(self.progress * num_steps)
            step_progress = (self.progress * num_steps) - step
            step_difficulty = self.initial_difficulty + \
                (self.final_difficulty - self.initial_difficulty) * (step / num_steps)
            next_step_difficulty = self.initial_difficulty + \
                (self.final_difficulty - self.initial_difficulty) * ((step + 1) / num_steps)
            self.current_difficulty = step_difficulty + \
                (next_step_difficulty - step_difficulty) * step_progress
    
    def sample_goal(self, initial_state=None) -> np.ndarray:
        """
        현재 난이도에 맞는 목표 샘플링
        
        Args:
            initial_state: 초기 상태 (거리 기반 curriculum에 사용)
        
        Returns:
            샘플링된 목표
        """
        if self.curriculum_type == 'distance':
            return self._sample_distance_based(initial_state)
        elif self.curriculum_type == 'complexity':
            return self._sample_complexity_based()
        else:
            # 기본: 전체 범위에서 샘플링
            return self.goal_sampler.sample()
    
    def _sample_distance_based(self, initial_state=None) -> np.ndarray:
        """거리 기반 curriculum: 가까운 목표부터 시작"""
        # 기본 목표 샘플링
        base_goal = self.goal_sampler.sample()
        
        if initial_state is None:
            return base_goal
        
        # 초기 상태에서의 목표 추출
        from env_wrapper import GoalExtractor
        task_type = getattr(self.goal_sampler, 'task_type', None)
        env_name = getattr(self.goal_sampler, 'env_name', None)
        
        # GoalSampler에서 정보 추출
        if task_type is None:
            # GoalSampler의 task_type이 없으면 env_name에서 추론
            if 'Reacher' in env_name or 'Pusher' in env_name:
                task_type = 'end_effector'
            elif 'Cheetah' in env_name:
                task_type = 'velocity'
            elif 'Ant' in env_name:
                task_type = 'position'
            else:
                task_type = 'end_effector'
        
        if env_name is None:
            env_name = 'Reacher-v5'  # 기본값
        
        goal_extractor = GoalExtractor(task_type, env_name)
        initial_goal = goal_extractor.extract(initial_state)
        
        # 난이도에 따라 목표 거리 조정
        # current_difficulty가 낮으면 가까운 목표, 높으면 먼 목표
        goal = initial_goal + (base_goal - initial_goal) * self.current_difficulty
        
        return goal
    
    def _sample_complexity_based(self) -> np.ndarray:
        """복잡도 기반 curriculum: 단순한 목표부터 시작"""
        # 현재는 거리 기반과 동일하게 구현
        # 필요시 환경별로 복잡도 정의 가능
        return self.goal_sampler.sample()
    
    def get_difficulty(self) -> float:
        """현재 난이도 반환"""
        return self.current_difficulty


class WarmUpPeriod:
    """
    Warm-up Period 구현
    초기 학습 단계에서 탐험을 강화
    """
    
    def __init__(self,
                 warmup_steps: int = 10000,
                 initial_noise_std: float = 0.5,
                 final_noise_std: float = 0.2,
                 initial_lr_multiplier: float = 0.1,
                 final_lr_multiplier: float = 1.0):
        """
        Args:
            warmup_steps: Warm-up 기간 (스텝 수)
            initial_noise_std: 초기 노이즈 표준편차
            final_noise_std: 최종 노이즈 표준편차
            initial_lr_multiplier: 초기 학습률 배수
            final_lr_multiplier: 최종 학습률 배수
        """
        self.warmup_steps = warmup_steps
        self.initial_noise_std = initial_noise_std
        self.final_noise_std = final_noise_std
        self.initial_lr_multiplier = initial_lr_multiplier
        self.final_lr_multiplier = final_lr_multiplier
        self.current_step = 0
    
    def update_step(self, step: int):
        """현재 스텝 업데이트"""
        self.current_step = step
    
    def get_noise_std(self, base_noise_std: float) -> float:
        """
        Warm-up 기간에 따른 노이즈 표준편차 반환
        
        Args:
            base_noise_std: 기본 노이즈 표준편차
        
        Returns:
            조정된 노이즈 표준편차
        """
        if self.current_step >= self.warmup_steps:
            return base_noise_std
        
        # 선형 보간
        progress = self.current_step / self.warmup_steps
        warmup_noise = self.initial_noise_std + \
            (self.final_noise_std - self.initial_noise_std) * progress
        
        # 기본 노이즈와 warm-up 노이즈 중 큰 값 사용
        return max(warmup_noise, base_noise_std)
    
    def get_lr_multiplier(self) -> float:
        """
        Warm-up 기간에 따른 학습률 배수 반환
        
        Returns:
            학습률 배수 (0.0 ~ 1.0)
        """
        if self.current_step >= self.warmup_steps:
            return self.final_lr_multiplier
        
        # 선형 보간
        progress = self.current_step / self.warmup_steps
        return self.initial_lr_multiplier + \
            (self.final_lr_multiplier - self.initial_lr_multiplier) * progress
    
    def is_warmup(self) -> bool:
        """Warm-up 기간인지 확인"""
        return self.current_step < self.warmup_steps


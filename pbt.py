"""
Population-based Training (PBT) for TDM
학습 곡선을 보고 동적으로 하이퍼파라미터를 조정
"""
import os
import yaml
import numpy as np
import copy
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class PBTMember:
    """PBT 개체 (하나의 실험)"""
    member_id: int
    hyperparameters: Dict[str, Any]
    best_distance: float = float('inf')
    current_distance: float = float('inf')
    training_steps: int = 0
    log_dir: str = ""
    model_path: str = ""
    performance_history: List[float] = None
    ready_for_perturb: bool = False
    
    # 상세 평가 데이터 저장
    evaluation_data: List[Dict[str, Any]] = None  # 평가 결과 리스트
    hyperparameter_history: List[Dict[str, Any]] = None  # 하이퍼파라미터 변화 이력
    
    # 조기 종료 관련
    early_stopped: bool = False
    early_stop_reason: str = None
    
    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []
        if self.evaluation_data is None:
            self.evaluation_data = []
        if self.hyperparameter_history is None:
            self.hyperparameter_history = [{
                'generation': 0,
                'training_steps': 0,
                'hyperparameters': copy.deepcopy(self.hyperparameters)
            }]


class PopulationBasedTraining:
    """
    Population-based Training (PBT)
    
    여러 실험을 동시에 실행하고, 학습 곡선을 보고 하이퍼파라미터를 동적으로 조정
    """
    
    def __init__(self,
                 base_config: Dict[str, Any],
                 population_size: int = 8,
                 exploit_frequency: int = 10000,  # 몇 스텝마다 exploit/explore
                 exploit_threshold: float = 0.25,  # 하위 25%를 교체
                 explore_perturbation: float = 0.2,  # 하이퍼파라미터 변형 정도 (20%)
                 hyperparameter_ranges: Optional[Dict[str, List[Any]]] = None):
        """
        Args:
            base_config: 기본 설정
            population_size: 개체 수 (동시에 실행할 실험 수)
            exploit_frequency: 몇 스텝마다 exploit/explore 수행
            exploit_threshold: 하위 몇 %를 교체할지 (0.25 = 하위 25%)
            explore_perturbation: 하이퍼파라미터 변형 정도
            hyperparameter_ranges: 조정 가능한 하이퍼파라미터 범위
        """
        self.base_config = base_config
        self.population_size = population_size
        self.exploit_frequency = exploit_frequency
        self.exploit_threshold = exploit_threshold
        self.explore_perturbation = explore_perturbation
        
        # 조정 가능한 하이퍼파라미터 범위
        if hyperparameter_ranges is None:
            self.hyperparameter_ranges = {
                'learning_rate_actor': [0.00005, 0.0003],
                'learning_rate_critic': [0.0005, 0.003],
                'tau_max': [15, 30],
                'batch_size': [64, 256],
                'updates_per_step': [5, 20],
                'noise_std': [0.1, 0.3],
                'reward_scale': [0.1, 10.0],
            }
        else:
            self.hyperparameter_ranges = hyperparameter_ranges
        
        # 개체 초기화
        self.population: List[PBTMember] = []
        self._initialize_population()
        
        # 통계
        self.generation = 0
        self.total_exploits = 0
        self.total_explores = 0
    
    def _initialize_population(self):
        """개체군 초기화 (랜덤 하이퍼파라미터로)"""
        self.population = []
        for i in range(self.population_size):
            hyperparams = self._sample_random_hyperparameters()
            member = PBTMember(
                member_id=i,
                hyperparameters=hyperparams,
                log_dir=f"pbt_member_{i:02d}"
            )
            self.population.append(member)
    
    def _sample_random_hyperparameters(self) -> Dict[str, Any]:
        """랜덤 하이퍼파라미터 샘플링"""
        hyperparams = {}
        for key, value_range in self.hyperparameter_ranges.items():
            if isinstance(value_range, list) and len(value_range) == 2:
                # 연속값: 범위 내에서 랜덤 샘플링
                if isinstance(value_range[0], (int, float)):
                    if isinstance(value_range[0], int):
                        # 정수 범위
                        hyperparams[key] = random.randint(value_range[0], value_range[1])
                    else:
                        # 실수 범위
                        hyperparams[key] = random.uniform(value_range[0], value_range[1])
                else:
                    # 이산값: 리스트에서 선택
                    hyperparams[key] = random.choice(value_range)
            elif isinstance(value_range, list):
                # 이산값 리스트
                hyperparams[key] = random.choice(value_range)
            else:
                hyperparams[key] = value_range
        
        return hyperparams
    
    def _perturb_hyperparameter(self, value: Any, key: str, perturbation_factor: float = None) -> Any:
        """
        하이퍼파라미터에 변형 가하기 (explore)
        
        Args:
            value: 현재 값
            key: 하이퍼파라미터 이름
        
        Returns:
            변형된 값
        """
        if key not in self.hyperparameter_ranges:
            return value
        
        # 변형 정도 결정
        if perturbation_factor is None:
            perturbation_factor = 1.0
        actual_perturbation = self.explore_perturbation * perturbation_factor
        
        value_range = self.hyperparameter_ranges[key]
        
        if isinstance(value, (int, float)) and isinstance(value_range, list) and len(value_range) == 2:
            # 연속값: 현재 값 주변에서 변형
            if isinstance(value, int):
                # 정수: perturbation_factor에 따라 변형 범위 조절
                perturbation = int(value * actual_perturbation)
                min_val = max(value_range[0], value - perturbation)
                max_val = min(value_range[1], value + perturbation)
                return random.randint(min_val, max_val)
            else:
                # 실수: perturbation_factor에 따라 변형 범위 조절
                perturbation = value * actual_perturbation
                min_val = max(value_range[0], value - perturbation)
                max_val = min(value_range[1], value + perturbation)
                return random.uniform(min_val, max_val)
        elif isinstance(value_range, list):
            # 이산값: 리스트 내에서 다른 값 선택
            if len(value_range) > 1:
                other_values = [v for v in value_range if v != value]
                if other_values:
                    return random.choice(other_values)
            return value
        
        return value
    
    def get_successful_members(self, min_training_steps: int = 1000) -> List['PBTMember']:
        """
        성공한 개체들 반환 (조기 종료되지 않고 충분히 학습된 개체들)
        
        Args:
            min_training_steps: 최소 학습 스텝 수
        
        Returns:
            성공한 개체 리스트
        """
        return [m for m in self.population 
                if not m.early_stopped and m.training_steps >= min_training_steps]
    
    def update_member(self, member_id: int, 
                     best_distance: float,
                     current_distance: float,
                     training_steps: int,
                     log_dir: str = "",
                     model_path: str = "",
                     eval_data: Dict[str, Any] = None):
        """
        개체의 성능 업데이트
        
        Args:
            member_id: 개체 ID
            best_distance: 최고 거리
            current_distance: 현재 거리
            training_steps: 학습 스텝 수
            log_dir: 로그 디렉토리
            model_path: 모델 경로
            eval_data: 평가 데이터 (선택사항)
        """
        member = self.population[member_id]
        member.best_distance = best_distance
        member.current_distance = current_distance
        member.training_steps = training_steps
        if log_dir:
            member.log_dir = log_dir
        if model_path:
            member.model_path = model_path
        
        # 평가 데이터 저장
        if eval_data is not None:
            eval_record = {
                'training_steps': training_steps,
                'generation': self.generation,
                'best_distance': best_distance,
                'current_distance': current_distance,
                'hyperparameters': copy.deepcopy(member.hyperparameters),
                **eval_data  # mean_distance, success_rate, mean_length 등
            }
            member.evaluation_data.append(eval_record)
        
        # 성능 히스토리 업데이트
        member.performance_history.append(current_distance)
        if len(member.performance_history) > 10:
            member.performance_history.pop(0)
        
        # Exploit/Explore 준비 여부 확인
        member.ready_for_perturb = (training_steps % self.exploit_frequency == 0 and 
                                   training_steps > 0)
    
    def should_exploit_explore(self) -> bool:
        """Exploit/Explore를 수행할 시점인지 확인"""
        return any(member.ready_for_perturb for member in self.population)
    
    def exploit_and_explore(self) -> Dict[int, Dict[str, Any]]:
        """
        Exploit and Explore 수행
        
        Returns:
            {member_id: new_hyperparameters} 딕셔너리
        """
        self.generation += 1
        
        # 성능 기준으로 정렬 (거리가 작을수록 좋음)
        sorted_population = sorted(self.population, key=lambda x: x.best_distance)
        
        # 하위 개체 수 계산
        num_exploit = max(1, int(self.population_size * self.exploit_threshold))
        
        # 하위 개체들의 하이퍼파라미터 업데이트
        updates = {}
        for i in range(num_exploit):
            poor_member = sorted_population[i]
            
            # 상위 개체 중 랜덤 선택 (exploit)
            top_members = sorted_population[num_exploit:]
            if top_members:
                good_member = random.choice(top_members)
                
                # 하이퍼파라미터 복사 (exploit)
                new_hyperparams = copy.deepcopy(good_member.hyperparameters)
                
                # 약간의 변형 가하기 (explore)
                for key in new_hyperparams:
                    if key in self.hyperparameter_ranges:
                        new_hyperparams[key] = self._perturb_hyperparameter(
                            new_hyperparams[key], key
                        )
                
                # 업데이트
                poor_member.hyperparameters = new_hyperparams
                poor_member.ready_for_perturb = False
                updates[poor_member.member_id] = new_hyperparams
                
                # 하이퍼파라미터 변화 이력 저장
                poor_member.hyperparameter_history.append({
                    'generation': self.generation,
                    'training_steps': poor_member.training_steps,
                    'hyperparameters': copy.deepcopy(new_hyperparams),
                    'exploited_from': good_member.member_id,
                    'old_best_distance': poor_member.best_distance
                })
                
                self.total_exploits += 1
                self.total_explores += 1
        
        return updates
    
    def get_best_member(self) -> PBTMember:
        """최고 성능 개체 반환"""
        return min(self.population, key=lambda x: x.best_distance)
    
    def get_member_config(self, member_id: int) -> Dict[str, Any]:
        """개체의 설정 반환"""
        member = self.population[member_id]
        config = copy.deepcopy(self.base_config)
        
        # 하이퍼파라미터 적용
        from hyperparameter_grid import create_config_from_hyperparameters
        config = create_config_from_hyperparameters(config, member.hyperparameters)
        
        return config
    
    def save_state(self, filepath: str):
        """PBT 상태 저장"""
        state = {
            'generation': self.generation,
            'total_exploits': self.total_exploits,
            'total_explores': self.total_explores,
            'population': [
                {
                    'member_id': m.member_id,
                    'hyperparameters': m.hyperparameters,
                    'best_distance': m.best_distance,
                    'current_distance': m.current_distance,
                    'training_steps': m.training_steps,
                    'log_dir': m.log_dir,
                    'model_path': m.model_path,
                }
                for m in self.population
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
    
    def save_evaluation_data(self, filepath: str):
        """평가 데이터 저장"""
        data = {
            'generation': self.generation,
            'total_exploits': self.total_exploits,
            'total_explores': self.total_explores,
            'members': [
                {
                    'member_id': m.member_id,
                    'evaluation_data': m.evaluation_data,
                    'hyperparameter_history': m.hyperparameter_history,
                    'best_distance': m.best_distance,
                    'current_distance': m.current_distance,
                    'training_steps': m.training_steps,
                }
                for m in self.population
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_state(self, filepath: str):
        """PBT 상태 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.generation = state['generation']
        self.total_exploits = state['total_exploits']
        self.total_explores = state['total_explores']
        
        for member_data in state['population']:
            member = self.population[member_data['member_id']]
            member.hyperparameters = member_data['hyperparameters']
            member.best_distance = member_data['best_distance']
            member.current_distance = member_data['current_distance']
            member.training_steps = member_data['training_steps']
            member.log_dir = member_data['log_dir']
            member.model_path = member_data['model_path']


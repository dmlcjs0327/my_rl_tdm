"""
하이퍼파라미터 그리드 정의 및 기본값 설정
TDM 논문 기반 일반적인 하이퍼파라미터 범위
"""
import itertools
import yaml
from typing import Dict, List, Any


# Grid Search를 위한 하이퍼파라미터 그리드
HYPERPARAMETER_GRID = {
    'learning_rate_actor': [0.00005, 0.0001, 0.0003],
    'learning_rate_critic': [0.0005, 0.001, 0.003],
    'tau_max': [15, 20, 25],
    'batch_size': [64, 128, 256],
    'updates_per_step': [5, 10, 20],
    'polyak': [0.995, 0.999],
    'noise_std': [0.1, 0.2, 0.3],
    'grad_clip': [0.5, 1.0, 2.0],
    'reward_scale': [0.1, 1.0, 10.0],
}

# 환경별 특화 그리드 (환경에 따라 조정 가능)
ENV_SPECIFIC_GRIDS = {
    'Reacher-v5': {
        'tau_max': [15, 20, 25],
        'learning_rate_actor': [0.0001, 0.0003],
        'learning_rate_critic': [0.001, 0.003],
    },
    'Pusher-v5': {
        'tau_max': [20, 25, 30],
        'learning_rate_actor': [0.00005, 0.0001],
        'learning_rate_critic': [0.0005, 0.001],
    },
    'HalfCheetah-v5': {
        'tau_max': [15, 20, 25],
        'learning_rate_actor': [0.0001, 0.0003],
        'learning_rate_critic': [0.001, 0.003],
    },
    'Ant-v5': {
        'tau_max': [20, 25, 30],
        'learning_rate_actor': [0.00005, 0.0001],
        'learning_rate_critic': [0.0005, 0.001],
    },
}


def get_hyperparameter_grid(env_name: str = None, grid_type: str = 'full') -> Dict[str, List[Any]]:
    """
    환경에 맞는 하이퍼파라미터 그리드 반환
    
    Args:
        env_name: 환경 이름 (None이면 기본 그리드)
        grid_type: 'full' (전체), 'reduced' (축소), 'minimal' (최소)
    
    Returns:
        하이퍼파라미터 그리드 딕셔너리
    """
    if grid_type == 'minimal':
        # 최소 그리드 (빠른 테스트용)
        grid = {
            'learning_rate_actor': [0.0001],
            'learning_rate_critic': [0.001],
            'tau_max': [20, 25],
            'batch_size': [128],
            'updates_per_step': [10],
            'polyak': [0.999],
            'noise_std': [0.2],
            'grad_clip': [1.0],
            'reward_scale': [1.0],
        }
    elif grid_type == 'reduced':
        # 축소 그리드 (중간 크기)
        grid = {
            'learning_rate_actor': [0.00005, 0.0001, 0.0003],
            'learning_rate_critic': [0.0005, 0.001, 0.003],
            'tau_max': [20, 25],
            'batch_size': [128, 256],
            'updates_per_step': [10, 20],
            'polyak': [0.999],
            'noise_std': [0.2],
            'grad_clip': [1.0],
            'reward_scale': [1.0, 10.0],
        }
    else:
        # 전체 그리드
        grid = HYPERPARAMETER_GRID.copy()
    
    # 환경별 특화 그리드 적용
    if env_name and env_name in ENV_SPECIFIC_GRIDS:
        env_grid = ENV_SPECIFIC_GRIDS[env_name]
        for key, values in env_grid.items():
            if key in grid:
                grid[key] = values
    
    return grid


def generate_hyperparameter_combinations(env_name: str = None, grid_type: str = 'full') -> List[Dict[str, Any]]:
    """
    하이퍼파라미터 조합 생성
    
    Args:
        env_name: 환경 이름
        grid_type: 그리드 타입
    
    Returns:
        하이퍼파라미터 조합 리스트
    """
    grid = get_hyperparameter_grid(env_name, grid_type)
    
    # 그리드의 키와 값 리스트 추출
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    
    # 모든 조합 생성
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def create_config_from_hyperparameters(base_config: Dict, hyperparams: Dict) -> Dict:
    """
    기본 설정에 하이퍼파라미터 적용하여 새 설정 생성
    
    Args:
        base_config: 기본 설정 딕셔너리
        hyperparams: 적용할 하이퍼파라미터 딕셔너리
    
    Returns:
        새로운 설정 딕셔너리
    """
    import copy
    config = copy.deepcopy(base_config)
    
    # 하이퍼파라미터를 설정에 적용
    if 'learning_rate_actor' in hyperparams:
        config['training']['learning_rate_actor'] = hyperparams['learning_rate_actor']
    if 'learning_rate_critic' in hyperparams:
        config['training']['learning_rate_critic'] = hyperparams['learning_rate_critic']
    if 'tau_max' in hyperparams:
        config['tdm']['tau_max'] = hyperparams['tau_max']
    if 'batch_size' in hyperparams:
        config['training']['batch_size'] = hyperparams['batch_size']
    if 'updates_per_step' in hyperparams:
        config['training']['updates_per_step'] = hyperparams['updates_per_step']
    if 'polyak' in hyperparams:
        config['training']['polyak'] = hyperparams['polyak']
    if 'noise_std' in hyperparams:
        config['training']['noise_std'] = hyperparams['noise_std']
    if 'grad_clip' in hyperparams:
        # grad_clip이 None일 수 있으므로 체크
        if hyperparams['grad_clip'] is not None:
            config['training']['grad_clip'] = hyperparams['grad_clip']
        else:
            config['training']['grad_clip'] = None
    if 'reward_scale' in hyperparams:
        config['tdm']['reward_scale'] = hyperparams['reward_scale']
    
    return config




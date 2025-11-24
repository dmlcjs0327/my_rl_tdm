"""
하이퍼파라미터 설정 유틸리티
PBT에서 하이퍼파라미터를 설정에 적용하기 위한 함수
"""
import copy
from typing import Dict, Any


def create_config_from_hyperparameters(base_config: Dict, hyperparams: Dict) -> Dict:
    """
    기본 설정에 하이퍼파라미터 적용하여 새 설정 생성
    PBT에서 사용됨
    
    Args:
        base_config: 기본 설정 딕셔너리
        hyperparams: 적용할 하이퍼파라미터 딕셔너리
    
    Returns:
        새로운 설정 딕셔너리
    """
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
        if hyperparams['grad_clip'] is not None:
            config['training']['grad_clip'] = hyperparams['grad_clip']
        else:
            config['training']['grad_clip'] = None
    if 'reward_scale' in hyperparams:
        config['tdm']['reward_scale'] = hyperparams['reward_scale']
    
    return config

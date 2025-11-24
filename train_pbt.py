"""
Population-based Training (PBT) for TDM
학습 곡선을 보고 동적으로 하이퍼파라미터를 조정하는 학습 스크립트
"""
import os
import yaml
import numpy as np
import torch
import multiprocessing as mp
import sys
import time
import copy
import random
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict, Any

import gymnasium as gym
from tdm import TDM
from env_wrapper import TDMEnvWrapper, GoalSampler
from mpc_planner import TaskSpecificPlanner
from curriculum_learning import CurriculumLearning, WarmUpPeriod
from policy_collapse_detector import PolicyCollapseDetector
from pbt import PopulationBasedTraining, PBTMember
from train_with_curriculum import (
    load_config, set_seed, create_env, evaluate
)
from pbt_monitor import PBTMonitor


def detect_system_resources():
    """
    시스템 자원을 확인하여 적절한 population_size 추천
    
    Returns:
        dict: {'cpu_count', 'gpu_count', 'total_memory_gb', 'available_memory_gb'}
    """
    resources = {
        'cpu_count': mp.cpu_count(),
        'gpu_count': 0,
        'total_memory_gb': 0,
        'available_memory_gb': 0,
    }
    
    # GPU 확인
    if torch.cuda.is_available():
        resources['gpu_count'] = torch.cuda.device_count()
        # GPU 메모리 확인
        if resources['gpu_count'] > 0:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            resources['gpu_memory_gb'] = gpu_memory_gb
    
    # 메모리 확인 (psutil이 있으면 사용)
    try:
        import psutil
        mem = psutil.virtual_memory()
        resources['total_memory_gb'] = mem.total / (1024**3)
        resources['available_memory_gb'] = mem.available / (1024**3)
    except ImportError:
        # psutil이 없으면 메모리 정보는 생략
        pass
    
    return resources


def calculate_optimal_population_size(resources, config=None):
    """
    시스템 자원을 기반으로 최적의 population_size 계산
    
    Args:
        resources: detect_system_resources()의 결과
        config: 설정 딕셔너리 (환경별 메모리 요구사항 확인용)
    
    Returns:
        int: 추천되는 population_size
    """
    cpu_count = resources['cpu_count']
    gpu_count = resources['gpu_count']
    available_memory_gb = resources.get('available_memory_gb', 0)
    
    # 환경별 예상 메모리 사용량 (GB per member)
    env_memory_estimates = {
        'Reacher-v5': 0.5,
        'Pusher-v5': 0.5,
        'HalfCheetah-v5': 1.0,
        'Ant-v5': 1.0,
    }
    
    # 환경 이름 확인
    env_name = None
    if config:
        env_name = config.get('env', {}).get('name', '')
    
    # 환경별 메모리 요구사항
    memory_per_member = env_memory_estimates.get(env_name, 0.8)  # 기본값 0.8GB
    
    recommended_size = 8  # 기본값
    
    # GPU 기반 계산 (우선순위 1)
    if gpu_count > 0:
        # GPU가 있으면 GPU 개수 기반
        # 각 GPU당 1개 개체 (안전 마진 고려)
        recommended_size = gpu_count
    else:
        # CPU 기반 계산
        # CPU 코어 수의 75% 사용 (안전 마진)
        # 최소 1개는 남김
        recommended_size = max(1, int(cpu_count * 0.75))
    
    # 메모리 기반 제한 (psutil이 있고 메모리 정보가 있으면)
    if available_memory_gb > 0:
        # 사용 가능한 메모리의 80% 사용 (안전 마진)
        max_by_memory = int((available_memory_gb * 0.8) / memory_per_member)
        if max_by_memory < recommended_size:
            recommended_size = max_by_memory
    
    # 최소/최대 제한
    min_population = 4
    max_population = 32
    
    recommended_size = max(min_population, min(recommended_size, max_population))
    
    return recommended_size


def train_single_member_chunk(member_data: tuple) -> Dict[str, Any]:
    """
    단일 PBT 개체를 지정된 스텝만큼 학습 (chunk 단위, 멀티프로세싱용)
    
    Args:
        member_data: (member_dict, base_config, log_dir_base, device, steps_to_train, hyperparams)
    """
    member_dict, base_config, log_dir_base, device, steps_to_train, hyperparams = member_data
    
    # PBTMember 객체 재구성 (멀티프로세싱을 위해 딕셔너리로 전달)
    from pbt import PBTMember
    member = PBTMember(
        member_id=member_dict['member_id'],
        hyperparameters=hyperparams,
        best_distance=member_dict.get('best_distance', float('inf')),
        current_distance=member_dict.get('current_distance', float('inf')),
        training_steps=member_dict.get('training_steps', 0),
        log_dir=member_dict.get('log_dir', f"pbt_member_{member_dict['member_id']:02d}"),
        model_path=member_dict.get('model_path', '')
    )
    
    return _train_member_internal(member, base_config, None, log_dir_base, device, max_steps=steps_to_train)


def _train_member_internal(member: PBTMember,
                          base_config: Dict[str, Any],
                          pbt: PopulationBasedTraining,
                          log_dir_base: str,
                          device: str,
                          max_steps: int = None) -> Dict[str, Any]:
    """
    단일 PBT 개체 학습
    
    Args:
        member: PBT 개체
        base_config: 기본 설정
        pbt: PBT 인스턴스
        log_dir_base: 로그 디렉토리 기본 경로
        device: 디바이스
    
    Returns:
        학습 결과
    """
    try:
        # 개체 설정 가져오기
        if pbt is not None:
            config = pbt.get_member_config(member.member_id)
        else:
            # 멀티프로세싱 환경: 하이퍼파라미터 직접 적용
            from hyperparameter_grid import create_config_from_hyperparameters
            config = create_config_from_hyperparameters(base_config, member.hyperparameters)
        
        # 시드 설정 (개체별로 다른 시드)
        set_seed(config['seed'] + member.member_id)
        
        # 환경 생성
        env = create_env(config)
        
        # 차원 정보
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_range = (env.action_space.low, env.action_space.high)
        goal_dim = env.goal_dim
        
        # TDM Agent 생성
        tdm = TDM(state_dim, action_dim, goal_dim, action_range, config, device)
        
        # Planner 생성
        planner = TaskSpecificPlanner(tdm, config, config['env']['name'],
                                     config['task']['locomotion_task_type'])
        
        # Goal Sampler
        goal_sampler = GoalSampler(config['task']['locomotion_task_type'],
                                  config['env']['name'])
        
        # Curriculum Learning
        use_curriculum = config['training'].get('use_curriculum', False)
        curriculum = None
        if use_curriculum:
            curriculum_config = config['training'].get('curriculum', {})
            curriculum = CurriculumLearning(
                goal_sampler,
                initial_difficulty=curriculum_config.get('initial_difficulty', 0.1),
                final_difficulty=curriculum_config.get('final_difficulty', 1.0),
                curriculum_type=curriculum_config.get('type', 'distance'),
                schedule=curriculum_config.get('schedule', 'linear')
            )
        
        # Warm-up Period
        use_warmup = config['training'].get('use_warmup', False)
        warmup = None
        if use_warmup:
            warmup_config = config['training'].get('warmup', {})
            warmup = WarmUpPeriod(
                warmup_steps=warmup_config.get('steps', 10000),
                initial_noise_std=warmup_config.get('initial_noise_std', 0.5),
                final_noise_std=warmup_config.get('final_noise_std', 0.2),
                initial_lr_multiplier=warmup_config.get('initial_lr_multiplier', 0.1),
                final_lr_multiplier=warmup_config.get('final_lr_multiplier', 1.0)
            )
        
        # Policy Collapse Detector
        use_collapse_detection = config['training'].get('detect_policy_collapse', False)
        collapse_detector = None
        if use_collapse_detection:
            collapse_config = config['training'].get('collapse_detection', {})
            collapse_detector = PolicyCollapseDetector(
                window_size=collapse_config.get('window_size', 5),
                collapse_threshold=collapse_config.get('collapse_threshold', 0.3),
                min_evaluations=collapse_config.get('min_evaluations', 3),
                stability_threshold=collapse_config.get('stability_threshold', 0.5)
            )
        
        # 로그 디렉토리
        member_log_dir = os.path.join(log_dir_base, member.log_dir)
        os.makedirs(member_log_dir, exist_ok=True)
        
        # TensorBoard
        if config['logging']['tensorboard']:
            writer = SummaryWriter(member_log_dir)
        
        # 설정 저장
        with open(os.path.join(member_log_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        # 학습 루프
        start_steps = member.training_steps  # 이전 학습 스텝부터 이어서
        total_steps = start_steps
        target_steps = start_steps + max_steps if max_steps else config['training']['total_timesteps']
        episode = 0
        episode_reward = 0
        episode_length = 0
        
        best_eval_distance = member.best_distance if member.best_distance != float('inf') else float('inf')
        best_model_path = os.path.join(member_log_dir, 'model_best.pt')
        patience = config['training'].get('patience', None)
        patience_counter = 0
        
        # 이전 모델 로드 (있는 경우)
        if member.model_path and os.path.exists(member.model_path):
            tdm.load(member.model_path)
            print(f"  [Member {member.member_id:02d}] Loaded previous model from {member.model_path}")
        
        obs, info = env.reset()
        initial_state = obs.copy()
        
        # Curriculum learning: 초기 목표 샘플링
        goal = env.get_goal()
        if curriculum is not None:
            # 진행도 복원
            progress = start_steps / config['training']['total_timesteps']
            curriculum.update_progress(progress)
            curriculum_goal = curriculum.sample_goal(initial_state)
            env.current_goal = curriculum_goal
            goal = curriculum_goal
        
        while total_steps < target_steps and total_steps < config['training']['total_timesteps']:
            # Warm-up 업데이트
            if warmup:
                warmup.update_step(total_steps)
                tdm.noise_std = warmup.get_noise_std(config['training']['noise_std'])
                if warmup.is_warmup():
                    lr_multiplier = warmup.get_lr_multiplier()
                    for param_group in tdm.actor_optimizer.param_groups:
                        param_group['lr'] = config['training']['learning_rate_actor'] * lr_multiplier
                    for param_group in tdm.critic_optimizer.param_groups:
                        param_group['lr'] = config['training']['learning_rate_critic'] * lr_multiplier
            
            # 행동 선택
            tau = config['tdm']['tau_max']
            action = planner.select_action(obs, goal, tau)
            
            # Safety check: ensure action is valid before adding noise
            if np.isnan(action).any() or np.isinf(action).any():
                print(f"  ⚠️  Warning: Invalid action detected at step {total_steps}, using zero action")
                action = np.zeros(action_dim)
            
            noise = np.random.normal(0, tdm.noise_std, size=action_dim)
            # Safety check: ensure noise is valid
            if np.isnan(noise).any() or np.isinf(noise).any():
                noise = np.zeros_like(noise)
            action = np.clip(action + noise, action_range[0], action_range[1])
            
            # Final safety check before environment step
            if np.isnan(action).any() or np.isinf(action).any():
                print(f"  ⚠️  Warning: Invalid action after noise at step {total_steps}, using zero action")
                action = np.zeros(action_dim)
                action = np.clip(action, action_range[0], action_range[1])
            
            # 환경 스텝
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Safety check: ensure observation is valid
            if np.isnan(next_obs).any() or np.isinf(next_obs).any():
                print(f"  ⚠️  Warning: Invalid observation detected at step {total_steps}")
                next_obs = np.nan_to_num(next_obs, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Replay Buffer에 저장
            done = terminated or truncated
            tdm.replay_buffer.add(obs, action, next_obs, reward, done, goal)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # 학습 (Replay Buffer에 충분한 샘플이 있을 때만)
            if total_steps >= config['training']['batch_size'] and tdm.replay_buffer.size >= config['training']['batch_size']:
                for _ in range(config['training']['updates_per_step']):
                    train_info = tdm.train_step()
                    
                    if train_info is not None and config['logging']['tensorboard']:
                        if total_steps % config['logging']['log_frequency'] == 0:
                            # Safety check: ensure loss values are valid before logging
                            critic_loss = train_info.get('critic_loss', 0.0)
                            actor_loss = train_info.get('actor_loss', 0.0)
                            if not (np.isnan(critic_loss) or np.isinf(critic_loss)):
                                writer.add_scalar('train/critic_loss', critic_loss, total_steps)
                            if not (np.isnan(actor_loss) or np.isinf(actor_loss)):
                                writer.add_scalar('train/actor_loss', actor_loss, total_steps)
            
            # 노이즈 감소
            if not warmup or not warmup.is_warmup():
                if total_steps % 1000 == 0:
                    tdm.noise_std = max(tdm.noise_std * config['training']['noise_decay'],
                                       config['training'].get('min_noise_std', 0.01))
            
            # 에피소드 종료
            done = terminated or truncated
            if done or episode_length >= config['env']['max_episode_steps']:
                if config['logging']['tensorboard']:
                    writer.add_scalar('train/episode_reward', episode_reward, episode)
                    writer.add_scalar('train/episode_length', episode_length, episode)
                    writer.add_scalar('train/noise_std', tdm.noise_std, total_steps)
                
                # 환경 리셋
                obs, info = env.reset()
                # Safety check: ensure observation is valid after reset
                if np.isnan(obs).any() or np.isinf(obs).any():
                    print(f"  ⚠️  Warning: Invalid observation after reset at episode {episode}")
                    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                initial_state = obs.copy()
                
                # Curriculum learning: 새로운 목표 샘플링
                goal = env.get_goal()
                if curriculum is not None:
                    progress = total_steps / config['training']['total_timesteps']
                    curriculum.update_progress(progress)
                    curriculum_goal = curriculum.sample_goal(initial_state)
                    env.current_goal = curriculum_goal
                    goal = curriculum_goal
                
                episode_reward = 0
                episode_length = 0
                episode += 1
            
            # 주기적 평가
            if total_steps % config['training']['eval_frequency'] == 0:
                # 상태 업데이트: 평가 중
                monitor = PBTMonitor(log_dir_base)
                monitor.update_member_status(member.member_id, {
                    'training_steps': total_steps,
                    'best_distance': best_eval_distance,
                    'current_distance': current_distance if 'current_distance' in locals() else float('inf'),
                    'state': 'evaluating'
                })
                
                # 평가 시작 출력
                progress_pct = (total_steps / config['training']['total_timesteps'] * 100) if config['training']['total_timesteps'] > 0 else 0
                print(f"[Member {member.member_id:02d}] 평가 중... (Steps: {total_steps}/{config['training']['total_timesteps']}, {progress_pct:.1f}%, Episode: {episode})")
                
                eval_results = evaluate(env, tdm, planner,
                                       config['training']['eval_episodes'],
                                       config)
                
                current_distance = eval_results['mean_distance']
                is_best = current_distance < best_eval_distance
                
                # 평가 결과 출력
                print(f"[Member {member.member_id:02d}] 평가 완료:")
                print(f"  현재 거리: {current_distance:.4f} | 최고 거리: {best_eval_distance:.4f} | 성공률: {eval_results['success_rate']:.2%}")
                
                # 최고 성능 모델 저장
                if is_best:
                    best_eval_distance = current_distance
                    patience_counter = 0
                    tdm.save(best_model_path)
                    print(f"  ✅ 최고 성능 달성! 모델 저장됨 (개선: {best_eval_distance:.4f})")
                else:
                    patience_counter += 1
                    improvement = best_eval_distance - current_distance
                    print(f"  📊 성능: {improvement:+.4f} (patience: {patience_counter}/{patience if patience else 'N/A'})")
                
                # Policy Collapse Detection
                collapsed = False
                if collapse_detector:
                    collapse_info = collapse_detector.update(
                        eval_results['mean_distance'],
                        eval_results['success_rate']
                    )
                    if collapse_info['is_collapsed']:
                        collapsed = True
                        print(f"  ⚠️  Policy Collapse 감지: {collapse_info.get('reason', 'Performance degradation')}")
                
                # 상태 업데이트: 학습 중
                monitor = PBTMonitor(log_dir_base)
                monitor.update_member_status(member.member_id, {
                    'training_steps': total_steps,
                    'best_distance': best_eval_distance,
                    'current_distance': current_distance,
                    'state': 'training',
                    'episode': episode,
                    'early_stopped': False,
                    'early_stop_reason': None
                })
                
                # PBT 업데이트는 메인 프로세스에서만 수행 (멀티프로세싱 환경에서는 생략)
                # 평가 데이터는 반환값에 포함
                
                # Early stopping
                early_stop_reason = None
                if patience is not None and patience_counter >= patience:
                    early_stop_reason = f"Early stopping: No improvement for {patience} evaluations (patience exceeded)"
                    print(f"\n⚠️  [Member {member.member_id:02d}] {early_stop_reason}")
                    print(f"   최종 Steps: {total_steps}, 최고 거리: {best_eval_distance:.4f}")
                    if best_model_path and os.path.exists(best_model_path):
                        tdm.load(best_model_path)
                        print(f"   최고 모델 로드 완료")
                    break
                
                if collapsed:
                    early_stop_reason = f"Policy collapse detected: {collapse_info.get('reason', 'Performance degradation')}"
                    print(f"\n⚠️  [Member {member.member_id:02d}] {early_stop_reason}")
                    print(f"   최종 Steps: {total_steps}, 최고 거리: {best_eval_distance:.4f}")
                    if best_model_path and os.path.exists(best_model_path):
                        tdm.load(best_model_path)
                        print(f"   최고 모델 로드 완료")
                    break
                
                if config['logging']['tensorboard']:
                    writer.add_scalar('eval/mean_distance', eval_results['mean_distance'], total_steps)
                    writer.add_scalar('eval/success_rate', eval_results['success_rate'], total_steps)
                    writer.add_scalar('eval/mean_length', eval_results['mean_length'], total_steps)
                    writer.add_scalar('eval/best_distance', best_eval_distance, total_steps)
            
            # 모델 저장
            if total_steps % config['logging']['save_frequency'] == 0:
                model_path = os.path.join(member_log_dir, f'model_{total_steps}.pt')
                tdm.save(model_path)
                print(f"[Member {member.member_id:02d}] 체크포인트 저장: {total_steps} steps")
        
        # 최종 저장
        final_model_path = os.path.join(member_log_dir, 'model_final.pt')
        if best_model_path and os.path.exists(best_model_path):
            import shutil
            shutil.copy(best_model_path, final_model_path)
        else:
            tdm.save(final_model_path)
        
        env.close()
        if config['logging']['tensorboard']:
            writer.close()
        
        # 조기 종료 여부 확인
        early_stopped = 'early_stop_reason' in locals() and early_stop_reason is not None
        
        return {
            'member_id': member.member_id,
            'best_distance': best_eval_distance,
            'current_distance': current_distance if 'current_distance' in locals() else best_eval_distance,
            'training_steps': total_steps,
            'log_dir': member_log_dir,
            'model_path': final_model_path,
            'collapsed': collapsed if 'collapsed' in locals() else False,
            'early_stopped': early_stopped,
            'early_stop_reason': early_stop_reason if early_stopped else None,
            'eval_data': {
                'mean_distance': eval_results['mean_distance'] if 'eval_results' in locals() else best_eval_distance,
                'std_distance': eval_results.get('std_distance', 0.0) if 'eval_results' in locals() else 0.0,
                'success_rate': eval_results.get('success_rate', 0.0) if 'eval_results' in locals() else 0.0,
                'mean_length': eval_results.get('mean_length', 0.0) if 'eval_results' in locals() else 0.0,
                'mean_reward': eval_results.get('mean_reward', 0.0) if 'eval_results' in locals() else 0.0,
            } if 'eval_results' in locals() else None
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # 에러 상태 업데이트
        monitor = PBTMonitor(log_dir_base)
        monitor.update_member_status(member.member_id, {
            'training_steps': member.training_steps,
            'best_distance': member.best_distance,
            'current_distance': member.current_distance,
            'state': 'error',
            'error': str(e)
        })
        
        return {
            'member_id': member.member_id,
            'best_distance': float('inf'),
            'current_distance': float('inf'),
            'training_steps': member.training_steps,
            'log_dir': '',
            'model_path': '',
            'error': str(e),
            'collapsed': False
        }


def train_pbt(config_path: str = 'config.yaml'):
    """
    Population-based Training 메인 함수
    """
    # 설정 로드
    base_config = load_config(config_path)
    
    # PBT 설정
    pbt_config = base_config.get('pbt', {})
    population_size_config = pbt_config.get('population_size', 'auto')  # 기본값: auto
    
    # population_size가 "auto" 또는 null이면 자동 결정
    if population_size_config is None or (isinstance(population_size_config, str) and population_size_config.lower() == 'auto'):
        print(f"\n{'='*60}")
        print("🔍 시스템 자원 확인 중...")
        print(f"{'='*60}")
        
        resources = detect_system_resources()
        print(f"  CPU 코어: {resources['cpu_count']}개")
        if resources['gpu_count'] > 0:
            print(f"  GPU: {resources['gpu_count']}개")
            if 'gpu_memory_gb' in resources:
                print(f"  GPU 메모리: {resources['gpu_memory_gb']:.1f}GB")
        if resources['total_memory_gb'] > 0:
            print(f"  총 메모리: {resources['total_memory_gb']:.1f}GB")
            print(f"  사용 가능 메모리: {resources['available_memory_gb']:.1f}GB")
        
        population_size = calculate_optimal_population_size(resources, base_config)
        print(f"\n✅ 자동 결정된 Population Size: {population_size}개")
        print(f"{'='*60}\n")
    else:
        population_size = population_size_config
    
    exploit_frequency = pbt_config.get('exploit_frequency', 10000)
    exploit_threshold = pbt_config.get('exploit_threshold', 0.25)
    explore_perturbation = pbt_config.get('explore_perturbation', 0.2)
    
    # 하이퍼파라미터 범위
    hyperparameter_ranges = pbt_config.get('hyperparameter_ranges', None)
    
    # PBT 초기화
    pbt = PopulationBasedTraining(
        base_config=base_config,
        population_size=population_size,
        exploit_frequency=exploit_frequency,
        exploit_threshold=exploit_threshold,
        explore_perturbation=explore_perturbation,
        hyperparameter_ranges=hyperparameter_ranges
    )
    
    # 로그 디렉토리
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir_base = os.path.join(base_config['logging']['log_dir'],
                                f"pbt_{base_config['env']['name']}_{timestamp}")
    os.makedirs(log_dir_base, exist_ok=True)
    
    # 디바이스 정보 상세 확인 및 출력
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        device = torch.device('cuda')
        use_gpu = True
    else:
        gpu_count = 0
        gpu_name = None
        device = torch.device('cpu')
        use_gpu = False
    
    print(f"\n{'='*60}")
    print(f"Population-based Training (PBT)")
    print(f"Environment: {base_config['env']['name']}")
    print(f"Device: {device} ({'GPU' if use_gpu else 'CPU'} 사용)")
    if use_gpu:
        print(f"GPU Count: {gpu_count}")
        print(f"GPU Name: {gpu_name}")
    else:
        print(f"⚠️  CUDA 사용 불가 - CPU로 학습합니다")
        print(f"   torch.cuda.is_available() = {cuda_available}")
        # PyTorch 버전 정보 확인
        try:
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                print(f"   PyTorch는 CUDA를 지원하지만, CUDA 드라이버가 인식되지 않습니다")
                print(f"   PyTorch CUDA 버전: {torch.version.cuda}")
            else:
                print(f"   PyTorch가 CPU 전용 버전으로 설치되어 있습니다")
        except:
            pass
        print(f"   PyTorch 버전: {torch.__version__}")
        print(f"   💡 GPU를 사용하려면 CUDA 지원 PyTorch를 설치하세요:")
        print(f"      conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    print(f"Population Size: {population_size}")
    print(f"Exploit Frequency: {exploit_frequency} steps")
    print(f"Log Directory: {log_dir_base}")
    print(f"{'='*60}\n")
    
    # 초기 개체군 하이퍼파라미터 출력
    print("Initial Population Hyperparameters:")
    print(f"{'='*60}")
    for member in pbt.population:
        print(f"Member {member.member_id:02d}:")
        for key, value in member.hyperparameters.items():
            print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # 학습 루프
    max_generations = pbt_config.get('max_generations', 100)
    total_timesteps = base_config['training']['total_timesteps']
    training_chunk = pbt_config.get('training_chunk', 5000)  # 한 번에 학습할 스텝 수
    
    print(f"\nStarting PBT training...")
    print(f"Total timesteps per member: {total_timesteps}")
    print(f"Training chunk: {training_chunk} steps")
    print(f"Exploit frequency: {exploit_frequency} steps\n")
    
    # 병렬 실행 설정
    num_workers = pbt_config.get('num_workers', None)
    if num_workers is None:
        num_workers = min(population_size, max(1, mp.cpu_count() - 1))
    
    use_parallel = pbt_config.get('use_parallel', True) and num_workers > 1
    
    if use_parallel:
        print(f"Using parallel execution with {num_workers} workers")
    else:
        print(f"Using sequential execution")
    
    # 모니터 초기화
    monitor = PBTMonitor(log_dir_base)
    monitor_interval = pbt_config.get('monitor_interval', 5)  # 몇 초마다 상태 출력
    last_monitor_time = time.time()
    
    print(f"\n💡 Tip: 학습 진행 상황은 {monitor_interval}초마다 자동으로 출력됩니다.")
    print(f"   TensorBoard로도 실시간 모니터링 가능: tensorboard --logdir {log_dir_base}\n")
    
    iteration = 0
    while True:
        iteration += 1
        # 모든 개체가 완료되었는지 확인
        if all(m.training_steps >= total_timesteps for m in pbt.population):
            print("\nAll members completed training!")
            break
        
        # 학습할 개체 선택 (조기 종료된 개체는 제외)
        members_to_train = [m for m in pbt.population 
                          if m.training_steps < total_timesteps and not m.early_stopped]
        
        # 조기 종료된 개체 재시작 (새로운 하이퍼파라미터로)
        early_stopped_members = [m for m in pbt.population 
                                if m.early_stopped and m.training_steps < total_timesteps]
        
        if early_stopped_members:
            print(f"\n🔄 Restarting {len(early_stopped_members)} early-stopped members using PBT-style exploit/explore...")
            
            # PBT 원논문 방식: Truncation Selection
            # 성능 기준으로 정렬 (거리가 작을수록 좋음)
            sorted_population = sorted(pbt.population, key=lambda x: x.best_distance)
            
            # 성공한 개체들 (조기 종료되지 않고 충분히 학습된 개체들)
            successful_members = pbt.get_successful_members(min_training_steps=1000)
            
            # 상위 성능 개체들 (exploit 대상) - 하위 20% 제외한 나머지
            exploit_threshold_idx = max(1, int(len(sorted_population) * (1 - pbt.exploit_threshold)))
            top_members = sorted_population[exploit_threshold_idx:] if exploit_threshold_idx < len(sorted_population) else sorted_population
            
            for member in early_stopped_members:
                # 조기 종료 사유에 따라 다른 전략 사용
                stop_reason = member.early_stop_reason or ""
                source_member = None
                
                # PBT 원논문 방식: Exploit (상위 개체 중 선택) + Explore (변형)
                if top_members:
                    # 성능 기반 가중치 선택 (더 좋은 개체가 선택될 확률 높음)
                    # 또는 단순 랜덤 선택 (원논문에서는 랜덤)
                    source_member = random.choice(top_members)
                    new_hyperparams = copy.deepcopy(source_member.hyperparameters)
                    
                    # 조기 종료 사유에 따라 변형 정도 조절
                    if "Policy collapse" in stop_reason or "collapse" in stop_reason.lower():
                        # Policy Collapse: 큰 변형 (2배) - 실패한 영역을 벗어나기 위해
                        perturbation_factor = 2.0
                        strategy_type = "Policy collapse recovery"
                    elif "Early stopping" in stop_reason or "patience" in stop_reason.lower():
                        # Early Stopping: 작은 변형 (1배) - 약간의 탐색만
                        perturbation_factor = 1.0
                        strategy_type = "Early stopping recovery"
                    else:
                        # 알 수 없는 사유: 중간 변형 (1.5배)
                        perturbation_factor = 1.5
                        strategy_type = "Unknown reason recovery"
                    
                    # Explore: 변형 적용
                    for key in new_hyperparams:
                        if key in pbt.hyperparameter_ranges:
                            new_hyperparams[key] = pbt._perturb_hyperparameter(
                                new_hyperparams[key], key, perturbation_factor=perturbation_factor
                            )
                    
                    strategy = f"{strategy_type}: Exploit from member {source_member.member_id:02d} (perturbation={perturbation_factor}x)"
                else:
                    # 상위 개체가 없으면 (모두 조기 종료된 경우) 완전히 새로운 랜덤
                    new_hyperparams = pbt._sample_random_hyperparameters()
                    strategy = "Complete random restart (no successful members available)"
                
                # 개체 재시작
                member.hyperparameters = new_hyperparams
                member.early_stopped = False
                member.early_stop_reason = None
                member.training_steps = 0  # 처음부터 재시작
                member.best_distance = float('inf')
                member.current_distance = float('inf')
                member.model_path = ""
                
                print(f"  Member {member.member_id:02d}: {strategy}")
                print(f"    New hyperparams: {new_hyperparams}")
                
                # 하이퍼파라미터 변화 이력 저장
                if member.hyperparameter_history is None:
                    member.hyperparameter_history = []
                member.hyperparameter_history.append({
                    'generation': pbt.generation,
                    'training_steps': 0,
                    'hyperparameters': copy.deepcopy(new_hyperparams),
                    'strategy': strategy,
                    'reason': stop_reason,
                    'exploited_from': source_member.member_id if source_member is not None else None
                })
                
                members_to_train.append(member)
        
        if not members_to_train:
            break
        
        # 병렬 또는 순차 실행
        if use_parallel and len(members_to_train) > 1:
            # 병렬 실행
            training_args = []
            for member in members_to_train:
                steps_to_train = min(training_chunk, total_timesteps - member.training_steps)
                # 멀티프로세싱을 위해 딕셔너리로 직렬화
                member_dict = {
                    'member_id': member.member_id,
                    'best_distance': member.best_distance,
                    'current_distance': member.current_distance,
                    'training_steps': member.training_steps,
                    'log_dir': member.log_dir,
                    'model_path': member.model_path
                }
                training_args.append((
                    member_dict, base_config, log_dir_base, str(device), 
                    steps_to_train, member.hyperparameters
                ))
            
            # 멀티프로세싱으로 병렬 실행
            print(f"\n🔄 Starting parallel training for {len(members_to_train)} members...")
            start_time = time.time()
            
            # 비동기 실행으로 진행 상황 모니터링 가능하게
            with mp.Pool(processes=min(num_workers, len(members_to_train))) as pool:
                # 비동기 실행
                async_results = pool.map_async(train_single_member_chunk, training_args)
                
                # 진행 상황 모니터링 (주기적으로 상태 출력)
                while not async_results.ready():
                    current_time = time.time()
                    if current_time - last_monitor_time >= monitor_interval:
                        monitor.print_status(population_size, total_timesteps)
                        last_monitor_time = current_time
                    time.sleep(1)  # 1초마다 체크
                
                results = async_results.get()
            
            elapsed_time = time.time() - start_time
            print(f"✅ Parallel training completed in {elapsed_time:.1f} seconds\n")
            
            # 결과 업데이트
            print(f"\n📊 병렬 학습 완료 - 결과 요약:")
            for result in results:
                if 'error' not in result:
                    eval_data = result.get('eval_data')
                    member = pbt.population[result['member_id']]
                    progress_pct = (result['training_steps'] / total_timesteps * 100) if total_timesteps > 0 else 0
                    
                    # 조기 종료 정보 업데이트
                    if result.get('early_stopped', False):
                        member.early_stopped = True
                        member.early_stop_reason = result.get('early_stop_reason', 'Unknown')
                        print(f"  ⚠️  Member {result['member_id']:02d}: 조기 종료")
                        print(f"     사유: {member.early_stop_reason}")
                        print(f"     Steps: {result['training_steps']}/{total_timesteps} ({progress_pct:.1f}%)")
                        print(f"     최고 거리: {result['best_distance']:.4f}")
                        print(f"     다음 반복에서 재시작 예정")
                    else:
                        print(f"  ✅ Member {result['member_id']:02d}: 학습 진행 중")
                        print(f"     Steps: {result['training_steps']}/{total_timesteps} ({progress_pct:.1f}%)")
                        print(f"     최고 거리: {result['best_distance']:.4f} | 현재 거리: {result['current_distance']:.4f}")
                        if eval_data:
                            print(f"     성공률: {eval_data.get('success_rate', 0):.2%}")
                    
                    pbt.update_member(
                        result['member_id'],
                        result['best_distance'],
                        result['current_distance'],
                        result['training_steps'],
                        result['log_dir'],
                        result['model_path'],
                        eval_data=eval_data
                    )
                else:
                    print(f"  ❌ Member {result['member_id']:02d}: 오류 발생")
                    print(f"     오류: {result['error']}")
            print()  # 빈 줄
        else:
            # 순차 실행 (단일 개체 또는 병렬 비활성화)
            for member in members_to_train:
                steps_to_train = min(training_chunk, total_timesteps - member.training_steps)
                
                print(f"\nTraining Member {member.member_id:02d} for {steps_to_train} steps...")
                print(f"  Current steps: {member.training_steps}/{total_timesteps}")
                print(f"  Hyperparameters: {member.hyperparameters}")
                
                # 개체 학습 (training_chunk만큼) - 순차 실행
                member_dict = {
                    'member_id': member.member_id,
                    'best_distance': member.best_distance,
                    'current_distance': member.current_distance,
                    'training_steps': member.training_steps,
                    'log_dir': member.log_dir,
                    'model_path': member.model_path
                }
                result = train_single_member_chunk((
                    member_dict, base_config, log_dir_base, str(device), 
                    steps_to_train, member.hyperparameters
                ))
                
                # 순차 실행에서는 PBT 업데이트
                if 'error' not in result and pbt is not None:
                    eval_data = result.get('eval_data')
                    member = pbt.population[result['member_id']]
                    
                    # 조기 종료 정보 업데이트
                    if result.get('early_stopped', False):
                        member.early_stopped = True
                        member.early_stop_reason = result.get('early_stop_reason', 'Unknown')
                        print(f"⚠️  Member {result['member_id']:02d}: {member.early_stop_reason}")
                        print(f"   Will be restarted with new hyperparameters in next iteration")
                    
                    pbt.update_member(
                        result['member_id'],
                        result['best_distance'],
                        result['current_distance'],
                        result['training_steps'],
                        result['log_dir'],
                        result['model_path'],
                        eval_data=eval_data
                    )
                    
                    if not result.get('early_stopped', False):
                        print(f"✅ Member {result['member_id']:02d}: Best Distance: {result['best_distance']:.4f}, Steps: {result['training_steps']}")
                else:
                    print(f"❌ Member {result['member_id']:02d}: Error: {result['error']}")
            
            # 상태 출력
            monitor.print_status(population_size, total_timesteps)
        
        # 주기적 상태 출력 (exploit/explore 전)
        current_time = time.time()
        if current_time - last_monitor_time >= monitor_interval:
            monitor.print_status(population_size, total_timesteps)
            last_monitor_time = current_time
        
        # 주기적 상태 출력 (exploit/explore 전)
        current_time = time.time()
        if current_time - last_monitor_time >= monitor_interval:
            monitor.print_status(population_size, total_timesteps)
            last_monitor_time = current_time
        
        # Exploit and Explore 체크
        if pbt.should_exploit_explore():
            print(f"\n{'='*60}")
            print(f"Exploit and Explore (Generation {pbt.generation})")
            print(f"{'='*60}")
            
            updates = pbt.exploit_and_explore()
            
            if updates:
                print(f"Updated {len(updates)} members:")
                for member_id, new_hyperparams in updates.items():
                    member = pbt.population[member_id]
                    print(f"  Member {member_id:02d}:")
                    print(f"    Old hyperparams: {member.hyperparameters}")
                    print(f"    New hyperparams: {new_hyperparams}")
                    
                    # 모델 복사 (exploit)
                    best_member = pbt.get_best_member()
                    if best_member.model_path and os.path.exists(best_member.model_path):
                        # 성능이 좋은 개체의 모델을 나쁜 개체에 복사
                        target_model_path = os.path.join(
                            log_dir_base, member.log_dir, 'model_best.pt'
                        )
                        os.makedirs(os.path.dirname(target_model_path), exist_ok=True)
                        import shutil
                        shutil.copy(best_member.model_path, target_model_path)
                        member.model_path = target_model_path
                        print(f"    Model copied from best member ({best_member.member_id:02d})")
            
            # PBT 상태 및 평가 데이터 저장
            pbt.save_state(os.path.join(log_dir_base, 'pbt_state.json'))
            pbt.save_evaluation_data(os.path.join(log_dir_base, 'pbt_evaluation_data.json'))
        
        # 최고 성능 개체 출력
        best_member = pbt.get_best_member()
        print(f"\n{'='*60}")
        print(f"Current Best Member: {best_member.member_id:02d}")
        print(f"  Best Distance: {best_member.best_distance:.4f}")
        print(f"  Training Steps: {best_member.training_steps}")
        print(f"  Hyperparameters: {best_member.hyperparameters}")
        print(f"{'='*60}")
    
    # 최종 결과
    print(f"\n{'='*60}")
    print("PBT Training Completed!")
    print(f"{'='*60}")
    
    best_member = pbt.get_best_member()
    print(f"\nBest Member: {best_member.member_id:02d}")
    print(f"Best Distance: {best_member.best_distance:.4f}")
    print(f"Best Hyperparameters:")
    for key, value in best_member.hyperparameters.items():
        print(f"  {key}: {value}")
    print(f"Model Path: {best_member.model_path}")
    
    # 최고 하이퍼파라미터 저장
    best_config = pbt.get_member_config(best_member.member_id)
    best_config_path = os.path.join(log_dir_base, 'best_hyperparameters.yaml')
    with open(best_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\nBest hyperparameters saved to: {best_config_path}")
    print(f"Results directory: {log_dir_base}")
    
    # 최종 평가 데이터 저장
    pbt.save_evaluation_data(os.path.join(log_dir_base, 'pbt_evaluation_data.json'))
    pbt.save_state(os.path.join(log_dir_base, 'pbt_state.json'))
    
    print(f"\nEvaluation data saved to: {os.path.join(log_dir_base, 'pbt_evaluation_data.json')}")
    print(f"Visualize results with: python visualize_pbt_results.py --log-dir {log_dir_base}")
    
    return log_dir_base, best_member


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train TDM with Population-based Training (PBT)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Windows에서 multiprocessing을 위한 설정
    if sys.platform == 'win32':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    train_pbt(args.config)


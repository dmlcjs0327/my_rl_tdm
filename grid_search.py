"""
분산적 Grid Search를 통한 하이퍼파라미터 튜닝
"""
import os
import sys
import yaml
import json
import argparse
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Tuple, Any
import traceback

from hyperparameter_grid import (
    generate_hyperparameter_combinations,
    create_config_from_hyperparameters,
    get_total_combinations
)
from train_with_curriculum import train, load_config


def run_single_experiment(args: Tuple[Dict, Dict, str, int, str]) -> Dict[str, Any]:
    """
    단일 실험 실행 (멀티프로세싱용)
    
    Args:
        args: (hyperparams, base_config, env_name, exp_id, log_dir_base)
    
    Returns:
        실험 결과 딕셔너리
    """
    hyperparams, base_config, env_name, exp_id, log_dir_base = args
    
    try:
        # 하이퍼파라미터로 설정 생성
        config = create_config_from_hyperparameters(base_config, hyperparams)
        
        # 환경 이름 설정 (중요: 하이퍼파라미터 적용 후에도 환경 이름 유지)
        config['env']['name'] = env_name
        
        # 실험 ID 설정
        experiment_id = f"{exp_id:04d}"
        
        print(f"\n{'='*60}")
        print(f"Experiment {experiment_id}")
        print(f"Environment: {env_name}")
        print(f"Hyperparameters: {hyperparams}")
        print(f"{'='*60}")
        
        # 학습 실행
        result = train(config, experiment_id=experiment_id, log_dir_base=log_dir_base)
        
        # 결과 반환
        return {
            'experiment_id': experiment_id,
            'hyperparameters': hyperparams,
            'best_distance': result['best_distance'],
            'log_dir': result['log_dir'],
            'model_path': result['model_path'],
            'success': True,
            'error': None
        }
    
    except Exception as e:
        error_msg = f"Error in experiment {exp_id}: {str(e)}"
        print(f"\n{error_msg}")
        traceback.print_exc()
        return {
            'experiment_id': f"{exp_id:04d}",
            'hyperparameters': hyperparams,
            'best_distance': float('inf'),
            'log_dir': None,
            'model_path': None,
            'success': False,
            'error': str(e)
        }


def run_grid_search(env_name: str,
                   base_config_path: str = 'config.yaml',
                   grid_type: str = 'reduced',
                   num_workers: int = None,
                   log_dir_base: str = None,
                   max_experiments: int = None):
    """
    Grid Search 실행
    
    Args:
        env_name: 환경 이름
        base_config_path: 기본 설정 파일 경로
        grid_type: 그리드 타입 ('minimal', 'reduced', 'full')
        num_workers: 병렬 작업자 수 (None이면 CPU 코어 수)
        log_dir_base: 로그 디렉토리 기본 경로
        max_experiments: 최대 실험 수 (None이면 전체)
    """
    # 기본 설정 로드
    base_config = load_config(base_config_path)
    
    # 환경 이름 설정
    base_config['env']['name'] = env_name
    
    # 하이퍼파라미터 조합 생성
    print(f"\nGenerating hyperparameter combinations for {env_name}...")
    combinations = generate_hyperparameter_combinations(env_name, grid_type)
    
    total_combinations = len(combinations)
    print(f"Total combinations: {total_combinations}")
    
    if max_experiments:
        combinations = combinations[:max_experiments]
        print(f"Limited to {len(combinations)} experiments")
    
    # 로그 디렉토리 설정
    if log_dir_base is None:
        log_dir_base = base_config['logging']['log_dir']
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    search_log_dir = os.path.join(log_dir_base, f'grid_search_{env_name}_{timestamp}')
    os.makedirs(search_log_dir, exist_ok=True)
    
    # 작업자 수 설정
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # CPU 코어 수 - 1
    
    print(f"\nStarting Grid Search with {num_workers} workers...")
    print(f"Results will be saved to: {search_log_dir}")
    
    # 실험 인자 준비
    experiment_args = [
        (hyperparams, base_config, env_name, idx, search_log_dir)
        for idx, hyperparams in enumerate(combinations)
    ]
    
    # 병렬 실행
    results = []
    if num_workers == 1:
        # 순차 실행 (디버깅용)
        print("Running sequentially (num_workers=1)...")
        for args in experiment_args:
            result = run_single_experiment(args)
            results.append(result)
    else:
        # 병렬 실행
        print(f"Running in parallel with {num_workers} workers...")
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(run_single_experiment, experiment_args)
    
    # 결과 분석
    print(f"\n{'='*60}")
    print("Grid Search Results")
    print(f"{'='*60}\n")
    
    # 성공한 실험만 필터링
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"Successful experiments: {len(successful_results)}/{len(results)}")
    if failed_results:
        print(f"Failed experiments: {len(failed_results)}")
        for r in failed_results:
            print(f"  - Experiment {r['experiment_id']}: {r['error']}")
    
    # 최고 성능 실험 찾기
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['best_distance'])
        
        print(f"\n{'='*60}")
        print("Best Hyperparameters")
        print(f"{'='*60}")
        print(f"Experiment ID: {best_result['experiment_id']}")
        print(f"Best Distance: {best_result['best_distance']:.4f}")
        print(f"Model Path: {best_result['model_path']}")
        print(f"\nHyperparameters:")
        for key, value in best_result['hyperparameters'].items():
            print(f"  {key}: {value}")
        
        # 최고 하이퍼파라미터 저장
        best_config = create_config_from_hyperparameters(
            base_config, best_result['hyperparameters']
        )
        best_config_path = os.path.join(search_log_dir, 'best_hyperparameters.yaml')
        with open(best_config_path, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)
        
        # 결과 요약 저장
        summary = {
            'env_name': env_name,
            'grid_type': grid_type,
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len(failed_results),
            'best_experiment': {
                'experiment_id': best_result['experiment_id'],
                'best_distance': best_result['best_distance'],
                'hyperparameters': best_result['hyperparameters'],
                'model_path': best_result['model_path'],
                'log_dir': best_result['log_dir']
            },
            'all_results': [
                {
                    'experiment_id': r['experiment_id'],
                    'best_distance': r['best_distance'],
                    'hyperparameters': r['hyperparameters'],
                    'success': r['success']
                }
                for r in results
            ]
        }
        
        summary_path = os.path.join(search_log_dir, 'grid_search_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nBest hyperparameters saved to: {best_config_path}")
        print(f"Summary saved to: {summary_path}")
        
        # 상위 5개 결과 출력
        sorted_results = sorted(successful_results, key=lambda x: x['best_distance'])[:5]
        print(f"\n{'='*60}")
        print("Top 5 Results")
        print(f"{'='*60}")
        for i, result in enumerate(sorted_results, 1):
            print(f"\n{i}. Experiment {result['experiment_id']}")
            print(f"   Distance: {result['best_distance']:.4f}")
            print(f"   Key hyperparameters:")
            for key in ['learning_rate_actor', 'learning_rate_critic', 'tau_max', 
                       'batch_size', 'updates_per_step']:
                if key in result['hyperparameters']:
                    print(f"     {key}: {result['hyperparameters'][key]}")
    else:
        print("\nNo successful experiments!")
    
    print(f"\n{'='*60}")
    print(f"Grid Search completed!")
    print(f"Results directory: {search_log_dir}")
    print(f"{'='*60}\n")
    
    return search_log_dir, best_result if successful_results else None


def main():
    parser = argparse.ArgumentParser(description='Grid Search for TDM Hyperparameters')
    parser.add_argument('--env', type=str, required=True,
                       choices=['Reacher-v5', 'Pusher-v5', 'HalfCheetah-v5', 'Ant-v5'],
                       help='Environment name')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Base configuration file')
    parser.add_argument('--grid-type', type=str, default='reduced',
                       choices=['minimal', 'reduced', 'full'],
                       help='Grid type (minimal: small, reduced: medium, full: large)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU cores - 1)')
    parser.add_argument('--max-experiments', type=int, default=None,
                       help='Maximum number of experiments to run')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Base log directory')
    
    args = parser.parse_args()
    
    # Grid Search 실행
    run_grid_search(
        env_name=args.env,
        base_config_path=args.config,
        grid_type=args.grid_type,
        num_workers=args.workers,
        log_dir_base=args.log_dir,
        max_experiments=args.max_experiments
    )


if __name__ == '__main__':
    # Windows에서 multiprocessing을 위한 설정
    if sys.platform == 'win32':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # 이미 설정된 경우 무시
            pass
    
    main()


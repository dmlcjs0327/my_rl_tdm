"""
구현 검증을 위한 평가 지표 및 테스트 스크립트
"""
import os
import sys
import yaml
import json
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
import traceback

# 프로젝트 모듈 import
from hyperparameter_grid import (
    generate_hyperparameter_combinations,
    create_config_from_hyperparameters
)
from curriculum_learning import CurriculumLearning, WarmUpPeriod
from env_wrapper import GoalSampler
from train_with_curriculum import load_config, create_env


class ImplementationValidator:
    """구현 검증을 위한 평가 지표 클래스"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.warnings = []
    
    def validate_hyperparameter_grid(self) -> Dict[str, Any]:
        """하이퍼파라미터 그리드 검증"""
        print("\n" + "="*60)
        print("1. 하이퍼파라미터 그리드 검증")
        print("="*60)
        
        results = {
            'passed': True,
            'tests': [],
            'errors': []
        }
        
        try:
            # Test 1: 그리드 생성
            print("\n[Test 1.1] 그리드 생성 테스트...")
            combinations = generate_hyperparameter_combinations('Reacher-v5', 'minimal')
            assert len(combinations) > 0, "조합이 생성되지 않음"
            print(f"  ✓ {len(combinations)}개 조합 생성됨")
            results['tests'].append({'name': '그리드 생성', 'passed': True})
            
            # Test 2: 조합 유효성
            print("\n[Test 1.2] 조합 유효성 테스트...")
            for i, combo in enumerate(combinations[:5]):  # 처음 5개만 테스트
                required_keys = ['learning_rate_actor', 'learning_rate_critic', 'tau_max']
                for key in required_keys:
                    assert key in combo, f"필수 키 {key}가 없음"
                    assert isinstance(combo[key], (int, float)), f"{key}가 숫자가 아님"
            print(f"  ✓ 조합 유효성 확인됨")
            results['tests'].append({'name': '조합 유효성', 'passed': True})
            
            # Test 3: 설정 생성
            print("\n[Test 1.3] 설정 생성 테스트...")
            base_config = load_config('config.yaml')
            test_hyperparams = combinations[0]
            config = create_config_from_hyperparameters(base_config, test_hyperparams)
            
            assert config['training']['learning_rate_actor'] == test_hyperparams['learning_rate_actor']
            assert config['tdm']['tau_max'] == test_hyperparams['tau_max']
            print(f"  ✓ 설정 생성 확인됨")
            results['tests'].append({'name': '설정 생성', 'passed': True})
            
            # Test 4: 환경별 그리드
            print("\n[Test 1.4] 환경별 그리드 테스트...")
            envs = ['Reacher-v5', 'Pusher-v5', 'HalfCheetah-v5', 'Ant-v5']
            for env in envs:
                combos = generate_hyperparameter_combinations(env, 'minimal')
                assert len(combos) > 0, f"{env}에 대한 그리드가 생성되지 않음"
            print(f"  ✓ 모든 환경에 대한 그리드 생성 확인됨")
            results['tests'].append({'name': '환경별 그리드', 'passed': True})
            
        except Exception as e:
            print(f"  ✗ 오류: {str(e)}")
            results['passed'] = False
            results['errors'].append(str(e))
            results['tests'].append({'name': '그리드 검증', 'passed': False, 'error': str(e)})
        
        self.test_results['hyperparameter_grid'] = results
        return results
    
    def validate_curriculum_learning(self) -> Dict[str, Any]:
        """Curriculum Learning 검증"""
        print("\n" + "="*60)
        print("2. Curriculum Learning 검증")
        print("="*60)
        
        results = {
            'passed': True,
            'tests': [],
            'errors': []
        }
        
        try:
            # Test 1: Curriculum Learning 초기화
            print("\n[Test 2.1] Curriculum Learning 초기화 테스트...")
            goal_sampler = GoalSampler('end_effector', 'Reacher-v5')
            curriculum = CurriculumLearning(
                goal_sampler,
                initial_difficulty=0.1,
                final_difficulty=1.0,
                curriculum_type='distance',
                schedule='linear'
            )
            assert curriculum.current_difficulty == 0.1, "초기 난이도가 올바르지 않음"
            print(f"  ✓ Curriculum Learning 초기화 확인됨")
            results['tests'].append({'name': '초기화', 'passed': True})
            
            # Test 2: 진행도 업데이트
            print("\n[Test 2.2] 진행도 업데이트 테스트...")
            curriculum.update_progress(0.5)
            assert 0.1 < curriculum.current_difficulty < 1.0, "난이도 업데이트가 올바르지 않음"
            print(f"  ✓ 진행도 업데이트 확인됨 (난이도: {curriculum.current_difficulty:.3f})")
            results['tests'].append({'name': '진행도 업데이트', 'passed': True})
            
            # Test 3: 목표 샘플링
            print("\n[Test 2.3] 목표 샘플링 테스트...")
            initial_state = np.random.randn(10)  # Reacher-v5 observation
            goal = curriculum.sample_goal(initial_state)
            assert goal is not None, "목표가 샘플링되지 않음"
            assert len(goal) == 2, "목표 차원이 올바르지 않음"  # Reacher는 2D goal
            print(f"  ✓ 목표 샘플링 확인됨 (goal shape: {goal.shape})")
            results['tests'].append({'name': '목표 샘플링', 'passed': True})
            
            # Test 4: 스케줄 타입
            print("\n[Test 2.4] 스케줄 타입 테스트...")
            schedules = ['linear', 'exponential', 'step']
            for schedule in schedules:
                curriculum_test = CurriculumLearning(
                    goal_sampler, schedule=schedule
                )
                curriculum_test.update_progress(0.5)
                assert 0.0 <= curriculum_test.current_difficulty <= 1.0
            print(f"  ✓ 모든 스케줄 타입 확인됨")
            results['tests'].append({'name': '스케줄 타입', 'passed': True})
            
        except Exception as e:
            print(f"  ✗ 오류: {str(e)}")
            traceback.print_exc()
            results['passed'] = False
            results['errors'].append(str(e))
            results['tests'].append({'name': 'Curriculum Learning 검증', 'passed': False, 'error': str(e)})
        
        self.test_results['curriculum_learning'] = results
        return results
    
    def validate_warmup(self) -> Dict[str, Any]:
        """Warm-up Period 검증"""
        print("\n" + "="*60)
        print("3. Warm-up Period 검증")
        print("="*60)
        
        results = {
            'passed': True,
            'tests': [],
            'errors': []
        }
        
        try:
            # Test 1: Warm-up 초기화
            print("\n[Test 3.1] Warm-up 초기화 테스트...")
            warmup = WarmUpPeriod(
                warmup_steps=1000,
                initial_noise_std=0.5,
                final_noise_std=0.2
            )
            assert warmup.is_warmup(), "초기에는 warm-up 상태여야 함"
            print(f"  ✓ Warm-up 초기화 확인됨")
            results['tests'].append({'name': '초기화', 'passed': True})
            
            # Test 2: 노이즈 조정
            print("\n[Test 3.2] 노이즈 조정 테스트...")
            warmup.update_step(500)  # 중간 지점
            noise_std = warmup.get_noise_std(0.2)
            assert 0.2 <= noise_std <= 0.5, "노이즈가 올바르게 조정되지 않음"
            print(f"  ✓ 노이즈 조정 확인됨 (noise_std: {noise_std:.3f})")
            results['tests'].append({'name': '노이즈 조정', 'passed': True})
            
            # Test 3: Warm-up 종료
            print("\n[Test 3.3] Warm-up 종료 테스트...")
            warmup.update_step(2000)  # Warm-up 종료 후
            assert not warmup.is_warmup(), "Warm-up이 종료되어야 함"
            noise_std = warmup.get_noise_std(0.2)
            assert noise_std == 0.2, "Warm-up 종료 후 기본 노이즈 사용해야 함"
            print(f"  ✓ Warm-up 종료 확인됨")
            results['tests'].append({'name': 'Warm-up 종료', 'passed': True})
            
            # Test 4: 학습률 배수
            print("\n[Test 3.4] 학습률 배수 테스트...")
            warmup_test = WarmUpPeriod(
                warmup_steps=1000,
                initial_lr_multiplier=0.1,
                final_lr_multiplier=1.0
            )
            warmup_test.update_step(500)
            lr_mult = warmup_test.get_lr_multiplier()
            assert 0.1 <= lr_mult <= 1.0, "학습률 배수가 올바르지 않음"
            print(f"  ✓ 학습률 배수 확인됨 (multiplier: {lr_mult:.3f})")
            results['tests'].append({'name': '학습률 배수', 'passed': True})
            
        except Exception as e:
            print(f"  ✗ 오류: {str(e)}")
            traceback.print_exc()
            results['passed'] = False
            results['errors'].append(str(e))
            results['tests'].append({'name': 'Warm-up 검증', 'passed': False, 'error': str(e)})
        
        self.test_results['warmup'] = results
        return results
    
    def validate_config_integration(self) -> Dict[str, Any]:
        """설정 파일 통합 검증"""
        print("\n" + "="*60)
        print("4. 설정 파일 통합 검증")
        print("="*60)
        
        results = {
            'passed': True,
            'tests': [],
            'errors': []
        }
        
        try:
            # Test 1: 설정 파일 로드
            print("\n[Test 4.1] 설정 파일 로드 테스트...")
            config = load_config('config.yaml')
            assert 'training' in config, "training 설정이 없음"
            assert 'tdm' in config, "tdm 설정이 없음"
            print(f"  ✓ 설정 파일 로드 확인됨")
            results['tests'].append({'name': '설정 파일 로드', 'passed': True})
            
            # Test 2: Curriculum Learning 설정
            print("\n[Test 4.2] Curriculum Learning 설정 테스트...")
            assert 'use_curriculum' in config['training'], "use_curriculum 설정이 없음"
            if config['training'].get('use_curriculum', False):
                assert 'curriculum' in config['training'], "curriculum 설정이 없음"
            print(f"  ✓ Curriculum Learning 설정 확인됨")
            results['tests'].append({'name': 'Curriculum Learning 설정', 'passed': True})
            
            # Test 3: Warm-up 설정
            print("\n[Test 4.3] Warm-up 설정 테스트...")
            assert 'use_warmup' in config['training'], "use_warmup 설정이 없음"
            if config['training'].get('use_warmup', False):
                assert 'warmup' in config['training'], "warmup 설정이 없음"
            print(f"  ✓ Warm-up 설정 확인됨")
            results['tests'].append({'name': 'Warm-up 설정', 'passed': True})
            
            # Test 4: Early Stopping 설정
            print("\n[Test 4.4] Early Stopping 설정 테스트...")
            assert 'patience' in config['training'], "patience 설정이 없음"
            print(f"  ✓ Early Stopping 설정 확인됨")
            results['tests'].append({'name': 'Early Stopping 설정', 'passed': True})
            
        except Exception as e:
            print(f"  ✗ 오류: {str(e)}")
            traceback.print_exc()
            results['passed'] = False
            results['errors'].append(str(e))
            results['tests'].append({'name': '설정 통합 검증', 'passed': False, 'error': str(e)})
        
        self.test_results['config_integration'] = results
        return results
    
    def validate_training_integration(self) -> Dict[str, Any]:
        """학습 통합 검증 (실제 학습 없이 구조만 확인)"""
        print("\n" + "="*60)
        print("5. 학습 통합 검증")
        print("="*60)
        
        results = {
            'passed': True,
            'tests': [],
            'errors': []
        }
        
        try:
            # Test 1: 모듈 import
            print("\n[Test 5.1] 모듈 import 테스트...")
            from train_with_curriculum import train, create_env, evaluate
            from tdm import TDM
            from mpc_planner import TaskSpecificPlanner
            print(f"  ✓ 모든 모듈 import 확인됨")
            results['tests'].append({'name': '모듈 import', 'passed': True})
            
            # Test 2: 환경 생성
            print("\n[Test 5.2] 환경 생성 테스트...")
            config = load_config('config.yaml')
            config['env']['name'] = 'Reacher-v5'  # 간단한 환경 사용
            try:
                env = create_env(config)
                assert env is not None, "환경이 생성되지 않음"
                print(f"  ✓ 환경 생성 확인됨")
                results['tests'].append({'name': '환경 생성', 'passed': True})
            except Exception as e:
                print(f"  ⚠ 환경 생성 실패 (MuJoCo 미설치 가능성): {str(e)}")
                results['tests'].append({'name': '환경 생성', 'passed': False, 'warning': str(e)})
                self.warnings.append(f"환경 생성 실패: {str(e)}")
            
            # Test 3: 설정 구조 검증
            print("\n[Test 5.3] 설정 구조 검증...")
            required_keys = [
                'env', 'task', 'tdm', 'network', 'training', 
                'mpc', 'logging', 'seed'
            ]
            for key in required_keys:
                assert key in config, f"필수 설정 키 {key}가 없음"
            print(f"  ✓ 설정 구조 확인됨")
            results['tests'].append({'name': '설정 구조', 'passed': True})
            
        except Exception as e:
            print(f"  ✗ 오류: {str(e)}")
            traceback.print_exc()
            results['passed'] = False
            results['errors'].append(str(e))
            results['tests'].append({'name': '학습 통합 검증', 'passed': False, 'error': str(e)})
        
        self.test_results['training_integration'] = results
        return results
    
    def validate_file_structure(self) -> Dict[str, Any]:
        """파일 구조 검증"""
        print("\n" + "="*60)
        print("6. 파일 구조 검증")
        print("="*60)
        
        results = {
            'passed': True,
            'tests': [],
            'errors': []
        }
        
        required_files = [
            'hyperparameter_grid.py',
            'curriculum_learning.py',
            'train_with_curriculum.py',
            'grid_search.py',
            'config.yaml',
            'tdm.py',
            'networks.py',
            'replay_buffer.py',
            'env_wrapper.py',
            'mpc_planner.py'
        ]
        
        try:
            print("\n[Test 6.1] 필수 파일 존재 확인...")
            missing_files = []
            for file in required_files:
                if not os.path.exists(file):
                    missing_files.append(file)
                else:
                    print(f"  ✓ {file}")
            
            if missing_files:
                print(f"  ✗ 누락된 파일: {missing_files}")
                results['passed'] = False
                results['errors'].append(f"누락된 파일: {missing_files}")
            else:
                print(f"  ✓ 모든 필수 파일 존재")
            
            results['tests'].append({
                'name': '필수 파일 존재',
                'passed': len(missing_files) == 0,
                'missing_files': missing_files
            })
            
        except Exception as e:
            print(f"  ✗ 오류: {str(e)}")
            results['passed'] = False
            results['errors'].append(str(e))
        
        self.test_results['file_structure'] = results
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        print("\n" + "="*60)
        print("구현 검증 테스트 시작")
        print("="*60)
        
        # 모든 검증 실행
        self.validate_file_structure()
        self.validate_hyperparameter_grid()
        self.validate_curriculum_learning()
        self.validate_warmup()
        self.validate_config_integration()
        self.validate_training_integration()
        
        # 결과 요약
        print("\n" + "="*60)
        print("테스트 결과 요약")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for category, result in self.test_results.items():
            category_passed = result['passed']
            category_tests = len([t for t in result['tests'] if t.get('passed', False)])
            category_total = len(result['tests'])
            
            total_tests += category_total
            passed_tests += category_tests
            failed_tests += (category_total - category_tests)
            
            status = "✓ PASSED" if category_passed else "✗ FAILED"
            print(f"\n{category}: {status}")
            print(f"  테스트: {category_tests}/{category_total} 통과")
            if result['errors']:
                for error in result['errors']:
                    print(f"  오류: {error}")
        
        print(f"\n{'='*60}")
        print(f"전체 결과: {passed_tests}/{total_tests} 테스트 통과")
        if failed_tests > 0:
            print(f"실패: {failed_tests}개")
        if self.warnings:
            print(f"\n경고: {len(self.warnings)}개")
            for warning in self.warnings:
                print(f"  - {warning}")
        print(f"{'='*60}\n")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'warnings': len(self.warnings),
            'results': self.test_results
        }


def main():
    """메인 함수"""
    validator = ImplementationValidator()
    results = validator.run_all_tests()
    
    # 결과를 JSON으로 저장
    output_file = 'test_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"테스트 결과가 {output_file}에 저장되었습니다.")
    
    # 종료 코드
    sys.exit(0 if results['failed_tests'] == 0 else 1)


if __name__ == '__main__':
    main()


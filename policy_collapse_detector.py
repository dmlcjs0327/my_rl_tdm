"""
정책 붕괴(Policy Collapse) 감지 모듈
학습 중 성능이 급격히 나빠지는 것을 감지하고 조기 종료
"""
import numpy as np
from typing import List, Optional, Dict, Any
from collections import deque


class PolicyCollapseDetector:
    """
    정책 붕괴 감지기
    
    정책 붕괴는 다음과 같은 상황을 의미:
    1. 성능이 급격히 나빠짐 (평가 거리가 급격히 증가)
    2. 성공률이 급격히 감소
    3. 학습이 불안정해짐 (변동성이 크게 증가)
    """
    
    def __init__(self,
                 window_size: int = 5,
                 collapse_threshold: float = 0.3,
                 min_evaluations: int = 3,
                 stability_threshold: float = 0.5):
        """
        Args:
            window_size: 성능 추적을 위한 윈도우 크기
            collapse_threshold: 붕괴로 간주하는 성능 저하 비율 (0.3 = 30% 저하)
            min_evaluations: 최소 평가 횟수 (이전에는 붕괴 판단 안 함)
            stability_threshold: 불안정성 임계값 (변동 계수)
        """
        self.window_size = window_size
        self.collapse_threshold = collapse_threshold
        self.min_evaluations = min_evaluations
        self.stability_threshold = stability_threshold
        
        # 성능 히스토리
        self.distance_history = deque(maxlen=window_size)
        self.success_rate_history = deque(maxlen=window_size)
        self.best_distance = float('inf')
        self.best_success_rate = 0.0
        
        # 평가 횟수
        self.evaluation_count = 0
        
        # 붕괴 상태
        self.is_collapsed = False
        self.collapse_reason = None
    
    def update(self, mean_distance: float, success_rate: float) -> Dict[str, Any]:
        """
        새로운 평가 결과 업데이트 및 붕괴 감지
        
        Args:
            mean_distance: 평균 거리 (낮을수록 좋음)
            success_rate: 성공률 (높을수록 좋음)
        
        Returns:
            {
                'is_collapsed': bool,
                'reason': str,
                'should_stop': bool,
                'metrics': {...}
            }
        """
        self.evaluation_count += 1
        self.distance_history.append(mean_distance)
        self.success_rate_history.append(success_rate)
        
        # 최고 성능 업데이트
        if mean_distance < self.best_distance:
            self.best_distance = mean_distance
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
        
        # 최소 평가 횟수 이전에는 판단 안 함
        if self.evaluation_count < self.min_evaluations:
            return {
                'is_collapsed': False,
                'reason': None,
                'should_stop': False,
                'metrics': self._get_metrics()
            }
        
        # 붕괴 감지
        collapse_info = self._detect_collapse(mean_distance, success_rate)
        
        return collapse_info
    
    def _detect_collapse(self, current_distance: float, current_success_rate: float) -> Dict[str, Any]:
        """
        정책 붕괴 감지 로직
        """
        metrics = self._get_metrics()
        
        # 1. 성능 급격 저하 감지
        if self.best_distance > 0:
            performance_degradation = (current_distance - self.best_distance) / self.best_distance
            if performance_degradation > self.collapse_threshold:
                self.is_collapsed = True
                self.collapse_reason = f"Performance degradation: {performance_degradation:.2%} worse than best"
                return {
                    'is_collapsed': True,
                    'reason': self.collapse_reason,
                    'should_stop': True,
                    'metrics': metrics
                }
        
        # 2. 성공률 급격 감소 감지
        if self.best_success_rate > 0:
            success_rate_drop = (self.best_success_rate - current_success_rate) / max(self.best_success_rate, 0.1)
            if success_rate_drop > self.collapse_threshold:
                self.is_collapsed = True
                self.collapse_reason = f"Success rate drop: {success_rate_drop:.2%} worse than best"
                return {
                    'is_collapsed': True,
                    'reason': self.collapse_reason,
                    'should_stop': True,
                    'metrics': metrics
                }
        
        # 3. 최근 성능 추세 분석 (급격한 악화)
        if len(self.distance_history) >= 3:
            recent_distances = list(self.distance_history)[-3:]
            # 최근 3회 평가 중 계속 나빠지는 경우
            if (recent_distances[-1] > recent_distances[-2] > recent_distances[-3] and
                (recent_distances[-1] - recent_distances[-3]) / max(recent_distances[-3], 0.1) > self.collapse_threshold):
                self.is_collapsed = True
                self.collapse_reason = "Continuous performance degradation in recent evaluations"
                return {
                    'is_collapsed': True,
                    'reason': self.collapse_reason,
                    'should_stop': True,
                    'metrics': metrics
                }
        
        # 4. 불안정성 감지 (변동성이 너무 큰 경우)
        if len(self.distance_history) >= self.window_size:
            distances_array = np.array(self.distance_history)
            cv = np.std(distances_array) / (np.mean(distances_array) + 1e-8)  # 변동 계수
            if cv > self.stability_threshold:
                # 불안정하지만 붕괴는 아님 (경고만)
                return {
                    'is_collapsed': False,
                    'reason': f"High instability detected (CV: {cv:.2f})",
                    'should_stop': False,
                    'metrics': metrics
                }
        
        return {
            'is_collapsed': False,
            'reason': None,
            'should_stop': False,
            'metrics': metrics
        }
    
    def _get_metrics(self) -> Dict[str, Any]:
        """현재 메트릭 반환"""
        if len(self.distance_history) == 0:
            return {}
        
        distances = np.array(self.distance_history)
        success_rates = np.array(self.success_rate_history)
        
        return {
            'current_distance': distances[-1] if len(distances) > 0 else None,
            'best_distance': self.best_distance,
            'current_success_rate': success_rates[-1] if len(success_rates) > 0 else None,
            'best_success_rate': self.best_success_rate,
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'mean_success_rate': np.mean(success_rates),
            'evaluation_count': self.evaluation_count
        }
    
    def reset(self):
        """감지기 리셋"""
        self.distance_history.clear()
        self.success_rate_history.clear()
        self.best_distance = float('inf')
        self.best_success_rate = 0.0
        self.evaluation_count = 0
        self.is_collapsed = False
        self.collapse_reason = None


"""
PBT 학습 진행 상황 모니터링 유틸리티
"""
import os
import json
import time
from typing import Dict, List, Any
from datetime import datetime


class PBTMonitor:
    """PBT 학습 진행 상황 모니터"""
    
    def __init__(self, log_dir_base: str):
        self.log_dir_base = log_dir_base
        self.status_file = os.path.join(log_dir_base, 'pbt_status.json')
        self.last_update = {}
    
    def update_member_status(self, member_id: int, status: Dict[str, Any]):
        """개체의 상태 업데이트 (각 프로세스에서 호출)"""
        # 파일 기반 상태 공유 (멀티프로세싱 안전)
        status_data = {}
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
            except:
                pass
        
        status_data[f'member_{member_id:02d}'] = {
            **status,
            'timestamp': datetime.now().isoformat()
        }
        
        # 파일 쓰기 (간단한 락 없이, 충돌 가능성 있지만 큰 문제는 아님)
        try:
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2)
        except:
            pass  # 다른 프로세스가 쓰는 중일 수 있음
    
    def get_all_status(self) -> Dict[str, Any]:
        """모든 개체의 상태 읽기"""
        if not os.path.exists(self.status_file):
            return {}
        
        try:
            with open(self.status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def print_status(self, population_size: int, total_timesteps: int):
        """현재 상태 출력"""
        status_data = self.get_all_status()
        
        if not status_data:
            return
        
        print(f"\n{'='*80}")
        print(f"PBT Training Status - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        
        # 헤더
        print(f"{'Member':<8} {'Steps':<12} {'Progress':<12} {'Best Dist':<12} {'Current Dist':<12} {'Status':<15}")
        print(f"{'-'*80}")
        
        # 각 개체 상태
        for i in range(population_size):
            member_key = f'member_{i:02d}'
            if member_key in status_data:
                status = status_data[member_key]
                steps = status.get('training_steps', 0)
                progress = (steps / total_timesteps * 100) if total_timesteps > 0 else 0
                best_dist = status.get('best_distance', float('inf'))
                current_dist = status.get('current_distance', float('inf'))
                state = status.get('state', 'training')
                early_stopped = status.get('early_stopped', False)
                early_stop_reason = status.get('early_stop_reason', '')
                
                # 상태 표시
                if early_stopped:
                    state_str = '⏸️  Early Stop'
                elif state == 'training':
                    state_str = '🔄 Training'
                elif state == 'evaluating':
                    state_str = '📊 Evaluating'
                elif state == 'completed':
                    state_str = '✅ Completed'
                elif state == 'error':
                    state_str = '❌ Error'
                else:
                    state_str = state
                
                # 거리 표시
                best_str = f"{best_dist:.4f}" if best_dist != float('inf') else "N/A"
                current_str = f"{current_dist:.4f}" if current_dist != float('inf') else "N/A"
                
                # 조기 종료 사유 표시
                if early_stopped and early_stop_reason:
                    state_str += f" ({early_stop_reason[:20]}...)" if len(early_stop_reason) > 20 else f" ({early_stop_reason})"
                
                print(f"{i:02d}      {steps:<12} {progress:>6.1f}%      {best_str:<12} {current_str:<12} {state_str:<15}")
            else:
                print(f"{i:02d}      {'-':<12} {'-':<12} {'-':<12} {'-':<12} {'Waiting':<15}")
        
        print(f"{'='*80}\n")


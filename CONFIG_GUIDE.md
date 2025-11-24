# 환경별 설정 파일 가이드

이 프로젝트는 논문에서 사용된 4개 환경에 대해 각각 최적화된 설정 파일을 제공합니다.

## 설정 파일 목록

| 환경 | 설정 파일 | 설명 |
|------|-----------|------|
| **Reacher-v5** | `config_reacher.yaml` | 7-DoF 로봇 팔로 목표 위치 도달 |
| **Pusher-v5** | `config_pusher.yaml` | 퍽을 목표 위치로 밀기 (2단계 작업) |
| **HalfCheetah-v5** | `config_halfcheetah.yaml` | 목표 속도로 달리기 |
| **Ant-v5** | `config_ant.yaml` | 목표 위치로 이동 (복잡한 dynamics) |

## 사용 방법

### 방법 1: 설정 파일 직접 지정

```bash
# Pusher-v5로 학습
python train_with_curriculum.py --config config_pusher.yaml

# HalfCheetah-v5로 학습
python train_with_curriculum.py --config config_halfcheetah.yaml

# Ant-v5로 학습
python train_with_curriculum.py --config config_ant.yaml
```

### 방법 2: config.yaml 수정

`config.yaml` 파일에서 환경 이름만 변경:

```yaml
env:
  name: "Pusher-v5"  # 원하는 환경으로 변경
```

그리고 기본 설정을 해당 환경에 맞게 조정하세요.

## 환경별 하이퍼파라미터 특징

### Reacher-v5
- **복잡도**: 중간
- **tau_max**: 20 (15-25 범위)
- **학습률**: Actor 0.0001, Critic 0.001
- **특징**: 가장 간단한 환경, 빠른 수렴

### Pusher-v5
- **복잡도**: 높음 (2단계 작업)
- **tau_max**: 25 (20-30 범위)
- **학습률**: Actor 0.00005, Critic 0.0005 (더 낮음)
- **특징**: 손 도달 → 퍽 밀기의 2단계 작업, 더 많은 학습 필요

### HalfCheetah-v5
- **복잡도**: 중간
- **tau_max**: 20 (15-25 범위)
- **학습률**: Actor 0.0001, Critic 0.001
- **특징**: 속도 제어, 더 긴 에피소드, 더 큰 배치 크기

### Ant-v5
- **복잡도**: 매우 높음
- **tau_max**: 25 (20-30 범위)
- **학습률**: Actor 0.00005, Critic 0.0005 (더 낮음)
- **특징**: 가장 복잡한 dynamics, 더 큰 네트워크, 더 많은 학습 필요

## 하이퍼파라미터 설정 원칙

각 환경별 설정은 다음 원칙에 따라 결정되었습니다:

1. **논문 기반**: 논문에서 언급된 일반적인 범위 참고
2. **환경 복잡도**: 복잡한 환경일수록 더 낮은 학습률, 더 긴 계획 지평선
3. **일반적인 강화학습 관행**: 각 환경에서 일반적으로 사용되는 값
4. **Grid Search 기반**: `hyperparameter_grid.py`의 환경별 그리드 참고

## 주의사항

⚠️ **이 설정들은 초기값입니다!**

- 실제 최적 하이퍼파라미터는 환경, 하드웨어, 시드에 따라 다를 수 있습니다
- **Grid Search를 통해 최적값을 찾는 것을 강력히 권장합니다**:

```bash
# 각 환경에 대해 Grid Search 실행
python grid_search.py --env Reacher-v5 --grid-type reduced
python grid_search.py --env Pusher-v5 --grid-type reduced
python grid_search.py --env HalfCheetah-v5 --grid-type reduced
python grid_search.py --env Ant-v5 --grid-type reduced
```

Grid Search 결과로 찾은 최적 하이퍼파라미터를 사용하면 더 나은 성능을 얻을 수 있습니다.

## 환경별 주요 차이점

### 1. 계획 지평선 (tau_max)
- **Reacher**: 20 (중간 복잡도)
- **Pusher**: 25 (복잡한 2단계 작업)
- **HalfCheetah**: 20 (중간 복잡도)
- **Ant**: 25 (복잡한 dynamics)

### 2. 학습률
- **Reacher/HalfCheetah**: 중간 학습률 (0.0001/0.001)
- **Pusher/Ant**: 낮은 학습률 (0.00005/0.0005) - 복잡한 작업

### 3. 배치 크기
- **Reacher/Pusher**: 128
- **HalfCheetah/Ant**: 256 (더 큰 상태/행동 공간)

### 4. 총 학습 스텝
- **Reacher**: 1,000,000
- **Pusher**: 1,500,000
- **HalfCheetah/Ant**: 2,000,000

### 5. 네트워크 크기
- **Reacher/Pusher/HalfCheetah**: [300, 300]
- **Ant**: [400, 400] (더 복잡한 작업)

## 커스터마이징

각 설정 파일을 직접 수정하여 실험할 수 있습니다:

```yaml
# 예: 학습률 조정
training:
  learning_rate_actor: 0.0002  # 기본값에서 변경
  
# 예: Curriculum Learning 비활성화
training:
  use_curriculum: false
```

변경 후 학습을 실행하면 새로운 설정이 적용됩니다.


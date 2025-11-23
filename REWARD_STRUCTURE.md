# TDM 보상 구조 분석 및 개선안

## 📊 현재 보상 구조

### 현재 구현

```python
def compute_reward(self, state, goal):
    current_goal = self.goal_extractor(state)
    distance = np.abs(current_goal - goal).sum()
    return -distance  # 단순히 음수 거리
```

### 문제점

1. **Mean Reward와 Mean Distance가 중복**
   - Mean Reward = -Mean Distance
   - 동일한 정보를 두 번 로깅
   - 불필요한 중복

2. **보상의 의미가 모호함**
   - 음수 보상은 직관적이지 않음
   - 거리 기반 보상만으로는 학습이 어려울 수 있음

## 💡 개선 방안

### 방안 1: 거리 기반 보상 유지 (현재)

**장점**:
- 논문과 일치
- 단순하고 명확

**단점**:
- Mean Reward와 Mean Distance 중복
- 음수 보상으로 인한 직관성 부족

**개선**:
- Mean Reward 로깅 제거
- Mean Distance만 사용

### 방안 2: 거리 기반 보상 + 성공 보너스

```python
def compute_reward(self, state, goal):
    current_goal = self.goal_extractor(state)
    distance = np.abs(current_goal - goal).sum()
    
    # 기본 보상: 음수 거리
    reward = -distance
    
    # 성공 보너스
    if distance < self.success_threshold:
        reward += self.success_bonus
    
    return reward
```

**장점**:
- 성공 시 긍정적 보상
- 더 명확한 학습 신호
- Mean Reward와 Mean Distance가 다른 의미

**단점**:
- 논문과 다를 수 있음
- 추가 하이퍼파라미터 필요

### 방안 3: 스케일된 보상

```python
def compute_reward(self, state, goal):
    current_goal = self.goal_extractor(state)
    distance = np.abs(current_goal - goal).sum()
    
    # 보상 스케일링
    reward = -distance * self.reward_scale
    
    return reward
```

**장점**:
- 보상 범위 조정 가능
- 학습 안정성 향상

**단점**:
- 여전히 중복 문제 존재

## 🎯 권장 개선안

### 옵션 1: 로깅만 개선 (간단)

Mean Reward 로깅을 제거하고 Mean Distance만 사용:

```python
# 제거
writer.add_scalar('eval/mean_reward', ...)

# 유지
writer.add_scalar('eval/mean_distance', ...)
```

### 옵션 2: 보상 구조 개선 (권장)

거리 기반 보상에 성공 보너스 추가:

```python
def compute_reward(self, state, goal):
    current_goal = self.goal_extractor(state)
    distance = np.abs(current_goal - goal).sum()
    
    # 기본 보상: 음수 거리
    reward = -distance * self.reward_scale
    
    # 성공 보너스 (선택사항)
    if distance < 0.1:
        reward += 10.0
    
    return reward
```

**효과**:
- Mean Reward: 성공 시 양수, 실패 시 음수
- Mean Distance: 목표까지의 실제 거리
- 두 메트릭이 서로 다른 정보 제공

## 📝 논문과의 비교

### 논문의 보상 구조

논문에서는 TDM의 보상 함수를 명확히 정의하지 않았습니다. 하지만:

1. **TDM Q-function**:
   ```
   Q(s, a, g, τ) = -||f(s, a, g, τ) - g||
   ```
   - 음수 거리 사용

2. **실제 task reward**:
   - 논문에서는 task-specific reward 사용
   - 예: Reacher는 end-effector와 목표의 거리

### 현재 구현의 문제

현재 구현은 TDM의 내부 보상과 task reward를 동일하게 사용하고 있습니다:

```python
# TDM 내부: 거리 기반
Q(s, a, g, τ) = -||f(s, a, g, τ) - g||

# Task reward: 동일하게 거리 기반
reward = -distance(s, goal)
```

## 🔧 구체적인 개선 제안

### 1. 로깅 개선 (즉시 적용 가능)

```python
# train.py, evaluate.py에서
# Mean Reward 로깅 제거
# Mean Distance만 사용
```

### 2. 보상 구조 개선 (선택사항)

```python
# env_wrapper.py
def compute_reward(self, state, goal):
    current_goal = self.goal_extractor(state)
    distance = np.abs(current_goal - goal).sum()
    
    # 기본 보상: 음수 거리 (스케일링)
    reward = -distance * self.reward_scale
    
    # 성공 보너스 (선택사항)
    if distance < 0.1:
        reward += 10.0
    
    return reward
```

## 📊 메트릭 비교

### 현재 (중복)

| 메트릭 | 값 | 의미 |
|--------|-----|------|
| Mean Reward | -5.0 | 평균 보상 |
| Mean Distance | 5.0 | 평균 거리 |
| **관계** | **-1배** | **동일한 정보** |

### 개선 후 (중복 제거)

| 메트릭 | 값 | 의미 |
|--------|-----|------|
| ~~Mean Reward~~ | ~~-5.0~~ | ~~제거됨~~ |
| Mean Distance | 5.0 | 평균 거리 |
| Success Rate | 0.8 | 성공률 |

### 개선 후 (보상 구조 변경)

| 메트릭 | 값 | 의미 |
|--------|-----|------|
| Mean Reward | 2.0 | 평균 보상 (성공 시 양수) |
| Mean Distance | 5.0 | 평균 거리 |
| Success Rate | 0.8 | 성공률 |

## 🎯 결론

사용자의 지적이 정확합니다:

1. **현재 문제**: Mean Reward = -Mean Distance (중복)
2. **해결 방안**:
   - 옵션 1: Mean Reward 로깅 제거 (간단)
   - 옵션 2: 보상 구조 개선 (더 나은 학습)

어떤 방향으로 개선할까요?








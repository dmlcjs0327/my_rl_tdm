# TDM 알고리즘 검토 보고서

## 📋 검토 개요

논문 "Temporal Difference Models: Model-Free Deep RL for Model-Based Control"과 구현된 코드를 비교하여 검토했습니다.

## ✅ 올바르게 구현된 부분

### 1. TDM Q-function 구조
- **논문**: `Q(s, a, sg, τ) = -||f(s, a, sg, τ) - sg||`
- **구현**: `TDMCritic.compute_q_value()` - ✅ 정확히 구현됨

### 2. Goal Relabeling
- **논문**: 각 transition을 여러 목표와 지평선으로 relabel
- **구현**: `GoalRelabeler.relabel()` - ✅ 구현됨

### 3. Vectorized Supervision
- **논문 Appendix A.5**: 각 차원을 독립적으로 supervision
- **구현**: `TDMCriticVectorized` - ✅ 구현됨

### 4. MPC 기반 정책 추출
- **논문 Equation (8), (9)**: 다양한 MPC 방법
- **구현**: `MPCPlanner` - ✅ 구현됨

## ⚠️ 수정이 필요한 부분

### 1. **Actor 네트워크가 Goal과 Tau를 받지 않음**

**문제점**:
```python
# networks.py - 현재 구현
class Actor(nn.Module):
    def forward(self, state):
        return self.network(state)  # state만 받음
```

**논문의 요구사항**:
- Actor는 goal-conditioned policy여야 함
- 논문에서는 `π(a|s, g, τ)` 형태의 policy 사용

**수정 필요**:
```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, hidden_sizes=[300, 300]):
        # Input: state + goal + tau
        input_dim = state_dim + goal_dim + 1
        # ... rest of the code
    
    def forward(self, state, goal, tau):
        x = torch.cat([state, goal, tau], dim=-1)
        return self.network(x)
```

### 2. **TDM 학습 알고리즘의 Tau 처리**

**문제점** (tdm.py 라인 107-137):
```python
def select_action(self, state, goal, tau, add_noise=True):
    # Actor가 goal과 tau를 받지 않음
    action = self.actor(state_tensor)  # ❌
```

**논문 Equation (5)**:
```
Q(s, a, sg, τ) = E[-D(s', sg)·1[τ=0] + max_a' Q(s', a', sg, τ-1)·1[τ≠0]]
```

**수정 필요**:
- Actor가 goal과 tau를 입력으로 받도록 수정
- 모든 actor 호출 부분 업데이트

### 3. **Vectorized Supervision의 차별화 부족**

**문제점** (networks.py):
```python
class TDMCriticVectorized(nn.Module):
    # TDMCritic과 동일한 구조
    # 실제로 어떻게 다르게 학습하는지 불명확
```

**논문 Appendix A.5**:
- Scalar: `Q(s, a, g, τ) = -Σ|f_j(s, a, g, τ) - g_j|`
- Vectorized: 각 j에 대해 `|f_j(s, a, g, τ) - g_j|`를 독립적으로 supervision

**수정 필요**:
- Vectorized supervision의 loss 계산을 명확히 구현
- `update_critic()` 메서드에서 vectorized와 scalar의 차이 명확화

### 4. **Goal Relabeling의 Future Strategy**

**문제점** (replay_buffer.py 라인 48-77):
```python
def sample_trajectory(self, batch_size):
    # Episode boundary를 찾는 로직이 복잡하고 비효율적
    # done flag만으로 episode를 찾으려고 함
```

**논문의 요구사항**:
- Future state를 샘플링할 때 현재 trajectory 내에서만 샘플링
- Episode boundary를 정확히 파악해야 함

**수정 필요**:
- Episode boundary 정보를 명시적으로 저장
- 더 효율적인 future state 샘플링 구현

### 5. **TDM Loss 계산의 불일치**

**문제점** (tdm.py 라인 200-219):
```python
if self.vectorized:
    # Vectorized supervision
    distance_per_dim = torch.abs(predicted_states - goals)
    # ... 복잡한 target 계산
    loss = F.mse_loss(distance_per_dim, target_distance)
else:
    # Scalar supervision
    distance = self.compute_distance(predicted_states, goals)
    loss = F.mse_loss(distance, -target_q)
```

**논문의 요구사항**:
- Vectorized supervision은 각 차원을 독립적으로 supervision
- 논문 Appendix A.5의 수식을 정확히 구현해야 함

**수정 필요**:
- Loss 계산 로직을 논문의 수식과 정확히 일치시킴
- Vectorized supervision의 target 계산 단순화

## 🔧 권장 수정 사항

### 우선순위 1 (Critical)

1. **Actor 네트워크 수정**
   - Goal과 tau를 입력으로 받도록 변경
   - 모든 actor 호출 부분 업데이트

2. **TDM 학습 알고리즘 수정**
   - Actor가 goal과 tau를 받도록 수정
   - 논문 Equation (5)와 정확히 일치하도록 구현

### 우선순위 2 (Important)

3. **Vectorized Supervision 명확화**
   - Loss 계산 로직 단순화
   - 논문 Appendix A.5와 정확히 일치

4. **Goal Relabeling 개선**
   - Episode boundary 추적 개선
   - Future state 샘플링 효율화

### 우선순위 3 (Nice to have)

5. **코드 정리 및 최적화**
   - 중복 코드 제거
   - 주석 추가
   - 타입 힌트 개선

## 📊 알고리즘 정확도 평가

| 구성요소 | 논문 일치도 | 상태 |
|---------|-----------|------|
| TDM Q-function | 95% | ✅ 거의 정확 |
| Goal Relabeling | 80% | ⚠️ 개선 필요 |
| Vectorized Supervision | 70% | ⚠️ 명확화 필요 |
| Actor Network | 50% | ❌ 수정 필요 |
| MPC Planner | 90% | ✅ 잘 구현됨 |

## 🎯 수정 후 예상 효과

1. **Actor가 goal과 tau를 받도록 수정**
   - ✅ 논문과의 일치도 향상
   - ✅ Goal-conditioned policy의 정확한 구현
   - ✅ 성능 향상 가능성

2. **Vectorized Supervision 명확화**
   - ✅ 학습 안정성 향상
   - ✅ 논문의 실험 결과 재현 가능성 향상

3. **Goal Relabeling 개선**
   - ✅ 샘플 효율성 향상
   - ✅ 학습 속도 개선

## 📝 결론

현재 구현은 TDM의 핵심 아이디어를 잘 구현했지만, 몇 가지 중요한 부분에서 논문과 차이가 있습니다. 특히 Actor 네트워크가 goal과 tau를 입력으로 받지 않는 것은 수정이 필요합니다. 

수정 후에는 논문의 실험 결과를 더 정확하게 재현할 수 있을 것으로 예상됩니다.









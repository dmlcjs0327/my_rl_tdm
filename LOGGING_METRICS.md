# TDM 로깅 메트릭 설명

## 📊 로깅되는 메트릭 개요

TDM 훈련 및 평가 중 로깅되는 모든 메트릭에 대한 상세 설명입니다.

## 🏃 훈련 중 로깅 메트릭

### 1. Critic Loss (`train/critic_loss`)

**정의**: TDM Critic 네트워크의 손실값

**수식**:
- **Vectorized supervision**:
  ```
  Loss = MSE(|f(s,a,g,τ) - g|, target_distance)
  ```
- **Scalar supervision**:
  ```
  Loss = MSE(||f(s,a,g,τ) - g||, -target_Q)
  ```

**의미**:
- Critic이 goal-conditioned value function을 얼마나 잘 학습하는지 측정
- 낮을수록 좋음 (0에 가까울수록 정확)
- 일반적으로 초기에는 높고, 학습이 진행되면서 감소

**기대 동작**:
```
초기: ~100-1000
중기: ~10-100
후기: ~0.1-10
```

### 2. Actor Loss (`train/actor_loss`)

**정의**: Actor 네트워크의 손실값 (policy gradient)

**수식**:
```
Loss = -Q(s, π(s,g,τ), g, τ)
```

**의미**:
- Actor가 Q-value를 최대화하려는 정도
- **음수**가 정상 (Q-value를 최대화하려고 하므로)
- 절댓값이 클수록 Actor가 더 큰 Q-value를 얻으려고 함

**기대 동작**:
```
초기: ~-10 ~ -100
중기: ~-1 ~ -10
후기: ~-0.1 ~ -1
```

### 3. Episode Reward (`train/episode_reward`)

**정의**: 한 에피소드 동안 받은 총 보상

**수식**:
```
Episode Reward = Σ reward_t
```

**의미**:
- Goal에 얼마나 가까워졌는지 측정
- TDM에서는 거리 기반 보상 사용: `reward = -distance(s, goal)`
- **음수**가 정상 (거리가 멀수록 더 큰 음수)
- 0에 가까울수록 목표에 도달한 것

**기대 동작**:
```
초기: ~-100 ~ -1000
중기: ~-10 ~ -100
후기: ~-0.1 ~ -10
```

### 4. Episode Length (`train/episode_length`)

**정의**: 한 에피소드의 길이 (스텝 수)

**의미**:
- 목표에 도달하는데 걸린 시간
- 짧을수록 빠르게 목표 도달
- `max_episode_steps`에 도달하면 시간 초과

**기대 동작**:
```
초기: max_episode_steps (시간 초과)
중기: max_episode_steps ~ max_episode_steps/2
후기: ~10-50 (빠른 도달)
```

### 5. Noise Std (`train/noise_std`)

**정의**: 탐험(exploration)에 사용되는 노이즈의 표준편차

**수식**:
```
noise_std = initial_noise * (decay_rate ^ (step / decay_steps))
```

**의미**:
- 탐험 vs 활용(exploitation)의 균형
- 초기에는 높아서 다양한 행동 탐험
- 학습이 진행되면서 감소하여 학습된 정책 활용

**기대 동작**:
```
초기: 0.2 (설정값)
중기: ~0.05-0.1
후기: ~0.01-0.05
```

## 📈 평가 중 로깅 메트릭

### 1. Mean Reward (`eval/mean_reward`)

**정의**: 평가 에피소드들의 평균 보상

**수식**:
```
Mean Reward = (1/N) * Σ episode_reward_i
```

**의미**:
- 학습된 정책의 성능
- 높을수록(0에 가까울수록) 좋음
- 훈련 중 가장 중요한 지표

**기대 동작**:
```
초기: ~-100 ~ -1000
중기: ~-10 ~ -100
후기: ~-0.1 ~ -10
```

### 2. Mean Distance (`eval/mean_distance`)

**정의**: 목표까지의 평균 거리

**수식**:
```
Mean Distance = (1/N) * Σ ||s_final - goal||
```

**의미**:
- 목표 도달 정확도
- 0에 가까울수록 정확
- Success threshold (0.1) 이하면 성공

**기대 동작**:
```
초기: ~10-100
중기: ~1-10
후기: ~0.01-1
```

### 3. Success Rate (`eval/success_rate`)

**정의**: 목표 도달 성공률

**수식**:
```
Success Rate = (성공한 에피소드 수) / (전체 에피소드 수)
```

**의미**:
- 정책이 목표를 얼마나 자주 달성하는지
- 0.0 ~ 1.0 범위
- 1.0에 가까울수록 좋음

**기대 동작**:
```
초기: ~0.0-0.2
중기: ~0.3-0.7
후기: ~0.8-1.0
```

### 4. Mean Length (`eval/mean_length`)

**정의**: 평가 에피소드들의 평균 길이

**의미**:
- 목표 도달에 필요한 평균 시간
- 짧을수록 효율적

**기대 동작**:
```
초기: max_episode_steps
중기: max_episode_steps/2
후기: ~10-50
```

## 📉 TensorBoard에서 확인하는 방법

### 1. TensorBoard 실행

```bash
tensorboard --logdir=./logs
```

### 2. 주요 그래프

#### Training Metrics
- **Episode Reward**: 점점 증가 (0에 가까워짐)
- **Episode Length**: 점점 감소
- **Critic Loss**: 점점 감소
- **Actor Loss**: 점점 감소 (절댓값)
- **Noise Std**: 점점 감소

#### Evaluation Metrics
- **Mean Reward**: 점점 증가 (0에 가까워짐)
- **Mean Distance**: 점점 감소 (0에 가까워짐)
- **Success Rate**: 점점 증가 (1.0에 가까워짐)

## 🎯 정상적인 학습 패턴

### 초기 단계 (0-10% 훈련)
- Critic Loss: 높음 (~100-1000)
- Actor Loss: 높음 (절댓값 ~10-100)
- Episode Reward: 매우 낮음 (~-1000)
- Success Rate: ~0%
- Noise Std: 높음 (0.2)

### 중기 단계 (10-50% 훈련)
- Critic Loss: 감소 (~10-100)
- Actor Loss: 감소 (절댓값 ~1-10)
- Episode Reward: 개선 (~-100)
- Success Rate: ~30-70%
- Noise Std: 감소 (0.05-0.1)

### 후기 단계 (50-100% 훈련)
- Critic Loss: 낮음 (~0.1-10)
- Actor Loss: 낮음 (절댓값 ~0.1-1)
- Episode Reward: 좋음 (~-10)
- Success Rate: ~80-100%
- Noise Std: 낮음 (0.01-0.05)

## ⚠️ 비정상적인 패턴

### 1. Critic Loss가 계속 증가
- **원인**: Learning rate가 너무 높음
- **해결**: `learning_rate_critic` 감소 (0.001 → 0.0001)

### 2. Actor Loss가 0에 가까움
- **원인**: Actor가 학습하지 못함
- **해결**: `learning_rate_actor` 증가 (0.0001 → 0.001)

### 3. Success Rate가 0%에서 멈춤
- **원인**: Exploration 부족 또는 task가 너무 어려움
- **해결**: `noise_std` 증가, `tau_max` 조정

### 4. Episode Reward가 변동이 심함
- **원인**: Batch size가 너무 작음
- **해결**: `batch_size` 증가 (128 → 256)

## 🔧 로깅 빈도 설정

`config.yaml`에서 조정:

```yaml
logging:
  log_frequency: 100      # 매 100 스텝마다 로깅
  save_frequency: 10000   # 매 10000 스텝마다 모델 저장
  eval_frequency: 5000    # 매 5000 스텝마다 평가
```

## 📊 메트릭 요약

| 메트릭 | 의미 | 좋은 값 | 나쁜 값 |
|--------|------|---------|---------|
| Critic Loss | Critic 학습 정도 | 낮음 (~0.1-10) | 높음 (>100) |
| Actor Loss | Policy 개선 정도 | 적당 (절댓값 ~1-10) | 0에 가까움 |
| Episode Reward | 목표 달성 정도 | 0에 가까움 | 매우 낮음 |
| Success Rate | 성공률 | 1.0 | 0.0 |
| Mean Distance | 목표까지 거리 | 0에 가까움 | 높음 |
| Noise Std | 탐험 정도 | 적당히 감소 | 고정 |

## 🎓 해석 팁

1. **Critic Loss와 Actor Loss가 함께 감소** → 정상적인 학습
2. **Episode Reward가 증가** → 정책이 개선 중
3. **Success Rate가 증가** → 목표 도달 능력 향상
4. **Noise Std가 감소** → 탐험에서 활용으로 전환

이 메트릭들을 모니터링하여 훈련 상태를 파악하고 필요시 하이퍼파라미터를 조정할 수 있습니다.







